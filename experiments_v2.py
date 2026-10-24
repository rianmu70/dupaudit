"""ARS 2026 revision run. Extends the original experiment to answer the reviewers.

Protocols (all stratified five-fold CV, repeated over seeds):
  P0_naive  : StratifiedKFold on the raw file (common practice).
  P1_group  : StratifiedGroupKFold grouped on the FEATURE vector, so every copy of a
              record -- including copies that carry conflicting labels -- stays on one
              side. Same rows as P0. Primary comparator. (R1.2)
  P2_dedup  : drop exact duplicates on (features + label), then StratifiedKFold.
              Leaves conflicting-label records behind, which is measured. (R1.2)
  P2b_dedup_feat : drop duplicates on FEATURES only, then StratifiedKFold. Removes the
              residual channel P2 leaves open. (R1.2)

Per-fold bookkeeping (R1.1): fold sizes and class balance in train and test are recorded
for every fold, so the fold-geometry confound can be measured rather than assumed.

Oversampling (R1.5), three settings on deduplicated data:
  smote_in_fold        : SMOTE inside training folds, real test rows.      (correct)
  smote_pre_resampled  : SMOTE before the split, test fold as resampled.   (what papers report)
  smote_pre_realtest   : SMOTE before the split, test fold filtered to REAL rows only.
                         Isolates leakage from the change in test-set composition.
"""
import glob, hashlib, json, os, platform, time, warnings

import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from dupaudit import audit, binarize_target, clean_frame, make_preprocessor, row_keys
from audit_v2 import feature_space_profile, frame_checksum, residual_overlap_under_dedup

warnings.filterwarnings("ignore")
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_XGB = False

DATASETS = {
    "heart_kaggle_1025": dict(src="kaggle", slug="johnsmith88/heart-disease-dataset",
                              file="heart.csv", target="target"),
    "heart_failure_918": dict(src="kaggle", slug="fedesoriano/heart-failure-prediction",
                              file="heart.csv", target="HeartDisease"),
    "stroke_kaggle": dict(src="kaggle", slug="fedesoriano/stroke-prediction-dataset",
                          file="healthcare-dataset-stroke-data.csv", target="stroke"),
    "cleveland_uci": dict(src="uci", id=45),
    "ckd_uci": dict(src="uci", id=336, positive="ckd"),
    "ilpd_uci": dict(src="uci", id=225, positive=1),
    "early_diabetes_uci": dict(src="uci", id=529, positive="Positive"),
    "maternal_risk_uci": dict(src="uci", id=863, positive="high risk"),
    "lung_cancer_kaggle": dict(src="kaggle", slug="mysarahmadbhat/lung-cancer",
                               file="survey lung cancer.csv", target="LUNG_CANCER",
                               positive="YES"),
}
METRICS = ["acc", "bacc", "f1", "auc"]


# ------------------------------------------------------------------ loading
def load_raw(spec):
    if spec["src"] == "frame":
        return spec["df"].copy(), spec["target"]
    if spec["src"] == "kaggle":
        import kagglehub
        path = kagglehub.dataset_download(spec["slug"])
        f = os.path.join(path, spec["file"])
        if not os.path.exists(f):
            f = sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))[0]
        return pd.read_csv(f), spec["target"]
    if spec["src"] == "uci":
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=spec["id"])
        X = ds.data.features.copy()
        t = ds.data.targets.iloc[:, 0].rename("__target__")
        return pd.concat([X.reset_index(drop=True), t.reset_index(drop=True)], axis=1), "__target__"
    raise ValueError(spec["src"])


def prepare(spec):
    raw, target = load_raw(spec)
    df, dropped = clean_frame(raw, target)
    target = str(target).strip()
    df = df.dropna(subset=[target]).reset_index(drop=True)
    df["y"], info = binarize_target(df[target], spec.get("positive"))
    if target != "y":
        df = df.drop(columns=[target])
    return df, dropped, info


# ------------------------------------------------------------------ models
def get_models(seed):
    m = {"LR": LogisticRegression(max_iter=3000),
         "RF": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
         "kNN": KNeighborsClassifier(n_neighbors=5),
         "SVM": SVC(kernel="rbf", gamma="scale", random_state=seed)}
    m["XGB"] = (XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                              subsample=0.9, eval_metric="logloss", random_state=seed,
                              n_jobs=-1, verbosity=0) if HAS_XGB
                else HistGradientBoostingClassifier(random_state=seed))
    return m


def scores(model, X, y):
    pred = model.predict(X)
    try:
        s = model.predict_proba(X)[:, 1]
    except (AttributeError, NotImplementedError):
        s = model.decision_function(X)
    auc = roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan
    return dict(acc=accuracy_score(y, pred), bacc=balanced_accuracy_score(y, pred),
                f1=f1_score(y, pred, zero_division=0), auc=auc)


# ------------------------------------------------------------------ protocols
def run_protocols(name, df, seeds, folds=5):
    """Returns (score rows, fold-geometry rows)."""
    X = df.drop(columns=["y"]).reset_index(drop=True)
    y = df["y"].values
    keys = row_keys(X)                          # feature-level identity
    groups = pd.factorize(keys)[0]

    ded = df.drop_duplicates().reset_index(drop=True)                     # features+label
    Xd, yd = ded.drop(columns=["y"]), ded["y"].values
    keys_d = row_keys(Xd)

    dedf = df.drop_duplicates(subset=[c for c in df.columns if c != "y"]).reset_index(drop=True)
    Xf, yf = dedf.drop(columns=["y"]), dedf["y"].values
    keys_f = row_keys(Xf)

    rows, geom = [], []
    for seed in seeds:
        plans = {
            "P0_naive": (X, y, keys,
                         StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y)),
            "P1_group": (X, y, keys,
                         StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(X, y, groups)),
            "P2_dedup": (Xd, yd, keys_d,
                         StratifiedKFold(folds, shuffle=True, random_state=seed).split(Xd, yd)),
            "P2b_dedup_feat": (Xf, yf, keys_f,
                               StratifiedKFold(folds, shuffle=True, random_state=seed).split(Xf, yf)),
        }
        mdls = get_models(seed)
        for proto, (XX, yy, kk, splits) in plans.items():
            for fold, (tr, te) in enumerate(splits):
                overlap = float(np.isin(kk[te], kk[tr]).mean())
                geom.append(dict(dataset=name, protocol=proto, seed=seed, fold=fold,
                                 n_train=len(tr), n_test=len(te),
                                 pos_rate_train=float(yy[tr].mean()),
                                 pos_rate_test=float(yy[te].mean()),
                                 overlap=overlap))
                for mname, model in mdls.items():
                    pipe = Pipeline([("pre", make_preprocessor(XX)), ("clf", clone(model))])
                    pipe.fit(XX.iloc[tr], yy[tr])
                    rows.append(dict(dataset=name, protocol=proto, seed=seed, fold=fold,
                                     model=mname, overlap=overlap, n_test=len(te),
                                     **scores(pipe, XX.iloc[te], yy[te])))
    return rows, geom


def run_smote(name, df, seeds, folds=5):
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    ded = df.drop_duplicates().reset_index(drop=True)
    X, y = ded.drop(columns=["y"]), ded["y"].values
    n_real = len(X)
    rows = []
    for seed in seeds:
        mdls = get_models(seed)
        # (a) correct: resample inside the training folds, test on real rows
        for fold, (tr, te) in enumerate(StratifiedKFold(folds, shuffle=True,
                                                        random_state=seed).split(X, y)):
            for mname, model in mdls.items():
                pipe = ImbPipeline([("pre", make_preprocessor(X)),
                                    ("smote", SMOTE(random_state=seed)),
                                    ("clf", clone(model))])
                pipe.fit(X.iloc[tr], y[tr])
                rows.append(dict(dataset=name, setting="smote_in_fold", seed=seed,
                                 fold=fold, model=mname, n_test=len(te),
                                 **scores(pipe, X.iloc[te], y[te])))
        # resample the whole set once; real rows keep indices 0..n_real-1
        Z = make_preprocessor(X).fit_transform(X)
        Zr, yr = SMOTE(random_state=seed).fit_resample(Z, y)
        is_real = np.arange(len(Zr)) < n_real
        for fold, (tr, te) in enumerate(StratifiedKFold(folds, shuffle=True,
                                                        random_state=seed).split(Zr, yr)):
            te_real = te[is_real[te]]
            for mname, model in mdls.items():
                mdl = clone(model).fit(Zr[tr], yr[tr])
                # (b) what a paper following this procedure reports
                rows.append(dict(dataset=name, setting="smote_pre_resampled", seed=seed,
                                 fold=fold, model=mname, n_test=len(te),
                                 **scores(mdl, Zr[te], yr[te])))
                # (c) same fitted model, scored only on real test rows
                if len(np.unique(yr[te_real])) == 2:
                    rows.append(dict(dataset=name, setting="smote_pre_realtest", seed=seed,
                                     fold=fold, model=mname, n_test=len(te_real),
                                     **scores(mdl, Zr[te_real], yr[te_real])))
    return rows



# ------------------------------------------------------------------ R1.4 capacity ladder
# Same model families at several capacity settings. If capacity is the mechanism, inflation
# should rise with capacity WITHIN a family, not only across families.
def capacity_ladder(seed):
    L = {
        "LR_C0.01":  (LogisticRegression(C=0.01, max_iter=3000), 1),
        "LR_C1":     (LogisticRegression(C=1.0, max_iter=3000), 2),
        "LR_C100":   (LogisticRegression(C=100.0, max_iter=3000), 3),
        "kNN_k25":   (KNeighborsClassifier(n_neighbors=25), 1),
        "kNN_k5":    (KNeighborsClassifier(n_neighbors=5), 2),
        "kNN_k1":    (KNeighborsClassifier(n_neighbors=1), 3),
        "RF_d2":     (RandomForestClassifier(n_estimators=200, max_depth=2,
                                             random_state=seed, n_jobs=-1), 1),
        "RF_d6":     (RandomForestClassifier(n_estimators=200, max_depth=6,
                                             random_state=seed, n_jobs=-1), 2),
        "RF_dNone":  (RandomForestClassifier(n_estimators=200, max_depth=None,
                                             random_state=seed, n_jobs=-1), 3),
    }
    if HAS_XGB:
        L.update({
            "XGB_d2": (XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.1,
                                     subsample=0.9, eval_metric="logloss",
                                     random_state=seed, n_jobs=-1, verbosity=0), 1),
            "XGB_d4": (XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                     subsample=0.9, eval_metric="logloss",
                                     random_state=seed, n_jobs=-1, verbosity=0), 2),
            "XGB_d8": (XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                                     subsample=0.9, eval_metric="logloss",
                                     random_state=seed, n_jobs=-1, verbosity=0), 3),
        })
    return L


def run_capacity_ladder(name, df, seeds, folds=5):
    """P0 vs P1 only; the question is how inflation moves with capacity inside a family."""
    X = df.drop(columns=["y"]).reset_index(drop=True)
    y = df["y"].values
    keys = row_keys(X)
    groups = pd.factorize(keys)[0]
    rows = []
    for seed in seeds:
        L = capacity_ladder(seed)
        plans = {
            "P0_naive": StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y),
            "P1_group": StratifiedGroupKFold(folds, shuffle=True,
                                             random_state=seed).split(X, y, groups),
        }
        for proto, splits in plans.items():
            for fold, (tr, te) in enumerate(splits):
                overlap = float(np.isin(keys[te], keys[tr]).mean())
                for cfg, (model, rank) in L.items():
                    pipe = Pipeline([("pre", make_preprocessor(X)), ("clf", clone(model))])
                    pipe.fit(X.iloc[tr], y[tr])
                    rows.append(dict(dataset=name, protocol=proto, seed=seed, fold=fold,
                                     config=cfg, family=cfg.split("_")[0],
                                     capacity_rank=rank, overlap=overlap,
                                     **scores(pipe, X.iloc[te], y[te])))
    return rows


# ------------------------------------------------------------------ main
def main(datasets=DATASETS, n_seeds=20, folds=5, outdir="results_v2",
         smote_max_minority=0.40, ladder_seeds=10):
    os.makedirs(outdir, exist_ok=True)
    seeds = list(range(n_seeds))
    audits, cv, geom, sm, lad, log, prof, resid = [], [], [], [], [], {}, {}, {}
    t0 = time.time()
    for name, spec in datasets.items():
        try:
            df, dropped, info = prepare(spec)
        except Exception as e:
            log[name] = f"LOAD FAILED: {e}"; print(f"[{name}] load failed: {e}"); continue
        a = audit(df, "y", eps_list=(0.05, 0.1, 0.25, 0.5))
        a.update(dataset=name, dropped_id_columns=",".join(dropped),
                 positive_label=info["positive"], checksum=frame_checksum(df))
        a.update({f"resid_{k}": v for k, v in
                  residual_overlap_under_dedup(df).items()})
        audits.append(a)
        prof[name] = feature_space_profile(df.drop(columns=["y"]))
        resid[name] = residual_overlap_under_dedup(df)
        print(f"[{name}] n={a['n_rows']} dups={a['exact_dup_rows_feat_label']} "
              f"({a['exact_dup_pct']}%) residual_after_dedup={a['resid_residual_pct']}% "
              f"sha={a['checksum']}")
        r, g = run_protocols(name, df, seeds, folds)
        cv += r; geom += g
        lad += run_capacity_ladder(name, df, seeds[:ladder_seeds], folds)
        if min(df["y"].mean(), 1 - df["y"].mean()) < smote_max_minority:
            sm += run_smote(name, df, seeds, folds)
        log[name] = f"ok ({time.time() - t0:.0f}s)"
        print(f"[{name}] done, {time.time() - t0:.0f}s")

    pd.DataFrame(audits).to_csv(f"{outdir}/table1_audit_v2.csv", index=False)
    pd.DataFrame(cv).to_csv(f"{outdir}/cv_raw_v2.csv", index=False)
    pd.DataFrame(geom).to_csv(f"{outdir}/fold_geometry.csv", index=False)
    if sm:
        pd.DataFrame(sm).to_csv(f"{outdir}/smote_raw_v2.csv", index=False)
    pd.DataFrame(lad).to_csv(f"{outdir}/capacity_ladder.csv", index=False)
    json.dump(prof, open(f"{outdir}/feature_space_profile.json", "w"), indent=1)
    json.dump(resid, open(f"{outdir}/residual_overlap.json", "w"), indent=1)

    import sklearn, scipy, imblearn
    env = dict(python=platform.python_version(), numpy=np.__version__,
               pandas=pd.__version__, scikit_learn=sklearn.__version__,
               scipy=scipy.__version__, imbalanced_learn=imblearn.__version__,
               xgboost=(__import__("xgboost").__version__ if HAS_XGB else None),
               n_seeds=n_seeds, ladder_seeds=ladder_seeds, folds=folds,
               datasets_log=log,
               elapsed_seconds=round(time.time() - t0))
    json.dump(env, open(f"{outdir}/environment.json", "w"), indent=1)
    print(json.dumps(env, indent=1))
    return env
