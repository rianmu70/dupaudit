"""ARS 2026 — duplicate-record leakage experiments.

Protocols (same 5-fold CV, repeated over seeds):
  P0_naive : StratifiedKFold on raw data (common practice in published papers)
  P1_group : StratifiedGroupKFold, identical feature vectors kept on one side (same rows, no leakage)
  P2_dedup : drop exact duplicates (features+label), then StratifiedKFold
Primary inflation = P0 - P1 (same rows, only the leakage differs).
SMOTE study (on deduplicated data): SMOTE inside training folds vs SMOTE before split.
"""
import glob
import json
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from dupaudit import audit, binarize_target, clean_frame, make_preprocessor, row_keys

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:  # local testing fallback only
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_XGB = False

DATASETS = {
    "heart_kaggle_1025": dict(src="kaggle", slug="johnsmith88/heart-disease-dataset", file="heart.csv", target="target"),
    "heart_failure_918": dict(src="kaggle", slug="fedesoriano/heart-failure-prediction", file="heart.csv", target="HeartDisease"),
    "stroke_kaggle": dict(src="kaggle", slug="fedesoriano/stroke-prediction-dataset", file="healthcare-dataset-stroke-data.csv", target="stroke"),
    "early_diabetes_uci": dict(src="uci", id=529, positive="Positive"),
    "maternal_risk_uci": dict(src="uci", id=863, positive="high risk"),
    "lung_cancer_kaggle": dict(src="kaggle", slug="mysarahmadbhat/lung-cancer",
                               file="survey lung cancer.csv", target="LUNG_CANCER", positive="YES"),
    "cleveland_uci": dict(src="uci", id=45),
    "ckd_uci": dict(src="uci", id=336, positive="ckd"),
    "ilpd_uci": dict(src="uci", id=225, positive=1),
}
METRICS = ["acc", "bacc", "f1", "auc"]


# ---------------------------------------------------------------- loading
def load_raw(spec):
    if spec["src"] == "frame":  # used for testing
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


# ---------------------------------------------------------------- models
def get_models(seed):
    m = {
        "LR": LogisticRegression(max_iter=3000),
        "RF": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel="rbf", gamma="scale", random_state=seed),
    }
    if HAS_XGB:
        m["XGB"] = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.9,
                                 eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0)
    else:
        m["XGB"] = HistGradientBoostingClassifier(random_state=seed)
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


# ---------------------------------------------------------------- CV
def run_protocols(name, df, seeds, folds=5):
    X = df.drop(columns=["y"]).reset_index(drop=True)
    y = df["y"].values
    keys = row_keys(X)
    groups = pd.factorize(keys)[0]
    ded = df.drop_duplicates().reset_index(drop=True)
    Xd, yd = ded.drop(columns=["y"]), ded["y"].values
    keys_d = row_keys(Xd)
    rows = []
    for seed in seeds:
        plans = {
            "P0_naive": (X, y, keys, StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y)),
            "P1_group": (X, y, keys, StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(X, y, groups)),
            "P2_dedup": (Xd, yd, keys_d, StratifiedKFold(folds, shuffle=True, random_state=seed).split(Xd, yd)),
        }
        for proto, (XX, yy, kk, splits) in plans.items():
            for fold, (tr, te) in enumerate(splits):
                overlap = float(np.isin(kk[te], kk[tr]).mean())
                for mname, model in get_models(seed).items():
                    pipe = Pipeline([("pre", make_preprocessor(XX)), ("clf", model)])
                    pipe.fit(XX.iloc[tr], yy[tr])
                    r = scores(pipe, XX.iloc[te], yy[te])
                    rows.append(dict(dataset=name, protocol=proto, seed=seed, fold=fold, model=mname,
                                     overlap=overlap, n_test=len(te), **r))
    return rows


def run_smote(name, df, seeds, folds=5):
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    ded = df.drop_duplicates().reset_index(drop=True)
    X, y = ded.drop(columns=["y"]), ded["y"].values
    rows = []
    for seed in seeds:
        # (a) SMOTE inside training folds (correct)
        for fold, (tr, te) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y)):
            for mname, model in get_models(seed).items():
                pipe = ImbPipeline([("pre", make_preprocessor(X)), ("smote", SMOTE(random_state=seed)), ("clf", model)])
                pipe.fit(X.iloc[tr], y[tr])
                rows.append(dict(dataset=name, setting="SMOTE_in_fold", seed=seed, fold=fold, model=mname,
                                 **scores(pipe, X.iloc[te], y[te])))
        # (b) SMOTE before split (common leaky practice)
        Z = make_preprocessor(X).fit_transform(X)
        Zr, yr = SMOTE(random_state=seed).fit_resample(Z, y)
        for fold, (tr, te) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=seed).split(Zr, yr)):
            for mname, model in get_models(seed).items():
                mdl = clone(model).fit(Zr[tr], yr[tr])
                rows.append(dict(dataset=name, setting="SMOTE_before_split", seed=seed, fold=fold, model=mname,
                                 **scores(mdl, Zr[te], yr[te])))
    return rows


# ---------------------------------------------------------------- stats & figures
def paired_stats(seed_df, a, b, key="protocol"):
    out = []
    for (ds, m), g in seed_df.groupby(["dataset", "model"]):
        pa = g[g[key] == a].set_index("seed")
        pb = g[g[key] == b].set_index("seed")
        common = pa.index.intersection(pb.index)
        if len(common) == 0:
            continue
        for met in METRICS:
            d = (pa.loc[common, met] - pb.loc[common, met]).values * 100
            if np.allclose(d, 0):
                p = 1.0
            else:
                try:
                    p = float(wilcoxon(d).pvalue)
                except ValueError:
                    p = np.nan
            out.append(dict(dataset=ds, model=m, metric=met, comparison=f"{a} - {b}",
                            mean_diff_pp=round(float(np.mean(d)), 3), std_diff_pp=round(float(np.std(d, ddof=1)) if len(d) > 1 else 0.0, 3),
                            n_seeds=len(d), wilcoxon_p=p))
    return pd.DataFrame(out)


def make_figures(raw, outdir):
    seed_df = raw.groupby(["dataset", "protocol", "model", "seed"], as_index=False)[METRICS + ["overlap"]].mean()
    m = seed_df.groupby(["dataset", "protocol", "model"], as_index=False)[METRICS + ["overlap"]].mean()
    p0 = m[m.protocol == "P0_naive"].set_index(["dataset", "model"])
    p1 = m[m.protocol == "P1_group"].set_index(["dataset", "model"])
    j = p0[["acc", "overlap"]].join(p1[["acc"]], rsuffix="_p1").reset_index()
    j["inflation_pp"] = (j["acc"] - j["acc_p1"]) * 100
    j["overlap_pct"] = j["overlap"] * 100

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for ds, g in j.groupby("dataset"):
        ax.scatter(g["overlap_pct"], g["inflation_pp"], label=ds, s=40, alpha=0.85)
    r = p = np.nan
    if j["overlap_pct"].nunique() > 1:
        r, p = pearsonr(j["overlap_pct"], j["inflation_pp"])
        k, b = np.polyfit(j["overlap_pct"], j["inflation_pp"], 1)
        xs = np.linspace(j["overlap_pct"].min(), j["overlap_pct"].max(), 50)
        ax.plot(xs, k * xs + b, "k--", lw=1)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("Test rows with an identical copy in training (%)")
    ax.set_ylabel("Accuracy inflation, naive − group split (pp)")
    ax.set_title(f"Pearson r = {r:.2f}" if not np.isnan(r) else "")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_overlap_vs_inflation.png"), dpi=300)
    plt.close(fig)

    piv = j.pivot(index="dataset", columns="model", values="inflation_pp")
    fig, ax = plt.subplots(figsize=(7, 4))
    piv.plot(kind="bar", ax=ax, width=0.8)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel("Accuracy inflation (pp)")
    ax.set_xlabel("")
    ax.legend(fontsize=7, frameon=False, ncol=5)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_inflation_by_dataset.png"), dpi=300)
    plt.close(fig)
    return j, (r, p)


# ---------------------------------------------------------------- main
def main(datasets=DATASETS, n_seeds=10, folds=5, outdir="results", smote_max_minority=0.40):
    os.makedirs(outdir, exist_ok=True)
    seeds = list(range(n_seeds))
    audits, cv_rows, sm_rows, log = [], [], [], {}
    t0 = time.time()
    for name, spec in datasets.items():
        try:
            df, dropped, info = prepare(spec)
        except Exception as e:  # keep going if one source fails
            log[name] = f"LOAD FAILED: {e}"
            print(f"[{name}] load failed: {e}")
            continue
        a = audit(df, "y")
        a.update(dataset=name, dropped_id_columns=",".join(dropped), positive_label=info["positive"])
        audits.append(a)
        print(f"[{name}] n={a['n_rows']} exact dups={a['exact_dup_rows_feat_label']} ({a['exact_dup_pct']}%)")
        cv_rows += run_protocols(name, df, seeds, folds)
        minority = min(df["y"].mean(), 1 - df["y"].mean())
        if minority < smote_max_minority:
            sm_rows += run_smote(name, df, seeds, folds)
        log[name] = f"ok ({time.time() - t0:.0f}s elapsed)"
        print(f"[{name}] done, {time.time() - t0:.0f}s")

    aud = pd.DataFrame(audits)
    aud = aud[["dataset"] + [c for c in aud.columns if c != "dataset"]]
    aud.to_csv(os.path.join(outdir, "table1_audit.csv"), index=False)
    raw = pd.DataFrame(cv_rows)
    raw.to_csv(os.path.join(outdir, "cv_raw.csv"), index=False)
    seed_df = raw.groupby(["dataset", "protocol", "model", "seed"], as_index=False)[METRICS + ["overlap"]].mean()
    summ = seed_df.groupby(["dataset", "protocol", "model"])[METRICS + ["overlap"]].agg(["mean", "std"]).round(4)
    summ.to_csv(os.path.join(outdir, "table2_cv_summary.csv"))
    st = pd.concat([paired_stats(seed_df, "P0_naive", "P1_group"), paired_stats(seed_df, "P0_naive", "P2_dedup")])
    st.to_csv(os.path.join(outdir, "table3_stats.csv"), index=False)
    j, (r, p) = make_figures(raw, outdir)
    j.to_csv(os.path.join(outdir, "fig1_data.csv"), index=False)

    head = {"pearson_r_overlap_vs_inflation": None if np.isnan(r) else round(float(r), 3),
            "pearson_p": None if np.isnan(p) else float(p),
            "max_acc_inflation_pp": round(float(j["inflation_pp"].max()), 2),
            "datasets_log": log, "n_seeds": n_seeds, "folds": folds, "xgboost_used": HAS_XGB}
    if sm_rows:
        sm = pd.DataFrame(sm_rows)
        sm.to_csv(os.path.join(outdir, "smote_raw.csv"), index=False)
        sm_seed = sm.groupby(["dataset", "setting", "model", "seed"], as_index=False)[METRICS].mean()
        sm_st = paired_stats(sm_seed, "SMOTE_before_split", "SMOTE_in_fold", key="setting")
        sm_st.to_csv(os.path.join(outdir, "table4_smote_stats.csv"), index=False)
        head["max_smote_acc_inflation_pp"] = round(float(sm_st[sm_st.metric == "acc"]["mean_diff_pp"].max()), 2)
    with open(os.path.join(outdir, "headline.json"), "w") as f:
        json.dump(head, f, indent=2)
    print(json.dumps(head, indent=2))
    return head
