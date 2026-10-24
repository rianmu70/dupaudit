"""Regenerates every table, figure and statistic reported in the paper from the raw
outputs of experiments_v2.main(). Nothing in the manuscript is computed anywhere else.

Usage:  python analysis_v2.py results_v2 analysis_v2
"""
import json, os, sys, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": .7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.labelsize": 7.5, "legend.fontsize": 6.5})

MODELS = ["LR", "kNN", "SVM", "RF", "XGB"]
CAP = {"LR": 1, "kNN": 2, "SVM": 3, "RF": 4, "XGB": 5}
COL = dict(zip(MODELS, ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]))
SHORT = {"heart_kaggle_1025": "HD-1025", "maternal_risk_uci": "MHR",
         "early_diabetes_uci": "ESD", "lung_cancer_kaggle": "LC", "ilpd_uci": "ILPD",
         "ckd_uci": "CKD", "cleveland_uci": "CLEV", "heart_failure_918": "HF-918",
         "stroke_kaggle": "STR"}
METRICS = ["acc", "bacc", "f1", "auc"]
RNG = np.random.default_rng(0)


def boot_ci(v, n=10000):
    bs = np.array([RNG.choice(v, len(v), replace=True).mean() for _ in range(n)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main(indir="results_v2", outdir="analysis_v2"):
    os.makedirs(outdir, exist_ok=True)
    out = {}
    raw = pd.read_csv(f"{indir}/cv_raw_v2.csv")
    geom = pd.read_csv(f"{indir}/fold_geometry.csv")
    aud = pd.read_csv(f"{indir}/table1_audit_v2.csv")
    n_seeds = int(raw.seed.nunique())
    out["n_seeds"] = n_seeds
    out["n_fits"] = int(len(raw))

    # ---------------- seed-level frame -------------------------------------
    sd = raw.groupby(["dataset", "protocol", "model", "seed"], as_index=False)[
            METRICS + ["overlap"]].mean()
    sd.to_csv(f"{outdir}/seed_level_scores.csv", index=False)

    def pair(a, b):
        A = sd[sd.protocol == a].set_index(["dataset", "model", "seed"])
        B = sd[sd.protocol == b].set_index(["dataset", "model", "seed"])
        j = A[["acc", "overlap"]].join(B[["acc"]], rsuffix="_b").reset_index()
        j["infl"] = (j.acc - j.acc_b) * 100
        j["ov"] = j.overlap * 100
        j["cap"] = j.model.map(CAP)
        return j

    d = pair("P0_naive", "P1_group")          # primary
    d2 = pair("P0_naive", "P2_dedup")
    d2b = pair("P0_naive", "P2b_dedup_feat")
    d.to_csv(f"{outdir}/inflation_p0_p1.csv", index=False)

    # ---------------- R1.1 fold geometry + sensitivity ---------------------
    g = geom[["dataset", "protocol", "seed", "fold", "n_train", "n_test",
              "pos_rate_train", "pos_rate_test", "overlap"]]
    fs = g.groupby(["dataset", "protocol", "seed"]).agg(
            n_min=("n_test", "min"), n_max=("n_test", "max"), n_mean=("n_test", "mean"),
            pr_min=("pos_rate_test", "min"), pr_max=("pos_rate_test", "max")).reset_index()
    fs["size_range_pct"] = 100 * (fs.n_max - fs.n_min) / fs.n_mean
    fs["posrate_range_pp"] = 100 * (fs.pr_max - fs.pr_min)
    tbl_geom = fs.groupby(["dataset", "protocol"])[
        ["size_range_pct", "posrate_range_pp"]].mean().round(2).reset_index()
    tbl_geom.to_csv(f"{outdir}/tableA_fold_geometry.csv", index=False)

    # sensitivity 1: fold-size-weighted inflation (weight each fold by its test size)
    w = raw.merge(geom[["dataset", "protocol", "seed", "fold", "n_test"]],
                  on=["dataset", "protocol", "seed", "fold"], suffixes=("", "_g"))
    def wmean(x):
        return np.average(x["acc"], weights=x["n_test"])
    ws = (w.groupby(["dataset", "protocol", "model", "seed"])
            .apply(wmean).rename("acc_w").reset_index())
    A = ws[ws.protocol == "P0_naive"].set_index(["dataset", "model", "seed"])
    B = ws[ws.protocol == "P1_group"].set_index(["dataset", "model", "seed"])
    dw = A[["acc_w"]].join(B[["acc_w"]], rsuffix="_b").reset_index()
    dw["infl_w"] = (dw.acc_w - dw.acc_w_b) * 100

    # sensitivity 2: restrict to seeds whose P1 folds are within 10% size range
    ok = fs[(fs.protocol == "P1_group") & (fs.size_range_pct <= 10)][["dataset", "seed"]]
    d_bal = d.merge(ok, on=["dataset", "seed"])

    sens = []
    for ds in sorted(d.dataset.unique()):
        base = d[d.dataset == ds].groupby("model").infl.mean()
        wrow = dw[dw.dataset == ds].groupby("model").infl_w.mean()
        bal = d_bal[d_bal.dataset == ds].groupby("model").infl.mean() if len(
            d_bal[d_bal.dataset == ds]) else pd.Series(dtype=float)
        p2 = d2[d2.dataset == ds].groupby("model").infl.mean()
        p2b = d2b[d2b.dataset == ds].groupby("model").infl.mean()
        for m in MODELS:
            sens.append(dict(dataset=ds, model=m,
                             infl_p0_p1=round(float(base.get(m, np.nan)), 3),
                             infl_size_weighted=round(float(wrow.get(m, np.nan)), 3),
                             infl_balanced_seeds=(round(float(bal.get(m, np.nan)), 3)
                                                  if len(bal) else None),
                             n_balanced_seeds=int((ok.dataset == ds).sum()),
                             infl_p0_p2=round(float(p2.get(m, np.nan)), 3),
                             infl_p0_p2b=round(float(p2b.get(m, np.nan)), 3)))
    sens = pd.DataFrame(sens)
    sens.to_csv(f"{outdir}/tableB_protocol_sensitivity.csv", index=False)
    agree = sens.dropna(subset=["infl_p0_p1", "infl_p0_p2b"])
    out["sensitivity_max_abs_gap_p1_vs_p2b"] = round(
        float((agree.infl_p0_p1 - agree.infl_p0_p2b).abs().max()), 3)
    out["sensitivity_corr_p1_vs_p2b"] = round(
        float(np.corrcoef(agree.infl_p0_p1, agree.infl_p0_p2b)[0, 1]), 4)
    out["sensitivity_corr_p1_vs_weighted"] = round(float(np.corrcoef(
        sens.infl_p0_p1, sens.infl_size_weighted)[0, 1]), 4)

    # ---------------- R1.2 residual overlap --------------------------------
    ro = geom.groupby(["dataset", "protocol"]).overlap.mean().unstack().round(5)
    ro.to_csv(f"{outdir}/tableC_residual_overlap.csv")
    out["max_overlap_P1"] = float(ro["P1_group"].max())
    out["max_overlap_P2"] = float(ro["P2_dedup"].max())
    out["max_overlap_P2b"] = float(ro["P2b_dedup_feat"].max())

    # ---------------- R1.3 inference ---------------------------------------
    m = d.groupby(["dataset", "model"], as_index=False)[["infl", "ov"]].mean()
    r, p = pearsonr(m.ov, m.infl)
    dl = d.groupby("dataset", as_index=False)[["infl", "ov"]].mean()
    r2, p2v = pearsonr(dl.ov, dl.infl); rs2, ps2 = spearmanr(dl.ov, dl.infl)
    cr = smf.ols("infl ~ ov", data=d).fit(cov_type="cluster",
                                          cov_kwds={"groups": d.dataset})
    try:
        mm = smf.mixedlm("infl ~ ov", d, groups=d["dataset"]).fit(reml=True)
        out["mixedlm_slope"] = dict(beta=round(mm.params["ov"], 4),
            se=round(mm.bse["ov"], 4), p=float(mm.pvalues["ov"]))
    except Exception as e:
        out["mixedlm_slope"] = f"failed: {e}"
    out["pooled_pearson"] = dict(r=round(r, 3), p=float(p))
    out["dataset_level_pearson"] = dict(r=round(r2, 3), p=float(p2v), n=int(len(dl)))
    out["dataset_level_spearman"] = dict(rho=round(rs2, 3), p=float(ps2), n=int(len(dl)))
    out["cluster_robust_slope"] = dict(beta=round(cr.params["ov"], 4),
        se=round(cr.bse["ov"], 4), p=float(cr.pvalues["ov"]),
        ci=[round(x, 4) for x in cr.conf_int().loc["ov"]])

    rows = []
    for (ds, mdl), gg in d.groupby(["dataset", "model"]):
        v = gg.infl.values
        lo, hi = boot_ci(v)
        try:
            pw = float(wilcoxon(v).pvalue) if not np.allclose(v, 0) else 1.0
        except ValueError:
            pw = np.nan
        rows.append(dict(dataset=ds, model=mdl, mean_infl=round(v.mean(), 3),
                         ci_lo=round(lo, 3), ci_hi=round(hi, 3), n_seeds=len(v),
                         wilcoxon_p=pw))
    ci = pd.DataFrame(rows)
    rej, q, _, _ = multipletests(ci.wilcoxon_p.fillna(1).values, alpha=0.05, method="holm")
    ci["p_holm"] = np.round(q, 6); ci["sig_holm"] = rej
    ci.sort_values("mean_infl", ascending=False).to_csv(
        f"{outdir}/tableD_inflation_ci.csv", index=False)
    out["n_sig_holm"] = int(rej.sum()); out["n_tests"] = int(len(ci))
    out["min_attainable_wilcoxon_p"] = float(wilcoxon(np.arange(1, n_seeds + 1)).pvalue)
    out["holm_threshold"] = 0.05 / len(ci)
    out["n_ci_above_zero"] = int((ci.ci_lo > 0).sum())

    # ---------------- R1.4 capacity ----------------------------------------
    try:
        inter = smf.ols("infl ~ ov * cap", data=d).fit(cov_type="cluster",
                                                       cov_kwds={"groups": d.dataset})
        out["interaction_ov_x_cap"] = dict(beta=round(inter.params["ov:cap"], 4),
            se=round(inter.bse["ov:cap"], 4), p=float(inter.pvalues["ov:cap"]),
            ci=[round(x, 4) for x in inter.conf_int().loc["ov:cap"]])
    except Exception as e:
        out["interaction_ov_x_cap"] = f"failed: {e}"
    sl = []
    for mdl in MODELS:
        gg = d[d.model == mdl]
        try:
            f = smf.ols("infl ~ ov", data=gg).fit(cov_type="cluster",
                                                  cov_kwds={"groups": gg.dataset})
            lo, hi = f.conf_int().loc["ov"]
            sl.append(dict(model=mdl, slope=round(f.params["ov"], 4),
                           se=round(f.bse["ov"], 4), ci_lo=round(lo, 4),
                           ci_hi=round(hi, 4), p=float(f.pvalues["ov"])))
        except Exception:
            sl.append(dict(model=mdl, slope=None, se=None, ci_lo=None, ci_hi=None, p=None))
    pd.DataFrame(sl).to_csv(f"{outdir}/tableE_slopes.csv", index=False)
    out["per_model_slopes"] = sl

    lodo = []
    for ds in sorted(d.dataset.unique()):
        sub = d[d.dataset != ds]
        if sub.ov.nunique() < 3:
            continue
        try:
            s_ = {mdl: np.polyfit(sub[sub.model == mdl].ov, sub[sub.model == mdl].infl, 1)[0]
                  for mdl in ["LR", "XGB"]}
        except Exception:
            continue
        lodo.append(dict(excluded=ds, LR=round(s_["LR"], 4), XGB=round(s_["XGB"], 4),
                         ratio=round(s_["XGB"] / s_["LR"], 2) if s_["LR"] else None))
    if lodo:
        lodo = pd.DataFrame(lodo); lodo.to_csv(f"{outdir}/tableF_lodo.csv", index=False)
        rr = lodo.ratio.dropna()
        out["lodo_ratio_range"] = [float(rr.min()), float(rr.max())] if len(rr) else None

    # within-family capacity ladder
    lp = f"{indir}/capacity_ladder.csv"
    if os.path.exists(lp):
        lad = pd.read_csv(lp)
        ls = lad.groupby(["dataset", "protocol", "config", "family", "capacity_rank",
                          "seed"], as_index=False)[["acc"]].mean()
        LA = ls[ls.protocol == "P0_naive"].set_index(
            ["dataset", "config", "family", "capacity_rank", "seed"])
        LB = ls[ls.protocol == "P1_group"].set_index(
            ["dataset", "config", "family", "capacity_rank", "seed"])
        dl2 = LA[["acc"]].join(LB[["acc"]], rsuffix="_b").reset_index()
        dl2["infl"] = (dl2.acc - dl2.acc_b) * 100
        ov = d.groupby("dataset").ov.first()
        dl2["ov"] = dl2.dataset.map(ov)
        fam = []
        for f_, gg in dl2.groupby("family"):
            means = gg.groupby("capacity_rank").infl.mean().round(3).to_dict()
            try:
                ft = smf.ols("infl ~ ov * capacity_rank", data=gg).fit(
                    cov_type="cluster", cov_kwds={"groups": gg.dataset})
                fam.append(dict(family=f_,
                                interaction_beta=round(ft.params["ov:capacity_rank"], 4),
                                p=float(ft.pvalues["ov:capacity_rank"]),
                                mean_infl_by_rank=means))
            except Exception as e:
                fam.append(dict(family=f_, interaction_beta=None, p=None,
                                mean_infl_by_rank=means, note=str(e)[:80]))
        out["within_family_capacity"] = fam
        dl2.groupby(["family", "config", "capacity_rank"]).infl.mean().round(3)\
           .reset_index().to_csv(f"{outdir}/tableG_capacity_ladder.csv", index=False)

    # ---------------- R1.5 oversampling ------------------------------------
    sp = f"{indir}/smote_raw_v2.csv"
    if os.path.exists(sp):
        sm = pd.read_csv(sp)
        ss = sm.groupby(["dataset", "setting", "model", "seed"], as_index=False)[METRICS].mean()
        piv = ss.pivot_table(index=["dataset", "model", "seed"], columns="setting",
                             values=METRICS)
        res = []
        for met in METRICS:
            try:
                leak = (piv[(met, "smote_pre_realtest")] - piv[(met, "smote_in_fold")]) * 100
                comp = (piv[(met, "smote_pre_resampled")] - piv[(met, "smote_pre_realtest")]) * 100
                tot = (piv[(met, "smote_pre_resampled")] - piv[(met, "smote_in_fold")]) * 100
            except KeyError:
                continue
            t = pd.DataFrame(dict(leakage=leak, composition=comp, total=tot)).reset_index()
            t["metric"] = met
            res.append(t)
        if res:
            dec = pd.concat(res)
            dec.to_csv(f"{outdir}/smote_decomposition_raw.csv", index=False)
            summ = dec.groupby(["metric", "dataset", "model"])[
                ["leakage", "composition", "total"]].mean().round(3).reset_index()
            summ.to_csv(f"{outdir}/tableH_smote_decomposition.csv", index=False)
            out["smote_decomposition"] = {
                met: dict(leakage_mean=round(float(gg.leakage.mean()), 2),
                          leakage_max=round(float(gg.leakage.max()), 2),
                          composition_mean=round(float(gg.composition.mean()), 2),
                          composition_max=round(float(gg.composition.max()), 2))
                for met, gg in dec.groupby("metric")}

    # ---------------- figures ----------------------------------------------
    mm2 = d.groupby(["dataset", "model"], as_index=False)[["infl", "ov"]].mean()
    fig, ax = plt.subplots(figsize=(3.4, 2.35))
    for mdl in MODELS:
        gg = mm2[mm2.model == mdl]
        k, b = np.polyfit(gg.ov, gg.infl, 1); rr, _ = pearsonr(gg.ov, gg.infl)
        ax.plot([0, 100], [b, k * 100 + b], color=COL[mdl], lw=1.1, alpha=.85)
        ax.scatter(gg.ov, gg.infl, color=COL[mdl], s=16, zorder=3,
                   label=f"{mdl}: slope {k:.3f}, r={rr:.2f}")
    ax.axhline(0, color="grey", lw=.5); ax.set_xlim(-4, 104)
    ax.set_xlabel("Train–test overlap (% of test rows)")
    ax.set_ylabel("Accuracy inflation (pp)")
    ax.legend(frameon=False, loc="upper left", handletextpad=.4, borderpad=.1,
              labelspacing=.25)
    fig.tight_layout(pad=.3); fig.savefig(f"{outdir}/fig1.png", dpi=500); plt.close(fig)

    try:
        mm2["short"] = mm2.dataset.map(lambda x: SHORT.get(x, x))
        order = mm2.groupby("short").ov.first().sort_values(ascending=False).index
        pv = mm2.pivot(index="short", columns="model", values="infl").loc[order, MODELS]
        ovs = mm2.groupby("short").ov.first().loc[order]
        fig, ax = plt.subplots(figsize=(3.4, 2.2))
        x = np.arange(len(order))
        for i, mdl in enumerate(MODELS):
            ax.bar(x + (i - 2) * .165, pv[mdl], .165, color=COL[mdl], label=mdl, lw=0)
        ax.axhline(0, color="grey", lw=.5); ax.set_xticks(x)
        ax.set_xticklabels([f"{s}\n{o:.0f}%" for s, o in zip(order, ovs)], fontsize=6.2)
        ax.set_ylabel("Accuracy inflation (pp)")
        ax.legend(frameon=False, ncol=5, columnspacing=.8, handlelength=.9,
                  handletextpad=.3, loc="upper right")
        fig.tight_layout(pad=.3); fig.savefig(f"{outdir}/fig2.png", dpi=500)
        plt.close(fig)
    except Exception as e:
        out["fig2_error"] = str(e)[:120]

    if "smote_decomposition" in out:
      try:
          fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.0))
          for ax, met, lab in zip(axes, ["bacc", "f1"], ["Balanced acc.", "F1"]):
              gg = dec[dec.metric == met].groupby("dataset")[
                  ["leakage", "composition"]].mean()
              gg.index = [SHORT.get(i, i) for i in gg.index]
              gg = gg.loc[gg.sum(axis=1).sort_values().index]
              xx = np.arange(len(gg))
              ax.bar(xx, gg.leakage, .5, label="leakage", color="#C44E52", lw=0)
              ax.bar(xx, gg.composition, .5, bottom=gg.leakage, label="test composition",
                     color="#4C72B0", lw=0)
              ax.axhline(0, color="grey", lw=.5); ax.set_xticks(xx)
              ax.set_xticklabels(gg.index, rotation=55, ha="right", fontsize=6)
              ax.set_ylabel(f"{lab} inflation (pp)", fontsize=6.8)
              ax.tick_params(axis="y", labelsize=6.5)
          axes[0].legend(frameon=False, fontsize=5.8, loc="upper left")
          fig.tight_layout(pad=.3, w_pad=1.2)
          fig.savefig(f"{outdir}/fig3.png", dpi=500); plt.close(fig)
      except Exception as e:
          out["fig3_error"] = str(e)[:120]

    json.dump(out, open(f"{outdir}/headline_v2.json", "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_v2",
         sys.argv[2] if len(sys.argv) > 2 else "analysis_v2")
