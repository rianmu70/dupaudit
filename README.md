# Duplicate-record leakage audit of public medical tabular datasets

Code and results for *How Much of the Reported Accuracy Is Real? A Duplicate-Record
Leakage Audit of Nine Public Medical Tabular Datasets* (ARS 2026).

Every table, figure and statistic in the paper is produced by `analysis_v2.py` from the
fold-level scores in `results_v2/`. Nothing reported in the paper is computed elsewhere.

## Reproducing the run

```
pip install -r requirements.txt
python -c "import experiments_v2 as E; E.main(n_seeds=20, ladder_seeds=10, outdir='results_v2')"
python analysis_v2.py results_v2 analysis_v2
```

All nine datasets are downloaded programmatically at run time, so no manual data
preparation is required. The run takes about 1 hour 33 minutes on two CPU cores; no GPU
is used. Library versions and the elapsed time of the reported run are recorded in
`results_v2/environment.json`, and a SHA-256 digest of each cleaned frame is in
`results_v2/table1_audit_v2.csv` so you can confirm you obtained the same file.

`run_revision_on_kaggle.ipynb` performs both steps in a Kaggle notebook.

## Contents

| File | Purpose |
|---|---|
| `dupaudit.py` | Standalone auditing tool (CLI). Exact and near-duplicate rates for any CSV. |
| `audit_v2.py` | Revision-stage audit helpers: dataset digests, residual overlap, feature-space profile. |
| `experiments_v2.py` | The experiment: duplicate audit, four protocols, capacity ladder, oversampling study. |
| `analysis_v2.py` | Regenerates every table, figure and statistic in the paper. |
| `results_v2/` | Fold-level scores and audit output from the reported run. |
| `analysis_v2/` | The tables and figures as they appear in the paper. |
| `experiments.py`, `results/`, `run_on_kaggle.ipynb` | The first version, kept for comparison. |

## Auditing your own dataset

```
python dupaudit.py data.csv --target OUTCOME --out report.json
```

`--positive LABEL` names the positive class and `--eps 0.05 0.1 0.25 0.5` sets the
near-duplicate thresholds. Record-identifier columns are dropped automatically, because
an identifier is unique by construction and hides every duplicate.

## Protocols

| Protocol | Rule | Train-test overlap it leaves |
|---|---|---|
| `P0_naive` | Stratified 5-fold CV on the raw file. Reproduces common practice. | up to 97.6% |
| `P1_group` | Stratified grouped 5-fold CV, grouped on the feature vector. Same rows as P0. Primary comparator. | 0% |
| `P2_dedup` | Drop duplicates on (features, label), then stratified 5-fold CV. | up to 3.8% |
| `P2b_dedup_feat` | Drop duplicates on features alone, then stratified 5-fold CV. Even folds, zero overlap. | 0% |

Inflation is reported as P0 − P1, which holds the sample size constant. P2b is the
geometry-controlled check: grouped assignment cannot hold fold size and class balance
fixed while eliminating overlap, so the two comparators hold different nuisances fixed
and their agreement is evidence that neither nuisance drives the result.

## Reporting checklist

1. Report row counts and class prevalence before and after deduplication.
2. Drop record-identifier columns before auditing.
3. Split by record, not by row — or deduplicate before splitting.
4. Deduplicate or group on the feature columns alone, not on features together with the
   label: deduplicating on the pair leaves conflicting-label records split across the
   partition.
5. Resample inside training folds only, and report metrics on a test partition that keeps
   the original class prevalence.

## Result files

| File | Contents |
|---|---|
| `results_v2/table1_audit_v2.csv` | Duplicate audit, four near-duplicate thresholds, digests |
| `results_v2/cv_raw_v2.csv` | Fold-level scores, 18,000 fits |
| `results_v2/fold_geometry.csv` | Fold sizes and class balance for every fold |
| `results_v2/capacity_ladder.csv` | Within-family capacity experiment, 10,800 fits |
| `results_v2/smote_raw_v2.csv` | Oversampling study, three settings, 9,000 fits |
| `results_v2/environment.json` | Library versions, seeds, elapsed time |
| `analysis_v2/tableA–tableH*.csv` | The paper's tables |
| `analysis_v2/fig1–fig3.png` | The paper's figures |
| `analysis_v2/headline_v2.json` | Every statistic quoted in the text |
