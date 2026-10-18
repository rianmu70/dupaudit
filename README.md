# Duplicate-record leakage audit of public medical tabular datasets

Code and results for the ARS 2026 paper *How Much of the Reported Accuracy Is Real?
A Duplicate-Record Leakage Audit of Nine Public Medical Tabular Datasets*.

## Contents

| File | Purpose |
|---|---|
| `dupaudit.py` | Standalone auditing tool (CLI). Reports exact and near-duplicate rates for any CSV. |
| `experiments.py` | Full experiment: duplicate audit, three partitioning protocols, oversampling study, figures. |
| `results/` | All tables, figures and fold-level scores from the reported run. |

## Auditing your own dataset

```
python dupaudit.py data.csv --target OUTCOME --out report.json
```

Optional: `--positive LABEL` names the positive class, `--eps 0.1 0.25` sets the
near-duplicate thresholds. Record-identifier columns are dropped automatically,
because an identifier is unique by construction and hides every duplicate.

## Reproducing the experiment

```
pip install ucimlrepo kagglehub xgboost imbalanced-learn
python -c "import experiments as E; E.main(n_seeds=10, outdir='results')"
```

All nine datasets are downloaded programmatically at run time, so no manual data
preparation is required. The run takes about 33 minutes on two CPU cores; no GPU
is used. Datasets that fail to download are logged and skipped rather than
aborting the run.

## Protocols

- `P0_naive` — stratified 5-fold CV on the raw file. Reproduces common practice.
- `P1_group` — stratified grouped 5-fold CV; every copy of a record stays on one
  side of the split. Same rows as P0, no overlap. Primary comparator.
- `P2_dedup` — exact duplicates removed, then stratified 5-fold CV.

Inflation is reported as P0 − P1, which holds sample size constant and so
isolates leakage from any change in the amount of training data.

## Reporting checklist

1. Report row counts and class prevalence before and after deduplication.
2. Drop record-identifier columns before auditing.
3. Split by record, not by row — or deduplicate before splitting.
4. Resample inside training folds only; report metrics on a test partition that
   keeps the original class prevalence.

## Result files

| File | Contents |
|---|---|
| `table1_audit.csv` | Duplicate audit per dataset |
| `table2_cv_summary.csv` | Mean and SD per dataset, protocol and model |
| `table3_stats.csv` | Paired differences with Wilcoxon p-values |
| `table4_smote_stats.csv` | Oversampling-before-split differences |
| `cv_raw.csv`, `smote_raw.csv` | Fold-level scores (6,750 and 3,000 model fits) |
| `fig1_data.csv` | Overlap and inflation per dataset and model |
| `headline.json` | Headline statistics for the reported run |
