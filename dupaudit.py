"""dupaudit: audit exact and near-duplicate records in tabular ML datasets.

Usage:
    python dupaudit.py data.csv --target TARGET [--positive LABEL] [--eps 0.1 0.25] [--out report.json]
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_LIKE = {"id", "patient_id", "patientid", "index", "unnamed: 0"}
MISSING_TOKENS = {"?", "nan", "none", "", "na", "n/a"}


def clean_frame(df, target):
    """Strip names/strings, drop ID-like columns, coerce numeric-looking text, bool->int."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    target = str(target).strip()
    dropped = [c for c in df.columns if c.lower() in ID_LIKE and c != target]
    df = df.drop(columns=dropped)
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
        elif pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            s = df[c].astype(str).str.strip()
            s = s.where(~s.str.lower().isin(MISSING_TOKENS), np.nan)
            num = pd.to_numeric(s, errors="coerce")
            if s.notna().sum() > 0 and num.notna().sum() >= 0.9 * s.notna().sum():
                df[c] = num
            else:
                df[c] = s
    return df, dropped


def binarize_target(y, positive=None):
    y = pd.Series(y).reset_index(drop=True)
    if pd.api.types.is_object_dtype(y) or pd.api.types.is_string_dtype(y):
        y = y.astype(object).where(y.isna(), y.astype(str).str.strip())
    vals = sorted(pd.unique(y.dropna()), key=str)
    if positive is not None:
        yb = (y == positive).astype(int)
        if yb.nunique() < 2:
            raise ValueError(f"positive={positive!r} not found in target values {vals[:10]}")
        return yb, {"positive": str(positive)}
    if len(vals) == 2:
        return (y == vals[1]).astype(int), {"positive": str(vals[1])}
    yn = pd.to_numeric(y, errors="coerce")
    if yn.notna().all():
        return (yn > 0).astype(int), {"positive": ">0"}
    raise ValueError(f"Cannot binarize target with values {vals[:10]}")


def make_preprocessor(X):
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    steps = []
    if num:
        steps.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                       ("sc", StandardScaler())]), num))
    if cat:
        steps.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                       ("oh", OneHotEncoder(handle_unknown="ignore",
                                                            sparse_output=False))]), cat))
    return ColumnTransformer(steps)


def row_keys(X):
    """Stable hash per feature vector (NaN-aware)."""
    return pd.util.hash_pandas_object(X.astype(str), index=False).values


def near_duplicate_rows(X, eps):
    """Rows absorbed into clusters whose members lie within `eps` (standardized Euclidean)."""
    Z = make_preprocessor(X).fit_transform(X)
    A = NearestNeighbors(radius=eps).fit(Z).radius_neighbors_graph(Z, mode="connectivity")
    n_clusters, _ = connected_components(A, directed=False)
    return int(len(X) - n_clusters)


def audit(df, target, eps_list=(0.1, 0.25)):
    """df must already be cleaned; target column must be binary 0/1."""
    X = df.drop(columns=[target])
    y = df[target].reset_index(drop=True)
    n = len(df)
    keys = row_keys(X)
    labels_per_vec = pd.Series(y.values).groupby(keys).nunique()
    out = {
        "n_rows": int(n),
        "n_features": int(X.shape[1]),
        "positive_rate": round(float(y.mean()), 4),
        "exact_dup_rows_feat_label": int(df.duplicated(keep="first").sum()),
        "exact_dup_rows_feat_only": int(X.duplicated(keep="first").sum()),
        "unique_feature_vectors": int(len(labels_per_vec)),
        "conflicting_label_groups": int((labels_per_vec > 1).sum()),
    }
    out["exact_dup_pct"] = round(100 * out["exact_dup_rows_feat_label"] / n, 2)
    ded = df.drop_duplicates()
    out["n_rows_after_dedup"] = int(len(ded))
    out["positive_rate_after_dedup"] = round(float(ded[target].mean()), 4)
    for eps in eps_list:
        out[f"near_dup_rows_eps{eps}"] = near_duplicate_rows(X, eps)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--target", required=True)
    ap.add_argument("--positive", default=None)
    ap.add_argument("--eps", type=float, nargs="*", default=[0.1, 0.25])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    raw = pd.read_csv(a.csv)
    df, dropped = clean_frame(raw, a.target)
    tgt = str(a.target).strip()
    df[tgt], _ = binarize_target(df[tgt], a.positive)
    df = df.dropna(subset=[tgt]).reset_index(drop=True)
    rep = audit(df, tgt, a.eps)
    rep["dropped_id_columns"] = dropped
    for k, v in rep.items():
        print(f"{k:32s} {v}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=2)


if __name__ == "__main__":
    main()
