"""Revision-stage audit additions (Reviewer 1 points 2 and 7)."""
import hashlib
import numpy as np
import pandas as pd
from dupaudit import make_preprocessor, row_keys


def frame_checksum(df):
    """Stable sha256 of the cleaned frame, so a reader can confirm the same file."""
    b = pd.util.hash_pandas_object(df.astype(str), index=False).values.tobytes()
    return hashlib.sha256(b).hexdigest()[:16]


def feature_space_profile(X):
    """R1.7: can identical feature vectors arise by coincidence rather than from one patient?

    Reports the cardinality of each feature and the size of the discrete grid they span.
    A grid far larger than n makes coincidental identity implausible; a grid near or below n
    makes it likely, which is the patient-level versus record-level distinction the reviewer
    asks for.
    """
    card = {c: int(X[c].nunique(dropna=False)) for c in X.columns}
    n_cont = sum(1 for c in X.columns
                 if pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 20)
    log10_grid = float(np.sum([np.log10(max(v, 1)) for v in card.values()]))
    return dict(cardinalities=card,
                min_cardinality=min(card.values()),
                median_cardinality=float(np.median(list(card.values()))),
                n_high_cardinality_features=n_cont,
                log10_grid_size=round(log10_grid, 2),
                log10_n=round(float(np.log10(len(X))), 2),
                coincidence_plausible=bool(log10_grid < np.log10(len(X)) + 2))


def residual_overlap_under_dedup(df, target="y"):
    """R1.2: deduplicating on (features, label) leaves conflicting-label records behind.

    Returns the number of rows that remain duplicated at the FEATURE level after an
    exact (features+label) deduplication, which is the residual leakage channel.
    """
    X = df.drop(columns=[target])
    ded = df.drop_duplicates()                      # features + label
    Xd = ded.drop(columns=[target])
    kd = row_keys(Xd)
    vc = pd.Series(kd).value_counts()
    repeated = vc[vc > 1]
    return dict(rows_after_dedup=int(len(ded)),
                feature_vectors_still_repeated=int(len(repeated)),
                rows_in_repeated_groups=int(repeated.sum()),
                residual_pct=round(100 * float(repeated.sum()) / max(len(ded), 1), 2))
