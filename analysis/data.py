"""Data loading and target engineering."""

import numpy as np
import pandas as pd

from analysis.config import (
    COLUMN_RENAME,
    LIKERT_COLS,
    PROFILE_ORDER,
    RANDOM_STATE,
    SCORE_COLS,
    TIME_MAP,
    TOP_PLATFORMS,
)


def load_raw_data(path) -> pd.DataFrame:
    """Load and rename survey columns.

    Parameters
    ----------
    path : str or Path
        Path to smmh.csv.

    Returns
    -------
    pd.DataFrame
        Renamed dataframe.
    """
    df = pd.read_csv(path)
    df = df.rename(columns=COLUMN_RENAME)
    return df


def _to_numeric_likert(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Likert columns to numeric."""
    out = df.copy()
    for col in LIKERT_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    return out


def compute_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute four composite mental-health scores."""
    out = _to_numeric_likert(df)
    out["score_ADHD"] = out[["q9_no_purpose", "q10_distracted", "q12_easily_distracted"]].mean(axis=1)
    out["score_anxiety"] = out[["q11_restless", "q13_worries"]].mean(axis=1)
    out["score_self_esteem"] = out[["q15_compare", "q16_comparison_feeling", "q17_validation"]].mean(axis=1)
    out["score_depression"] = out[["q18_depressed", "q19_interest", "q20_sleep"]].mean(axis=1)
    return out


def assign_behavior_profile(df: pd.DataFrame, rng: np.random.Generator | None = None) -> tuple[pd.DataFrame, int]:
    """Assign dominant profile with random tie-breaking.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with composite scores.
    rng : np.random.Generator, optional
        RNG for reproducible tie breaks.

    Returns
    -------
    tuple[pd.DataFrame, int]
        Updated dataframe and number of tie-break rows.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)

    out = df.copy()
    score_matrix = out[SCORE_COLS].to_numpy()
    max_vals = score_matrix.max(axis=1)
    is_tie = (score_matrix == max_vals[:, None]).sum(axis=1) > 1

    profiles = []
    for i, row_scores in enumerate(score_matrix):
        max_val = row_scores.max()
        tied_idx = np.where(row_scores == max_val)[0]
        if len(tied_idx) == 1:
            profiles.append(PROFILE_ORDER[tied_idx[0]])
        else:
            choice = int(rng.choice(tied_idx))
            profiles.append(PROFILE_ORDER[choice])

    out["behavior_profile"] = profiles
    return out, int(is_tie.sum())


def add_predictive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer exogenous features for modeling."""
    out = df.copy()
    out["time_social_media_ord"] = out["time_on_social_media"].map(TIME_MAP)
    out["n_platforms"] = out["platforms"].fillna("").apply(
        lambda x: len([p.strip() for p in str(x).split(",") if p.strip()])
    )
    for platform in TOP_PLATFORMS:
        col = f"uses_{platform.lower()}"
        out[col] = out["platforms"].fillna("").str.contains(platform, case=False, regex=False).astype(int)
    return out


def prepare_dataset(path) -> tuple[pd.DataFrame, int]:
    """Full data preparation pipeline."""
    df = load_raw_data(path)
    df = df[df["uses_social_media"].str.strip().str.lower() == "yes"].copy()
    df = compute_composite_scores(df)
    df, n_ties = assign_behavior_profile(df)
    df = add_predictive_features(df)
    return df, n_ties


def get_feature_columns() -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature column names."""
    numeric = ["age", "time_social_media_ord", "n_platforms"] + [
        f"uses_{p.lower()}" for p in TOP_PLATFORMS
    ]
    categorical = ["gender", "relationship", "occupation", "organization"]
    return numeric, categorical
