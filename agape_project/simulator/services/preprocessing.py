import pandas as pd
from pathlib import Path

# ----------------------------------------
# Load median values safely
# ----------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MEDIAN_PATH = BASE_DIR / "saved_model" / "feature_medians.csv"

median_df = pd.read_csv(MEDIAN_PATH)
FEATURE_MEDIANS = dict(zip(median_df["feature"], median_df["median"]))


# ----------------------------------------
# Alignment + Median Imputation
# ----------------------------------------

def align_and_impute(descriptor_df, feature_list, medians_dict=FEATURE_MEDIANS):
    """
    - Keep only model features
    - Add missing features if needed
    - Convert to numeric safely
    - Replace NaN using medians

    Returns:
        df_clean (DataFrame)
        replaced_percentage (float)
        per_row_missing (list[int])        # e.g. [2, 0, 5, ...]
        n_features (int)                   # e.g. 75
        per_row_missing_str (list[str])    # e.g. ["2/75", "0/75", "5/75", ...]
    """

    # Work on a copy
    df = descriptor_df.copy()

    # Keep only relevant features (ignore extra Mordred columns)
    df = df[[col for col in df.columns if col in feature_list]]

    # Add missing features with NaN
    for feature in feature_list:
        if feature not in df.columns:
            df[feature] = pd.NA

    # Enforce exact model order
    df = df[feature_list]

    # Convert everything to numeric safely (non-numeric -> NaN)
    df = df.apply(pd.to_numeric, errors="coerce")

    total_values = df.shape[0] * df.shape[1]
    n_features = df.shape[1]

    # per-molecule missing counts BEFORE imputation
    per_row_missing = df.isna().sum(axis=1).astype(int).tolist()
    per_row_missing_str = [f"{m}/{n_features}" for m in per_row_missing]

    # Count missing BEFORE imputation (global)
    missing_before = int(df.isna().sum().sum())

    # Median imputation
    for feature in feature_list:
        median_value = medians_dict.get(feature)
        if median_value is not None:
            df[feature] = df[feature].fillna(median_value)

    replaced_percentage = (
        (missing_before / total_values) * 100
        if total_values > 0 else 0
    )

    return df, replaced_percentage, per_row_missing, n_features, per_row_missing_str