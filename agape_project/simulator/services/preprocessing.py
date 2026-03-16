import pandas as pd
from pathlib import Path
import numpy as np
import joblib
import pickle

# ----------------------------------------
# Paths
# ----------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_model"

# ----------------------------------------
# Load helpers
# ----------------------------------------

def _load_pickle_or_joblib(path):
    path = Path(path)

    if path.suffix == ".joblib":
        return joblib.load(path)

    with open(path, "rb") as f:
        return pickle.load(f)

# ----------------------------------------
# Model-specific artifacts
# ----------------------------------------

MODEL_CONFIG = {
    "DNN": {
        "feature_path": MODEL_DIR / "featuresDNN.pkl",
        "imputer_path": MODEL_DIR / "imputerDNN.joblib",
    },
    "ML": {
        "feature_path": MODEL_DIR / "xgb_feature_list.pkl",
        "imputer_path": MODEL_DIR / "xgb_final_imputer.pkl",
    },
}
# Preload artifacts once
LOADED_ARTIFACTS = {}

for model_name, cfg in MODEL_CONFIG.items():
    LOADED_ARTIFACTS[model_name] = {
        "feature_list": _load_pickle_or_joblib(cfg["feature_path"]),
        "imputer": _load_pickle_or_joblib(cfg["imputer_path"]),
    }

# ----------------------------------------
# Alignment + Imputation
# ----------------------------------------

def align_and_impute(descriptor_df, model_type="DNN"):
    """
    Align descriptors to the selected model feature space and apply the
    corresponding saved imputer.

    Parameters
    ----------
    descriptor_df : pd.DataFrame
        Raw descriptor dataframe.
    model_type : str
        "dnn" or "xgb"

    Returns
    -------
    df_clean : pd.DataFrame
        Imputed dataframe aligned to model feature order
    replaced_percentage : float
        Percentage of values that were missing before imputation
    per_row_missing : list[int]
        Number of missing values per row before imputation
    n_features : int
        Number of model features
    per_row_missing_str : list[str]
        Same info as strings like "2/97"
    """

    model_type = model_type.upper()

    if model_type not in LOADED_ARTIFACTS:
        raise ValueError(f"Unknown model_type '{model_type}'. Expected one of: {list(LOADED_ARTIFACTS.keys())}")

    feature_list = list(LOADED_ARTIFACTS[model_type]["feature_list"])
    imputer = LOADED_ARTIFACTS[model_type]["imputer"]

    # ----------------------------------------
    # Work on a copy
    # ----------------------------------------

    df = descriptor_df.copy()

    # ----------------------------------------
    # Keep only model features + enforce order
    # Missing columns are added as NaN
    # ----------------------------------------

    df = df.reindex(columns=feature_list)

    # ----------------------------------------
    # Convert everything to numeric safely
    # ----------------------------------------

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    total_values = df.shape[0] * df.shape[1]
    n_features = df.shape[1]

    # ----------------------------------------
    # Per-row missing counts BEFORE imputation
    # ----------------------------------------

    per_row_missing = df.isna().sum(axis=1).astype(int).tolist()
    per_row_missing_str = [f"{m}/{n_features}" for m in per_row_missing]

    # ----------------------------------------
    # Count missing BEFORE imputation
    # ----------------------------------------

    missing_before = int(df.isna().sum().sum())

    # ----------------------------------------
    # Apply saved imputer
    # ----------------------------------------

    df_imputed = pd.DataFrame(
        imputer.transform(df),
        columns=feature_list,
        index=df.index
    )

    replaced_percentage = (
        (missing_before / total_values) * 100
        if total_values > 0 else 0
    )

    return df_imputed, replaced_percentage, per_row_missing, n_features, per_row_missing_str