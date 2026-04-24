import tensorflow as tf
import joblib
import pandas as pd
import pickle
from pathlib import Path
import numpy as np

from .preprocessing import align_and_impute

BASE_DIR = Path(__file__).resolve().parent


class AGAPEPredictor:

    def __init__(self):

        model_dir = BASE_DIR / "saved_model"

        # -------- DNN --------
        self.dnn_folds = []

        with open(model_dir / "DNN" / "featuresDNN.pkl", "rb") as f:
            self.dnn_features = pickle.load(f)

        dnn_dir = model_dir / "DNN"

        for i in range(1, 11):
            fold_dir = dnn_dir / f"fold_{i}"

            fold_assets = {
                "model": tf.keras.models.load_model(fold_dir / "model.keras"),
                "imputer": joblib.load(fold_dir / "imputer.joblib"),
                "scaler": joblib.load(fold_dir / "scaler.joblib"),
            }

            self.dnn_folds.append(fold_assets)

        # -------- XGBoost --------
        with open(model_dir / "xgb_final_model.pkl", "rb") as f:
            self.xgb_model = pickle.load(f)

        with open(model_dir / "xgb_final_scaler.pkl", "rb") as f:
            self.xgb_scaler = pickle.load(f)

        with open(model_dir / "xgb_feature_list.pkl", "rb") as f:
            self.xgb_features = pickle.load(f)

    # ---------------------------------------------------
    # PREPROCESSING
    # ---------------------------------------------------

    def preprocess(self, descriptor_df: pd.DataFrame, model_type: str):
        """
        Align features, apply saved imputer,
        and scale according to the selected model.
        """

        descriptor_df = descriptor_df.drop(columns=["SMILES"], errors="ignore")

        # -------- DNN --------
        if model_type.upper() == "DNN":

            feature_list = list(self.dnn_features)

            df = descriptor_df.copy()
            df = df.reindex(columns=feature_list)
            df = df.apply(pd.to_numeric, errors="coerce")
            df = df.replace([np.inf, -np.inf], np.nan)

            total_values = df.shape[0] * df.shape[1]
            n_features = df.shape[1]

            per_row_missing = df.isna().sum(axis=1).astype(int).tolist()
            per_row_missing_str = [f"{m}/{n_features}" for m in per_row_missing]

            missing_before = int(df.isna().sum().sum())
            imputation_percent = (
                (missing_before / total_values) * 100
                if total_values > 0 else 0
            )

            return df, imputation_percent, per_row_missing_str

        # -------- XGBoost --------
        else:

            X_clean, imputation_percent, _, _, per_row_missing_str = align_and_impute(
                descriptor_df,
                model_type="ML"
            )

            X_scaled = self.xgb_scaler.transform(X_clean.values)

            return X_scaled, imputation_percent, per_row_missing_str

    # ---------------------------------------------------
    # PREDICTION
    # ---------------------------------------------------

    def predict(self, descriptor_df: pd.DataFrame, model_type: str):

        print("FIRST ROW DESCRIPTORS:")
        print(descriptor_df.iloc[0])

        print("MODEL TYPE:", model_type)

        # -------- DNN --------
        if model_type.upper() == "DNN":

            X_raw, imputation_percent, per_row_missing_str = self.preprocess(
                descriptor_df,
                model_type
            )

            fold_probs = []

            for fold_assets in self.dnn_folds:
                X_imputed = fold_assets["imputer"].transform(X_raw)
                X_scaled = fold_assets["scaler"].transform(X_imputed)
                probs = fold_assets["model"].predict(X_scaled, verbose=0).flatten()
                fold_probs.append(probs)

            probs = np.mean(np.vstack(fold_probs), axis=0)
            preds = (probs > 0.5).astype(int)

            return (
                preds.flatten(),
                probs.flatten(),
                imputation_percent,
                per_row_missing_str
            )

        # -------- XGBoost --------
        else:

            X_scaled, imputation_percent, per_row_missing_str = self.preprocess(
                descriptor_df,
                model_type
            )

            probs = self.xgb_model.predict_proba(X_scaled)[:, 1]
            preds = (probs > 0.5).astype(int)

            return (
                preds.flatten(),
                probs.flatten(),
                imputation_percent,
                per_row_missing_str
            )