import tensorflow as tf
import joblib
import pandas as pd
import pickle
from pathlib import Path
import numpy as np

from .preprocessing import align_and_impute

from .mordred_engine import compute_mordred_from_smiles_list
from .smiles_utils import clean_smiles_list

BASE_DIR = Path(__file__).resolve().parent
PRED_THRESHOLD = 0.5
MODEL_DNN = "DNN"
MODEL_XGB = "ML"

class AGAPEPredictor:

    def __init__(self):

        model_dir = BASE_DIR / "saved_model"

        # -------- DNN --------
        self.dnn_folds = []

        with open(model_dir / MODEL_DNN / "featuresDNN.pkl", "rb") as f:
            self.dnn_features = pickle.load(f)

        dnn_dir = model_dir / MODEL_DNN

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
        if model_type.upper() == MODEL_DNN:

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
                model_type=MODEL_XGB
            )

            X_scaled = self.xgb_scaler.transform(X_clean.values)

            return X_scaled, imputation_percent, per_row_missing_str

    # ---------------------------------------------------
    # PREDICTION
    # ---------------------------------------------------

    def predict(self, descriptor_df: pd.DataFrame, model_type: str):

        # -------- DNN --------
        if model_type.upper() == MODEL_DNN:

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
            preds = (probs > PRED_THRESHOLD).astype(int)

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
            preds = (probs > PRED_THRESHOLD).astype(int)

            return (
                preds.flatten(),
                probs.flatten(),
                imputation_percent,
                per_row_missing_str
            )
        
    def predict_smiles(self, smiles_list, model_type: str = MODEL_XGB):
        """
        Internal G4PT AGAPE scoring function.
        Takes a list of SMILES and returns prediction results.
        """

        clean_smiles = clean_smiles_list(smiles_list)

        if len(clean_smiles) == 0:
            return pd.DataFrame(columns=[
                "canonical_smiles",
                "agape_prediction",
                "agape_score",
                "agape_label",
                "agape_model",
                "imputation_percent",
                "missing_features",
            ])

        descriptor_df = compute_mordred_from_smiles_list(clean_smiles)

        preds, probs, imputation_percent, per_row_missing = self.predict(
            descriptor_df,
            model_type=model_type,
        )

        return pd.DataFrame({
            "canonical_smiles": descriptor_df["SMILES"].tolist(),
            "agape_prediction": preds,
            "agape_score": probs,
            "agape_label": ["active" if p == 1 else "inactive" for p in preds],
            "agape_model": model_type,
            "imputation_percent": imputation_percent,
            "missing_features": per_row_missing,
        })
    

agape_predictor = AGAPEPredictor()