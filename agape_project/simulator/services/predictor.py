import tensorflow as tf
import joblib
import pandas as pd
import pickle
from pathlib import Path
import numpy as np

from .preprocessing import align_and_impute, FEATURE_MEDIANS


BASE_DIR = Path(__file__).resolve().parent


class AGAPEPredictor:

    def __init__(self):

        model_dir = BASE_DIR / "saved_model"

        # -------- DNN --------
        self.dnn_model = tf.keras.models.load_model(model_dir / "modelDNN.keras")
        self.dnn_scaler = joblib.load(model_dir / "scalerDNN.joblib")

        with open(model_dir / "featuresDNN.pkl", "rb") as f:
            self.dnn_features = pickle.load(f)

        # -------- XGBoost --------
        with open(model_dir / "xgb_final_model.pkl", "rb") as f:
            self.xgb_model = pickle.load(f)

        with open(model_dir / "xgb_final_scaler.pkl", "rb") as f:
            self.xgb_scaler = pickle.load(f)

        with open(model_dir / "xgb_feature_list.pkl", "rb") as f:
            self.xgb_features = pickle.load(f)

    def preprocess(self, descriptor_df: pd.DataFrame, model_type: str):
        """
        Align features, apply median imputation,
        scale according to the selected model.
        """
        descriptor_df = descriptor_df.drop(columns=["SMILES"], errors="ignore")

        if model_type.upper() == "DNN":
            feature_list = self.dnn_features
            scaler = self.dnn_scaler
        else:
            feature_list = self.xgb_features
            scaler = self.xgb_scaler

        X_clean, imputation_percent, per_row_missing, n_features, per_row_missing_str = align_and_impute(
            descriptor_df,
            feature_list,
            FEATURE_MEDIANS
        )

        X_scaled = scaler.transform(X_clean.values)

        return X_scaled, imputation_percent, per_row_missing_str



    def predict(self, descriptor_df: pd.DataFrame, model_type: str):

        X_scaled, imputation_percent, per_row_missing_str = self.preprocess(
            descriptor_df,
            model_type
        )

        print("FIRST ROW DESCRIPTORS:")
        print(descriptor_df.iloc[0])

        print("MODEL TYPE:", model_type)
        print("SCALER USED:", "DNN" if model_type.upper()=="DNN" else "XGB")
        if model_type.upper() == "DNN":

            probs = self.dnn_model.predict(X_scaled, verbose=0)
            preds = (probs > 0.5).astype(int)

            return (
                preds.flatten(),
                probs.flatten(),
                imputation_percent,
                per_row_missing_str
            )

        else:  # XGBoost

            probs = self.xgb_model.predict_proba(X_scaled)[:, 1]
            preds = (probs > 0.5).astype(int)


            return (
                preds.flatten(),
                probs.flatten(),
                imputation_percent,
                per_row_missing_str
            )
