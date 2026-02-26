import tensorflow as tf
import joblib
import pandas as pd
from pathlib import Path

from .preprocessing import align_and_impute, FEATURE_MEDIANS


BASE_DIR = Path(__file__).resolve().parent


class AGAPEPredictor:

    def __init__(self):
        model_path = BASE_DIR / "saved_model" / "agape_model.keras"
        scaler_path = BASE_DIR / "saved_model" / "scaler.joblib"
        feature_path = BASE_DIR / "saved_model" / "feature_list.pkl"

        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(feature_path)

    def preprocess(self, descriptor_df: pd.DataFrame):
        """
        Align features, apply median imputation,
        scale and return imputation percentage.
        """

        # Use align + impute 
        X_clean, imputation_percent, per_row_missing, n_features, per_row_missing_str = align_and_impute(
            descriptor_df,
            self.feature_names,
            FEATURE_MEDIANS
        )

        X_scaled = self.scaler.transform(X_clean)

        return X_scaled, imputation_percent, per_row_missing_str

    def predict(self, descriptor_df: pd.DataFrame):

        X_scaled, imputation_percent, per_row_missing_str = self.preprocess(descriptor_df)

        probs = self.model.predict(X_scaled)
        preds = (probs > 0.5).astype(int)

        return (
            preds.flatten(),
            probs.flatten(),
            imputation_percent,
            per_row_missing_str
        )