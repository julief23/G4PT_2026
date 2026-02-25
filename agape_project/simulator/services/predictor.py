import tensorflow as tf
import joblib
import pandas as pd
import numpy as np
from pathlib import Path


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
        # Check missing features
        missing = set(self.feature_names) - set(descriptor_df.columns)
        if missing:
            raise ValueError(f"Missing descriptors: {missing}")

        # Keep correct columns
        X = descriptor_df[self.feature_names]

        # Enforce ordering
        X = X[self.feature_names]

        # Clean
        X = X.fillna(0)

        # Scale
        return self.scaler.transform(X)

    def predict(self, descriptor_df: pd.DataFrame):
        X_scaled = self.preprocess(descriptor_df)

        probs = self.model.predict(X_scaled)
        preds = (probs > 0.5).astype(int)

        return preds.flatten(), probs.flatten()