import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf


class DNNModel:
    """
    Minimal wrapper around a trained DNN model.
    - Loads model + feature list
    - Selects/reorders columns
    - Predicts
    """

    def __init__(self, model_path: str, features_path: str):
        self.model_path = model_path
        self.features_path = features_path

        # Load once
        self.model = tf.keras.models.load_model(self.model_path)

        with open(self.features_path, "rb") as f:
            self.features = pickle.load(f)  # must be a LIST in the right order

    def prepare_X(self, df_desc: pd.DataFrame) -> np.ndarray:
        """
        Ensure the input has the exact columns the model was trained on.
        """
        missing = [c for c in self.features if c not in df_desc.columns]
        if missing:
            raise ValueError(f"Missing {len(missing)} features (e.g. {missing[:5]}).")

        X = df_desc[self.features].copy()
        return X.values

    def predict(self, df_desc: pd.DataFrame):
        """
        Returns:
        - proba: probability (DNN output)
        - pred: class (0/1) using threshold 0.5
        """
        X = self.prepare_X(df_desc)

        # Keras outputs shape (n, 1) usually
        proba = self.model.predict(X, verbose=0).reshape(-1)
        pred = (proba >= 0.5).astype(int)

        return pred, proba