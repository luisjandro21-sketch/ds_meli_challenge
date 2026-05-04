import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import BEST_PARAMS, BOOLEANS, CATEGORICAL, DUMMIES, NUMERICAL


class ModelTrainer:
    """Builds, trains, and serializes the tuned XGBoost pipeline."""

    def __init__(self, params: dict | None = None):
        self._params = params or BEST_PARAMS
        self.pipeline: Pipeline | None = None

    def _build_pipeline(self) -> Pipeline:
        bool_transformer = FunctionTransformer(lambda x: x.astype(int))

        preprocessor = ColumnTransformer(
            transformers=[
                ("numerical", StandardScaler(), NUMERICAL),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                ("booleans", bool_transformer, BOOLEANS),
                ("dummies", "passthrough", DUMMIES),
            ]
        )

        classifier = xgb.XGBClassifier(**self._params)

        return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "ModelTrainer":
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Call fit() before predict().")
        return self.pipeline.predict(X)

    def save(self, path: str) -> None:
        if self.pipeline is None:
            raise RuntimeError("No trained model to save.")
        joblib.dump(self.pipeline, path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> "ModelTrainer":
        self.pipeline = joblib.load(path)
        return self
