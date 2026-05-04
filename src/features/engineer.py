import pandas as pd

from src.config import BOOLEANS, CATEGORICAL, DUMMIES, NUMERICAL, TEST_SIZE


class FeatureEngineer:
    """Extracts the feature matrix and target from the cleaned DataFrame."""

    ALL_FEATURES = NUMERICAL + CATEGORICAL + BOOLEANS + DUMMIES

    def get_feature_matrix(self, df: pd.DataFrame):
        available = [c for c in self.ALL_FEATURES if c in df.columns]
        X = df[available].copy()
        y = df["condition"].map({"new": 0, "used": 1})
        return X, y

    def get_train_test_split(self, df: pd.DataFrame):
        X, y = self.get_feature_matrix(df)
        X_train = X.iloc[:-TEST_SIZE]
        X_test = X.iloc[-TEST_SIZE:]
        y_train = y.iloc[:-TEST_SIZE]
        y_test = y.iloc[-TEST_SIZE:]
        return X_train, X_test, y_train, y_test
