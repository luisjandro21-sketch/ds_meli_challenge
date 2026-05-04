from src.data.cleaner import DataCleaner
from src.data.loader import DataLoader
from src.features.engineer import FeatureEngineer
from src.models.evaluator import ModelEvaluator
from src.models.trainer import ModelTrainer


class MLPipeline:
    """Orchestrates the full flow: load → clean → engineer → train → evaluate."""

    def __init__(self, model_params: dict | None = None, alert_threshold: float | None = None):
        self.loader = DataLoader()
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        self.trainer = ModelTrainer(params=model_params)
        kwargs = {}
        if alert_threshold is not None:
            kwargs["threshold"] = alert_threshold
        self.evaluator = ModelEvaluator(**kwargs)

    def run(self, data_path: str, save_model_path: str | None = None) -> dict:
        print("[ 1/5 ] Loading data...")
        raw = self.loader.load(data_path)
        print(f"        {len(raw):,} records loaded.")

        print("[ 2/5 ] Cleaning and extracting features...")
        cleaned = self.cleaner.fit_transform(raw)
        print(f"        {cleaned.shape[1]} columns available after cleaning.")

        print("[ 3/5 ] Splitting into train / test...")
        X_train, X_test, y_train, y_test = self.engineer.get_train_test_split(cleaned)
        print(f"        Train: {len(X_train):,}  |  Test: {len(X_test):,}")

        print("[ 4/5 ] Training tuned XGBoost model...")
        self.trainer.fit(X_train, y_train)
        print("        Training completed.")

        print("[ 5/5 ] Evaluating model...")
        y_pred = self.trainer.predict(X_test)
        metrics = self.evaluator.evaluate(y_test, y_pred)
        alerts = self.evaluator.check_alerts(metrics)
        self.evaluator.report(metrics, alerts)

        if save_model_path:
            self.trainer.save(save_model_path)

        return {"metrics": metrics, "alerts": alerts}
