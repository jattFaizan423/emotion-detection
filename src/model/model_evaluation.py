import os
import json
import pickle
import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


class ModelEvaluator:

    def __init__(
        self,
        model_path: str = "data/model.pkl",
        test_path: str = "./data/features_data/test_bow.csv",
        output_path: str = "metrics.json"
    ) -> None:

        self.model_path: str = model_path
        self.test_path: str = test_path
        self.output_path: str = output_path

        self.logger: logging.Logger = self._setup_logger()
        self.model = None

    # ---------------- LOGGER ----------------
    def _setup_logger(self) -> logging.Logger:
        logger: logging.Logger = logging.getLogger("model_evaluation")
        logger.setLevel(logging.INFO)

        ch: logging.Handler = logging.StreamHandler()
        formatter: logging.Formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)

        if not logger.handlers:
            logger.addHandler(ch)

        return logger

    # ---------------- LOAD MODEL ----------------
    def load_model(self) -> None:
        try:
            self.logger.info(f"Loading model from {self.model_path}")

            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)

            self.logger.info("Model loaded successfully")

        except Exception:
            self.logger.exception("Error loading model")
            raise

    # ---------------- LOAD DATA ----------------
    def load_data(self) -> pd.DataFrame:
        try:
            self.logger.info(f"Loading test data from {self.test_path}")

            df: pd.DataFrame = pd.read_csv(self.test_path)

            self.logger.info(f"Test data shape: {df.shape}")
            return df

        except Exception:
            self.logger.exception("Error loading test data")
            raise

    # ---------------- SPLIT ----------------
    def split_features_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        try:
            X: np.ndarray = df.iloc[:, 0:-1].values
            y: np.ndarray = df.iloc[:, -1].values
            return X, y

        except Exception:
            self.logger.exception("Error splitting features and labels")
            raise

    # ---------------- EVALUATE ----------------
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        try:
            self.logger.info("Running model evaluation")

            y_pred = self.model.predict(X)
            y_pred_proba = self.model.predict_proba(X)[:, 1]

            metrics: Dict[str, float] = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred),
                'recall': recall_score(y, y_pred),
                'auc': roc_auc_score(y, y_pred_proba)
            }

            self.logger.info(f"Evaluation metrics: {metrics}")
            return metrics

        except Exception:
            self.logger.exception("Error during evaluation")
            raise

    # ---------------- SAVE ----------------
    def save_metrics(self, metrics: Dict[str, float]) -> None:
        try:
            with open(self.output_path, "w") as f:
                json.dump(metrics, f, indent=4)

            self.logger.info(f"Metrics saved at {self.output_path}")

        except Exception:
            self.logger.exception("Error saving metrics")
            raise

    # ---------------- PIPELINE ----------------
    def run(self) -> None:
        try:
            self.logger.info("Model Evaluation Pipeline Started")

            self.load_model()

            df: pd.DataFrame = self.load_data()

            X, y = self.split_features_labels(df)

            metrics: Dict[str, float] = self.evaluate(X, y)

            self.save_metrics(metrics)

            self.logger.info("Pipeline completed successfully ✅")

        except Exception:
            self.logger.exception("Pipeline failed ❌")
            raise


# ---------------- RUN ----------------
if __name__ == "__main__":
    evaluator: ModelEvaluator = ModelEvaluator()
    evaluator.run()