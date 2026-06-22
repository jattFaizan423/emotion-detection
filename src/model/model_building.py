import os
import pickle
import numpy as np
import pandas as pd
import yaml
import logging
from typing import Tuple
from sklearn.ensemble import GradientBoostingClassifier


class ModelTrainer:

    def __init__(
        self,
        train_path: str,
        params_path: str = "params.yaml",
        model_dir: str = "data"
    ) -> None:

        self.train_path: str = train_path
        self.params_path: str = params_path
        self.model_dir: str = model_dir

        self.logger: logging.Logger = self._setup_logger()
        self.n_estimators: int | None = None
        self.model: GradientBoostingClassifier | None = None

    # ---------------- LOGGER ----------------
    def _setup_logger(self) -> logging.Logger:
        logger: logging.Logger = logging.getLogger("model_training")
        logger.setLevel(logging.INFO)

        ch: logging.Handler = logging.StreamHandler()
        formatter: logging.Formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)

        if not logger.handlers:
            logger.addHandler(ch)

        return logger

    # ---------------- LOAD PARAMS ----------------
    def load_params(self) -> None:
        try:
            self.logger.info(f"Loading params from {self.params_path}")

            with open(self.params_path, "r") as f:
                params: dict = yaml.safe_load(f)

            self.n_estimators: int = params['model_building']['n_estimators']

            self.logger.info(f"n_estimators: {self.n_estimators}")

        except Exception:
            self.logger.exception("Error loading params")
            raise

    # ---------------- LOAD DATA ----------------
    def load_data(self) -> pd.DataFrame:
        try:
            self.logger.info(f"Loading training data from {self.train_path}")

            df: pd.DataFrame = pd.read_csv(self.train_path)

            self.logger.info(f"Data shape: {df.shape}")
            return df

        except Exception:
            self.logger.exception("Error loading data")
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

    # ---------------- TRAIN ----------------
    def train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            self.logger.info("Training GradientBoostingClassifier")

            self.model = GradientBoostingClassifier(
                n_estimators=self.n_estimators
            )

            self.model.fit(X, y)

            self.logger.info("Model training completed")

        except Exception:
            self.logger.exception("Error during model training")
            raise

    # ---------------- SAVE ----------------
    def save_model(self) -> None:
        try:
            os.makedirs(self.model_dir, exist_ok=True)

            model_path: str = os.path.join(self.model_dir, "model.pkl")

            with open(model_path, "wb") as f:
                pickle.dump(self.model, f)

            self.logger.info(f"Model saved at {model_path}")

        except Exception:
            self.logger.exception("Error saving model")
            raise

    # ---------------- PIPELINE ----------------
    def run(self) -> None:
        try:
            self.logger.info("Model Training Pipeline Started")

            self.load_params()

            df: pd.DataFrame = self.load_data()

            X, y = self.split_features_labels(df)

            self.train_model(X, y)

            self.save_model()

            self.logger.info("Pipeline completed successfully ✅")

        except Exception:
            self.logger.exception("Pipeline failed ❌")
            raise


# ---------------- RUN ----------------
if __name__ == "__main__":
    trainer: ModelTrainer = ModelTrainer(
        train_path="./data/features_data/train_bow.csv"
    )
    trainer.run()