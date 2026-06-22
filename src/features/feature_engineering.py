import os
import numpy as np
import pandas as pd
import yaml
import logging
from typing import Tuple
from scipy.sparse import csr_matrix
# CHANGE 1: CountVectorizer ki jagah TfidfVectorizer import kiya
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureEngineering:

    def __init__(
        self,
        train_path: str,
        test_path: str,
        params_path: str = "params.yaml",
        output_dir: str = "data/features_data"
    ) -> None:

        self.train_path: str = train_path
        self.test_path: str = test_path
        self.params_path: str = params_path
        self.output_dir: str = output_dir

        self.logger: logging.Logger = self._setup_logger()
        self.max_features: int | None = None
        # CHANGE 2: Type hint ko TfidfVectorizer kiya
        self.vectorizer: TfidfVectorizer | None = None

    # ---------------- LOGGER ----------------
    def _setup_logger(self) -> logging.Logger:
        logger: logging.Logger = logging.getLogger("feature_engineering")
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

            self.max_features: int = params['feature_engineering']['max_features']

            self.logger.info(f"Max features: {self.max_features}")

        except Exception:
            self.logger.exception("Error loading params")
            raise

    # ---------------- LOAD DATA ----------------
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            self.logger.info("Loading train and test data")

            train_df: pd.DataFrame = pd.read_csv(self.train_path)
            test_df: pd.DataFrame = pd.read_csv(self.test_path)

            self.logger.info(f"Train shape: {train_df.shape}")
            self.logger.info(f"Test shape: {test_df.shape}")

            return train_df, test_df

        except Exception:
            self.logger.exception("Error loading data")
            raise

    # ---------------- SPLIT X, y ----------------
    def split_features_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        try:
            X: np.ndarray = df['content'].values
            y: np.ndarray = df['sentiment'].values
            return X, y
        except KeyError:
            self.logger.error("Missing required columns")
            raise

    # ---------------- TF-IDF VECTORIZATION ----------------
    # CHANGE 3: Method ka naam aur vectorizer badal diya
    def apply_tfidf(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[csr_matrix, csr_matrix]:
        try:
            self.logger.info("Applying TfidfVectorizer")

            # CountVectorizer ki jagah TfidfVectorizer initialize kiya
            self.vectorizer = TfidfVectorizer(max_features=self.max_features)

            X_train_tfidf: csr_matrix = self.vectorizer.fit_transform(X_train)
            X_test_tfidf: csr_matrix = self.vectorizer.transform(X_test)

            return X_train_tfidf, X_test_tfidf

        except Exception:
            self.logger.exception("Error in TF-IDF vectorization")
            raise

    # ---------------- CONVERT TO DF ----------------
    def convert_to_dataframe(
        self,
        X: csr_matrix,
        y: np.ndarray
    ) -> pd.DataFrame:
        try:
            df: pd.DataFrame = pd.DataFrame(X.toarray())
            df['label'] = y
            return df

        except Exception:
            self.logger.exception("Error converting to dataframe")
            raise

    # ---------------- SAVE ----------------
    # CHANGE 4: Output files ke naam *_tfidf.csv kar diye version 2 ke liye
    def save_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        try:
            os.makedirs(self.output_dir, exist_ok=True)

            train_path: str = os.path.join(self.output_dir, "train_tfidf.csv")
            test_path: str = os.path.join(self.output_dir, "test_tfidf.csv")

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            self.logger.info(f"Saved files in {self.output_dir}")

        except Exception:
            self.logger.exception("Error saving data")
            raise

    # ---------------- PIPELINE ----------------
    def run(self) -> None:
        try:
            self.logger.info("Feature Engineering Pipeline (TF-IDF) Started")

            self.load_params()

            train_data, test_data = self.load_data()

            X_train, y_train = self.split_features_labels(train_data)
            X_test, y_test = self.split_features_labels(test_data)

            # CHANGE 5: Naye method ko call kiya
            X_train_tfidf, X_test_tfidf = self.apply_tfidf(X_train, X_test)

            train_df = self.convert_to_dataframe(X_train_tfidf, y_train)
            test_df = self.convert_to_dataframe(X_test_tfidf, y_test)

            self.save_data(train_df, test_df)

            self.logger.info("Pipeline completed successfully ✅")

        except Exception:
            self.logger.exception("Pipeline failed ❌")
            raise


# ---------------- RUN ----------------
if __name__ == "__main__":
    fe: FeatureEngineering = FeatureEngineering(
        train_path="./data/preprocessed_data/train_processed.csv",
        test_path="./data/preprocessed_data/test_processed.csv"
    )
    fe.run()