import os
import numpy as np
import pandas as pd
import yaml
import logging
from typing import Tuple
from sklearn.model_selection import train_test_split


class DataIngestion:

    def __init__(self, params_path: str) -> None:
        self.params_path: str = params_path
        self.logger: logging.Logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger: logging.Logger = logging.getLogger("data_ingestion")
        logger.setLevel(logging.DEBUG)

        ch: logging.Handler = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        formatter: logging.Formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        ch.setFormatter(formatter)

        if not logger.handlers:
            logger.addHandler(ch)

        return logger

    def load_params(self) -> float:
        try:
            self.logger.info(f"Loading parameters from {self.params_path}")

            with open(self.params_path, 'r') as f:
                params: dict = yaml.safe_load(f)

            test_size: float = params['data_ingestion']['test_size']

            self.logger.info(f"Test size loaded: {test_size}")
            return test_size

        except FileNotFoundError:
            self.logger.error(f"Params file not found: {self.params_path}")
            raise
        except KeyError:
            self.logger.error("Missing key: data_ingestion -> test_size")
            raise
        except Exception:
            self.logger.exception("Unexpected error while loading params")
            raise

    def read_data(self, url: str) -> pd.DataFrame:
        try:
            self.logger.info(f"Reading data from {url}")

            df: pd.DataFrame = pd.read_csv(url)

            self.logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df

        except Exception:
            self.logger.exception("Error reading data")
            raise

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            self.logger.info("Starting data processing")

            if 'tweet_id' not in df.columns or 'sentiment' not in df.columns:
                self.logger.error("Required columns missing")
                raise ValueError("Required columns missing in dataset")

            df = df.drop(columns=['tweet_id'])

            final_df: pd.DataFrame = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()

            final_df['sentiment'] = final_df['sentiment'].map({
                'happiness': 1,
                'sadness': 0
            }).astype(int)

            self.logger.info(f"Data processing completed. Final shape: {final_df.shape}")
            return final_df

        except Exception:
            self.logger.exception("Error in data processing")
            raise

    def save_data(
        self,
        data_path: str,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> None:
        try:
            self.logger.info(f"Saving data to {data_path}")

            os.makedirs(data_path, exist_ok=True)

            train_path: str = os.path.join(data_path, "train.csv")
            test_path: str = os.path.join(data_path, "test.csv")

            train_data.to_csv(train_path, index=False)
            test_data.to_csv(test_path, index=False)

            self.logger.info("Data saved successfully")

        except Exception:
            self.logger.exception("Error saving data")
            raise

    def run_pipeline(self) -> None:
        try:
            self.logger.info("Starting data ingestion pipeline")

            test_size: float = self.load_params()

            url: str = (
                'https://raw.githubusercontent.com/campusx-official/'
                'jupyter-masterclass/main/tweet_emotions.csv'
            )

            df: pd.DataFrame = self.read_data(url)

            final_df: pd.DataFrame = self.process_data(df)

            self.logger.info("Splitting data into train and test")

            train_data, test_data = train_test_split(
                final_df,
                test_size=test_size,
                random_state=42
            )

            data_path: str = os.path.join("data", "raw_data")

            self.save_data(data_path, train_data, test_data)

            self.logger.info("Pipeline completed successfully")

        except Exception:
            self.logger.exception("Pipeline failed")
            raise


if __name__ == "__main__":
    ingestion: DataIngestion = DataIngestion(params_path='params.yaml')
    ingestion.run_pipeline()