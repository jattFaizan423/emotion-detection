import pandas as pd
import numpy as np
import os
import re
import nltk
import logging
from typing import Optional
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


class DataPreprocessor:

    def __init__(
        self,
        train_path: str,
        test_path: str,
        output_dir: str = "data/preprocessed_data"
    ) -> None:
        self.train_path: str = train_path
        self.test_path: str = test_path
        self.output_dir: str = output_dir

        self.logger: logging.Logger = self._setup_logger()

        self.lemmatizer: Optional[WordNetLemmatizer] = None
        self.stop_words: Optional[set[str]] = None

    def _setup_logger(self) -> logging.Logger:
        logger: logging.Logger = logging.getLogger("data_preprocessing")
        logger.setLevel(logging.INFO)

        ch: logging.Handler = logging.StreamHandler()
        formatter: logging.Formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)

        if not logger.handlers:
            logger.addHandler(ch)

        return logger

    # ---------------- NLTK ----------------
    def download_nltk_resources(self) -> None:
        try:
            nltk.download('wordnet', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.logger.info("NLTK resources downloaded")
        except Exception as e:
            self.logger.error("Failed to download NLTK resources")
            raise RuntimeError("Failed to download NLTK resources") from e

    def init_nlp_tools(self) -> None:
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words("english"))
            self.logger.info("NLP tools initialized")
        except Exception as e:
            self.logger.error("Failed to initialize NLP tools")
            raise RuntimeError("Failed to initialize NLP tools") from e

    # ---------------- TEXT CLEANING ----------------
    def clean_text(self, text: str) -> str:
        try:
            text = str(text).lower()

            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            text = re.sub(r'[^\w\s]', ' ', text)
            text = ''.join([i for i in text if not i.isdigit()])

            text = " ".join([
                word for word in text.split()
                if self.stop_words and word not in self.stop_words
            ])

            text = " ".join([
                self.lemmatizer.lemmatize(word)
                for word in text.split()
                if self.lemmatizer is not None
            ])

            return text

        except Exception:
            self.logger.warning(f"Failed to clean text: {text}")
            return ""

    # ---------------- DATAFRAME PROCESS ----------------
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if 'content' not in df.columns:
                raise KeyError("Column 'content' not found")

            self.logger.info("Starting preprocessing on dataframe")

            df['content'] = df['content'].apply(self.clean_text)

            df['content'] = df['content'].apply(
                lambda x: x if len(str(x).split()) >= 3 else np.nan
            )

            df.dropna(inplace=True)

            self.logger.info(f"Preprocessing complete. Rows remaining: {len(df)}")
            return df

        except Exception as e:
            self.logger.error("Error during dataframe preprocessing")
            raise RuntimeError("Error during dataframe preprocessing") from e

    # ---------------- SAVE ----------------
    def save_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        try:
            os.makedirs(self.output_dir, exist_ok=True)

            train_path: str = os.path.join(self.output_dir, "train_processed.csv")
            test_path: str = os.path.join(self.output_dir, "test_processed.csv")

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            self.logger.info(f"Files saved in {self.output_dir}")

        except Exception as e:
            self.logger.error("Error saving processed files")
            raise RuntimeError("Error saving processed files") from e

    # ---------------- PIPELINE ----------------
    def run(self) -> None:
        try:
            self.logger.info("Pipeline started")

            self.download_nltk_resources()
            self.init_nlp_tools()

            if not os.path.exists(self.train_path):
                raise FileNotFoundError(f"{self.train_path} not found")

            if not os.path.exists(self.test_path):
                raise FileNotFoundError(f"{self.test_path} not found")

            train_data: pd.DataFrame = pd.read_csv(self.train_path)
            test_data: pd.DataFrame = pd.read_csv(self.test_path)

            self.logger.info(f"Train shape: {train_data.shape}")
            self.logger.info(f"Test shape: {test_data.shape}")

            train_data = self.preprocess_dataframe(train_data)
            test_data = self.preprocess_dataframe(test_data)

            self.save_data(train_data, test_data)

            self.logger.info("Pipeline completed successfully ✅")

        except Exception as e:
            self.logger.critical("Pipeline execution failed ❌")
            self.logger.exception(e)
            raise RuntimeError("Pipeline execution failed") from e


# ---------------- RUN ----------------
if __name__ == "__main__":
    processor: DataPreprocessor = DataPreprocessor(
        train_path='./data/raw_data/train.csv',
        test_path='./data/raw_data/test.csv'
    )
    processor.run()