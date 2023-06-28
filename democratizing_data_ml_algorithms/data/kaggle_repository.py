from functools import partial
import json
import logging
import os
from unidecode import unidecode
from typing import Dict, Iterator, List, Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from democratizing_data_ml_algorithms.data.repository import Repository

logger = logging.getLogger("kaggle_repository")


class KaggleRepository(Repository):
    """A repository for the kaggle data set."""

    def __init__(self):
        self.local = os.path.dirname(__file__)
        self.train_labels_location = os.path.join(
            self.local, "../../data/kaggle/train.csv"
        )
        self.train_files_location = os.path.join(self.local, "../../data/kaggle/train")
        self.validation_files_location = os.path.join(
            self.local, "../../data/kaggle/validation"
        )
        self.train_dataframe_location = os.path.join(
            self.local, "../../data/kaggle/train_dataframe.csv"
        )

        self.test_dataframe_location = os.path.join(
            self.local, "../../data/kaggle/test_dataframe.csv"
        )

        self.validation_dataframe_location = os.path.join(
            self.local, "../../data/kaggle/validation.csv"
        )

        assert os.path.exists(
            self.train_labels_location
        ), "train.csv not found, please download kaggle data"

        assert os.path.exists(
            self.train_files_location
        ), "train folder not found, please download kaggle data"
        if not os.path.exists(self.train_dataframe_location):
            logger.info("Building train/test dataframes")
            self.build()

    def get_sample_text(self, parent_dir: str, id: str) -> Dict[str, str]:
        with open(os.path.join(parent_dir, (id + ".json"))) as fp:
            data = json.load(fp)
        return data

    def process_text(self, text: List[Dict[str, str]]) -> str:
        all_text = " ".join([x["text"].replace("\n", " ").strip() for x in text])
        return unidecode(all_text)

    def retrieve_text(self, parent_dir: str, row: pd.DataFrame) -> str:
        doc_id = str(row["id"])
        json_text = self.get_sample_text(parent_dir, doc_id)
        return self.process_text(json_text)

    def get_training_data(
        self, batch_size: Optional[int] = None, balance_labels: Optional[bool] = False
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        if balance_labels:
            raise ValueError("Label balancing not supported for kaggle data")

        fetch_f = partial(self.retrieve_text, self.train_files_location)

        def iter_f():
            for batch in pd.read_csv(
                self.train_dataframe_location, chunksize=batch_size
            ):
                batch["text"] = batch.apply(fetch_f, axis=1)
                yield batch

        if batch_size:
            return iter_f()
        else:
            df = pd.read_csv(self.train_dataframe_location)
            df["text"] = df.apply(fetch_f, axis=1)
            return df

    def get_test_data(
        self, batch_size: Optional[int] = None
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        fetch_f = partial(self.retrieve_text, self.train_files_location)

        def iter_f():
            for batch in pd.read_csv(
                self.test_dataframe_location, chunksize=batch_size
            ):
                batch["text"] = batch.apply(fetch_f, axis=1)
                yield batch

        if batch_size:
            return iter_f()
        else:
            df = pd.read_csv(self.test_dataframe_location)
            df["text"] = df.apply(fetch_f, axis=1)
            return df

    def get_validation_data(
        self, batch_size: Optional[int] = None
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        fetch_f = partial(self.retrieve_text, self.validation_files_location)

        def iter_f():
            for batch in pd.read_csv(
                self.validation_dataframe_location, chunksize=batch_size
            ):
                batch = batch.loc[:, ["Id", "PredictionString"]].rename(
                    columns={"Id": "id", "PredictionString": "label"}
                )
                batch["text"] = batch.apply(fetch_f, axis=1)
                yield batch

        if batch_size:
            return iter_f()
        else:
            df = pd.read_csv(self.validation_dataframe_location)
            df = df.loc[:, ["Id", "PredictionString"]].rename(
                columns={"Id": "id", "PredictionString": "label"}
            )
            df["text"] = df.apply(fetch_f, axis=1)
            return df

    def build(self) -> None:
        """Builds a dataframe of the training/testing/validation data and saves it to disk"""
        raw = pd.read_csv(self.train_labels_location)

        def aggregate_clean_label(row: pd.DataFrame):
            labels = list(
                map(lambda x: x.lower().strip(), row["dataset_label"].unique())
            )
            return "|".join(labels)

        unique_labels = raw.groupby("Id").apply(aggregate_clean_label)

        all_df = pd.DataFrame({"id": raw["Id"].unique()})
        all_df["label"] = all_df["id"].apply(lambda x: unique_labels[x])

        train_df, test_df = train_test_split(all_df, test_size=0.2, random_state=42)

        logger.info("Saving training data to: ", self.train_dataframe_location)
        train_df.to_csv(self.train_dataframe_location, index=False)
        logger.info("Saving test data to: ", self.test_dataframe_location)
        test_df.to_csv(self.test_dataframe_location, index=False)

    def __repr__(self) -> str:
        return "kaggle_repository"


if __name__ == "__main__":
    repo = KaggleRepository()
    df = repo.get_training_data_dataframe(batch_size=10)
    print(next(df))
