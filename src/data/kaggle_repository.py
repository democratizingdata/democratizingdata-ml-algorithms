import json
import os
from typing import Dict, Iterator, List, Tuple

import pandas as pd

from src.data.repository import Repository


class KaggleRepository(Repository):
    def __init__(self):
        self.local = os.path.dirname(__file__)
        self.train_labels_location = os.path.join(
            self.local, "../../data/kaggle/train.csv"
        )
        self.train_files_location = os.path.join(self.local, "../../data/kaggle/train")
        self.train_dataframe_location = os.path.join(
            self.local, "../../data/kaggle/train_dataframe.pkl"
        )
        assert os.path.exists(
            self.train_labels_location
        ), "train.csv not found, please download kaggle data"
        assert os.path.exists(
            self.train_files_location
        ), "train folder not found, please download kaggle data"

    def get_training_data_raw(self, batch_size: int) -> Iterator[Tuple[str, str]]:
        raise NotImplementedError()

    def get_test_data_raw(self, batch_size: int) -> Iterator[Tuple[str, str]]:
        raise NotImplementedError()

    def get_vaidation_data_frame(self, batch_size: int) -> Iterator[Tuple[str, str]]:
        raise NotImplementedError()

    def get_training_data_dataframe(self, batch_size: int) -> Iterator[pd.DataFrame]:
        for batch in pd.read_csv(self.train_labels_location, chunksize=batch_size):
            batch["texts"] = batch.apply(self.append_training_data, axis=1)
            yield batch

    def append_training_data(self, row: pd.DataFrame) -> None:
        return self.get_training_sample(row["Id"])

    def get_training_sample(self, id: str) -> Dict[str, str]:
        with open(os.path.join(self.train_files_location, (id + ".json"))) as fp:
            data = json.load(fp)
        return data

    def get_test_data_dataframe(self, batch_size: int) -> Iterator[pd.DataFrame]:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return "KaggleRepository"


if __name__ == "__main__":
    repo = KaggleRepository()
    df = repo.get_training_data_dataframe(1)
    print(type(df))
