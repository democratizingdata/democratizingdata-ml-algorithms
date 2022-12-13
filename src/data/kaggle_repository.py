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
            self.local, "../../data/kaggle/train_dataframe.csv"
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

    def get_training_data_dataframe(self, batch_size: int) -> Iterator[pd.DataFrame]:

        if os.path.exists(self.train_dataframe_location):
            df = pd.read_csv(self.train_dataframe_location)
        else:

            df_labels = pd.read_csv(self.train_labels_location)

            text_dict: Dict[str, List[str]] = dict(Id=[], text=[])
            for fname in os.listdir(self.train_files_location):
                with open(os.path.join(self.train_files_location, fname), "r") as f:
                    json_data = json.load(f)
                    text_dict["text"].append(
                        " ".join(
                            [l["text"].replace("\n", "").strip() for l in json_data]
                        )
                    )
                    text_dict["Id"].append(os.path.basename(f.name).split(".")[0])

            df_texts = pd.DataFrame(text_dict)
            df = pd.merge(df_labels, df_texts, on="Id")
            df.to_csv(self.train_dataframe_location, index=False)

        return iter(
            [
                df.loc[:, ["Id", "text", "dataset_label"]],
            ]
        )

    def get_test_data_dataframe(self, batch_size: int) -> Iterator[pd.DataFrame]:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return "KaggleRepository"


if __name__ == "__main__":
    repo = KaggleRepository()
    df = next(repo.get_training_data_dataframe(1))
    print(df)
