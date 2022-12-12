import json
import os
from typing import Iterable, Tuple

import pandas as pd

from src.data.repository import Repository

class KaggleRepository(Repository):

    def __init__(self):
        pass

    def get_training_data_raw(self, batch_size: int) -> Iterable[Tuple[str, str]]:
        raise NotImplementedError()

    def get_test_data_raw(self, batch_size: int) -> Iterable[Tuple[str, str]]:
        raise NotImplementedError()

    def get_training_data_dataframe(self, batch_size: int) -> Iterable[pd.DataFrame]:
        df_labels = pd.read_csv(
            os.path.join(
                os.path.dirname(__file__),
                "../../data/kaggle/train.csv"
            )
        )

        text_dict = dict(Id=[], texts=[])
        train_files_dir = os.path.join(
            os.path.dirname(__file__),
            "../../data/kaggle/train"
        )
        for f in os.listdir(train_files_dir):
            with open(os.path.join(train_files_dir, f), "r") as f:
                json_data = json.load(f)
                text_dict["texts"].append(" ".join([l["text"].replace("\n", "").strip() for l in json_data]))
                text_dict["Id"].append(os.path.basename(f.name).split(".")[0])

        df_texts = pd.DataFrame(text_dict)
        print(df_texts.head(5))
        df = pd.merge(df_labels, df_texts, on="Id")

        return iter([df.loc[:, ["Id", "texts", "dataset_label"]],])

    def get_test_data_dataframe(self, batch_size: int) -> Iterable[pd.DataFrame]:
        raise NotImplementedError()



if __name__=="__main__":
    repo = KaggleRepository()
    df = repo.get_training_data_dataframe(1)
    print(df)