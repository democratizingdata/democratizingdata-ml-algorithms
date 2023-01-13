import logging
from itertools import islice
import os
from typing import Iterator, Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

from src.data.repository import Repository

logger = logging.getLogger("entity_repository")

class EntityRepository(Repository):
    def __init__(self, rebalance:bool=False):
        self.local = os.path.dirname(__file__)

        self.paths = [
            os.path.join(
                self.local, "../../data/entity_classification/usda_priorities.csv"
            ),
            os.path.join(
                self.local, "../../data/entity_classification/ncses_priorities.csv"
            ),
            os.path.join(
                self.local, "../../data/entity_classification/original_entity_list.csv"
            ),
        ]
        self.train_dataframe_location = os.path.join(
            self.local, "../../data/entity_classification/training_data.csv"
        )
        self.test_dataframe_location = os.path.join(
            self.local, "../../data/entity_classification/test_data.csv"
        )

        if not os.path.exists(self.train_dataframe_location):
            self.build(rebalance)

    def get_training_data(self, batch_size: Optional[int] = None) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        def iter_f():
            for batch in pd.read_csv(self.train_dataframe_location, chunksize=batch_size):
                yield batch

        if batch_size:
            return iter_f()
        else:
            df = pd.read_csv(self.train_dataframe_location)
            return df

    def get_test_data(self, batch_size: Optional[int] = None) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:

        def iter_f():
            for batch in pd.read_csv(self.test_dataframe_location, chunksize=batch_size):
                yield batch

        if batch_size:
            return iter_f()
        else:
            df = pd.read_csv(self.test_dataframe_location)
            return df


    def build(self, rebalance:bool) -> None:
        """Builds a dataframe of the training/testing data and saves it to disk"""

        all_data = pd.concat(
            [pd.read_csv(path) for path in self.paths], ignore_index=True
        ).drop_duplicates().rename(columns={"long": "entity", "is_dataset": "label"})

        train_df, test_df = train_test_split(all_data, test_size=0.2, random_state=42)

        if rebalance:
            ros = RandomOverSampler(random_state=42, sampling_strategy=1)
            train_df, train_df["label"] = ros.fit_resample(train_df.drop(columns=["label"]), train_df["label"])

        train_df.to_csv(self.train_dataframe_location, index=False)
        test_df.to_csv(self.test_dataframe_location, index=False)


    def __repr__(self) -> str:
        return "entity_repository"


if __name__ == "__main__":
    repo = EntityRepository()
    df = repo.get_training_data()
    assert isinstance(df, pd.DataFrame), f"Should be a dataframe but is a {type(df)}"
