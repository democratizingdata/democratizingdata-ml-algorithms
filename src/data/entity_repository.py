from itertools import islice
import os
from typing import Iterator, Tuple

import pandas as pd

from src.data.repository import Repository


class EntityRepository(Repository):
    def __init__(self):
        self.local = os.path.dirname(__file__)

        paths = [
            os.path.join(
                self.local, "../../data/entity_classification/ncses_priorities.csv"
            ),
            os.path.join(
                self.local, "../../data/entity_classification/original_entity_list.csv"
            ),
        ]

        all_data = pd.concat(
            [pd.read_csv(path) for path in paths], ignore_index=True
        ).drop_duplicates()

        self.training_data = all_data.sample(frac=0.8, random_state=0)
        self.test_data = all_data.drop(self.training_data.index)

    def get_training_data_raw(self, batch_size: int) -> Iterator[Tuple[str, str]]:
        raise NotImplementedError()

    def get_test_data_raw(self, batch_size: int) -> Iterator[Tuple[str, str]]:
        raise NotImplementedError()

    def get_vaidation_data_frame(self, batch_size: int) -> Iterator[Tuple[str, str]]:
        raise NotImplementedError()

    def get_training_data_dataframe(self, batch_size: int) -> Iterator[pd.DataFrame]:

        batched_df = self.training_data.iterrows()

        for _ in range(batch_size):
            batch = pd.DataFrame(
                list(map(lambda x: x[1], islice(batched_df, batch_size)))
            )
            yield batch

    def get_test_data_dataframe(self, batch_size: int) -> Iterator[pd.DataFrame]:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return "EntityRepository"


if __name__ == "__main__":
    repo = EntityRepository()
    df = repo.training_data
    from itertools import islice

    df_iter = df.iterrows()
    vals = [
        pd.DataFrame(list(map(lambda x: x[1], islice(df_iter, 6)))) for _ in range(20)
    ]
    print(vals[0].columns)
