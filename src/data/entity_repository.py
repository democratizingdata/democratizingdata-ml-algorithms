from typing import Iterator, Tuple

import pandas as pd

from src.data.repository import Repository


class EntityRepository(Repository):
    def __init__(self):
        pass

    def get_training_data_raw(self, batch_size: int) -> Iterator[Tuple[str, str]]:
        raise NotImplementedError()

    def get_test_data_raw(self, batch_size: int) -> Iterator[Tuple[str, str]]:
        raise NotImplementedError()

    def get_vaidation_data_frame(self, batch_size: int) -> Iterator[Tuple[str, str]]:
        raise NotImplementedError()

    def get_training_data_dataframe(self, batch_size: int) -> Iterator[pd.DataFrame]:
        raise NotImplementedError()

    def get_test_data_dataframe(self, batch_size: int) -> Iterator[pd.DataFrame]:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return "EntityRepository"