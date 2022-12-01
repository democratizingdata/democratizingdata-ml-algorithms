from typing import Iterable, Tuple

import pandas as pd

class Repository:

    def get_training_data_raw(self) -> Iterable[Tuple[str, str]]:
        raise NotImplementedError()

    def get_test_data_raw(self) -> Iterable[Tuple[str, str]]:
        raise NotImplementedError()

    def get_training_data_dataframe(self) -> pd.DataFrame:
        raise NotImplementedError()

    def get_test_data_dataframe(self) -> pd.DataFrame:
        raise NotImplementedError()