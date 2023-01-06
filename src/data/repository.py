from typing import Iterator, Optional, Tuple, Union

import pandas as pd


class Repository:
    def get_training_data(self, batch_size: Optional[int] = None) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        raise NotImplementedError()

    def get_test_data(self, batch_size: Optional[int] = None) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        raise NotImplementedError()

    def get_validation(self, batch_size: Optional[int] = None) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        raise NotImplementedError()