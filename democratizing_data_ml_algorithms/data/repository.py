from enum import Enum
from typing import Iterator, Optional, Protocol, Tuple, Union

import pandas as pd


class SnippetRepositoryMode(Enum):
    NER = "ner"
    CLASSIFICATION = "classification"
    MASKED_LM = "masked_lm"


class Repository(Protocol):
    def get_training_data(
        self, batch_size: Optional[int] = None, balance_labels: Optional[bool] = False
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        ...

    def get_test_data(
        self, batch_size: Optional[int] = None
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        ...

    def get_validation_data(
        self, batch_size: Optional[int] = None
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        ...
