import copy
import logging
from typing import Iterator, List, Optional, Union

import pandas as pd

from democratizing_data_ml_algorithms.data.repository import Repository

logger = logging.getLogger("icsr_parquet_repository")


class IcsrParquetRepository(Repository):
    """A repository for the ICSR parquet data set."""

    def __init__(self, parquet_files: List[str]):
        self.parquet_files = parquet_files

    def get_training_data(
        self, batch_size: Optional[int] = None, balance_labels: Optional[bool] = False
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        raise NotImplementedError(
            "IcsrParquetRepository does not support training. Only get_test_data is supported."
        )

    def get_validation_data(
        self, batch_size: Optional[int] = None
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        raise NotImplementedError(
            "IcsrParquetRepository does not support training. Only get_test_data is supported."
        )

    def get_test_data(
        self, batch_size: Optional[int] = None
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        def iter_f():
            parquet_files = copy.copy(self.parquet_files)
            cache_dataframe = pd.DataFrame({"id": [], "text": []})
            while parquet_files or len(cache_dataframe) > batch_size:
                if len(cache_dataframe) < batch_size and parquet_files:
                    cache_dataframe = pd.concat(
                        [
                            cache_dataframe,
                            (
                                pd.read_parquet(parquet_files.pop())
                                .loc[:, ["Eid", "full_text"]]
                                .rename(columns={"Eid": "id", "full_text": "text"})
                            ),
                        ]
                    )

                batch = cache_dataframe.loc[:batch_size, :]
                yield batch

                cache_dataframe = cache_dataframe.loc[batch_size:, :]

            yield cache_dataframe

        if batch_size:
            return iter_f()
        else:
            return pd.concat([pd.read_parquet(f) for f in self.parquet_files])
