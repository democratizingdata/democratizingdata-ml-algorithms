
from itertools import islice
import logging
from typing import Any, Dict, Optional

import pandas as pd


from src.data.repository import Repository
import src.models.base_model as bm

logger = logging.getLogger("ner_model")

def validate_config(config: Dict[str, Any]) -> None:

    expected_keys = {

    }

    missing_keys = expected_keys - set(config.keys())
    assert not missing_keys, f"Missing keys: {missing_keys}"

def train(
    repository: Repository,
    config: Dict[str, Any],
    training_logger: Optional[bm.SupportsLogging] = None,
) -> None:

        validate_config(config)
        training_logger.log_parameters(config)


def vaidate(repository: Repository, config: Dict[str, Any]) -> None:

    validate_config(config)


# https://stackoverflow.com/a/62913856/2691018
def batcher(iterable, batch_size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


class NERModel_pytorch(bm.Model):

    def inference(self, config: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
        return super().inference(config, df)


    def train(self, repository: Repository, config: Dict[str, Any], training_logger: bm.SupportsLogging) -> None:
        return super().train(repository, config, training_logger)