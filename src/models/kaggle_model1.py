import dataclasses as dc
from typing import Any, Dict, Optional

import pandas as pd

from src.data.repository import Repository
import src.models.base_model as bm


def train(
    repository: Repository,
    config: Dict[str, Any],
    training_logger: Optional[bm.SupportsLogging] = None,
) -> None:
    pass


def validate(repository: Repository, config: Dict[str, Any]) -> None:
    pass


class KaggleModel1(bm.Model):
    def __init__(self) -> None:
        super().__init__()

    def train(
        self,
        repository: Repository,
        config: Dict[str, Any],
        exp_logger: bm.SupportsLogging,
    ) -> None:
        ...

    def inference(self, config: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
        ...


if __name__ == "__main__":
    bm.train = train
    bm.validate = validate
    bm.main()
