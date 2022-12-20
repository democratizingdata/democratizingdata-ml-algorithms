import dataclasses as dc
from typing import Any, Dict

import pandas as pd

from src.data.repository import Repository
from src.models.base_model import Model, Hyperparameters


class KaggleModel1(Model):
    def train(self, repository: Repository, config: Hyperparameters) -> None:
        raise NotImplementedError()

    def inference_string(self, config: Dict[str, Any], text: str) -> str:
        raise NotImplementedError()

    def inference_dataframe(
        self, config: Dict[str, Any], df: pd.DataFrame
    ) -> pd.DataFrame:
        raise NotImplementedError()
