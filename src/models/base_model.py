import dataclasses as dc
from typing import Any, Dict

import pandas as pd

from src.data.repository import Repository


class Model:
    def train(self, repository: Repository, config: Dict[str, Any]) -> None:
        raise NotImplementedError()

    def inference_string(self, config: Dict[str, Any], text: str) -> str:
        raise NotImplementedError()

    def inference_dataframe(
        self, config: Dict[str, Any], df: pd.DataFrame
    ) -> pd.DataFrame:
        raise NotImplementedError()


@dc.dataclass
class Hyperparameters:
    pretrained_model: str
    save_model: bool
    model_path: str
    optimizer: Any
    learning_rate: float
    num_epochs: int
    batch_size: int
