import dataclasses as dc
from typing import Any, Dict

import pandas as pd

from src.data.repository import Repository


@dc.dataclass
class Hyperparameters:
    pretrained_model: str
    save_model: bool
    model_path: str
    optimizer: Any
    learning_rate: float
    num_epochs: int
    batch_size: int
    training_logger: Any  # this should be comet, wandb, tensorboard, etc.


class Model:
    def train(self, repository: Repository, config: Hyperparameters) -> None:
        raise NotImplementedError()

    def inference_string(self, config: Dict[str, Any], text: str) -> str:
        raise NotImplementedError()

    def inference_dataframe(
        self, config: Dict[str, Any], df: pd.DataFrame
    ) -> pd.DataFrame:
        raise NotImplementedError()
