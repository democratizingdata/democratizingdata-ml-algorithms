from typing import Any, Dict
import pandas as pd

from src.data.repository import Repository

class Model:
    def train(self, repository:Repository, config:Dict[str, Any]) -> None:
        raise NotImplementedError()

    def inference_string(self, config:Dict[str, Any], text: str) -> str:
        raise NotImplementedError()

    def inference_dataframe(self, config:Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
