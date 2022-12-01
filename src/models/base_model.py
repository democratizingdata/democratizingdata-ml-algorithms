import dataclasses as dc

import pandas as pd

from src.data.repository import Repository

@dc.dataclass
class Model:
    storage_dir: str
    repository: Repository

    def train(self) -> None:
        raise NotImplementedError()

    def inference_string(self, text: str) -> str:
        raise NotImplementedError()

    def inference_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()