import dataclasses as dc

import pandas as pd

from src.data.repository import Repository
from src.models.base_model import Model


class KaggleModel1(Model):
    def train(self):
        pass

    def inference_string(self, text: str) -> str:
        raise NotImplementedError()

    def inference_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
