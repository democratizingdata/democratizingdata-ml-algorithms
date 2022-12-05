import dataclasses as dc

import pandas as pd


@dc.dataclass
class Model:
    def train(self) -> None:
        raise NotImplementedError()

    def inference_string(self, text: str) -> str:
        raise NotImplementedError()

    def inference_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
