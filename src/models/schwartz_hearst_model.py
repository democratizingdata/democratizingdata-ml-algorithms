# This model is a naive baseline model that uses the schwartz hearst algorithm
# extract entities from the text. This model will obviously extract entities
# that are not datasets, but it will give a good baseline for the recall for
# models that do binary entity classification and use the schwartz hearst
# algorithm to extract entities.

from itertools import chain
from typing import Any, Dict, List

import pandas as pd


from src.data.repository import Repository
from src.models.base_model import Model
from src.models.schwartz_hearst import extract_abbreviation_definition_pairs


class SchwartzHearstModel(Model):
    def train(self, repository: Repository, config: Dict[str, Any]) -> None:
        pass

    def inference_string(self, config: Dict[str, Any], text: str) -> str:
        raise NotImplementedError()

    def inference(self, config: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
        def infer_sample(text: List[Dict[str, str]]) -> str:
            extractions = extract_abbreviation_definition_pairs(doc_text=text)

            return "|".join(chain.from_iterable(extractions.items()))

        df["model_prediction"] = df["text"].apply(infer_sample)

        return df
