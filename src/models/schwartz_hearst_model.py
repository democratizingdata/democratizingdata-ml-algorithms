# This model is a naive baseline model that uses the schwartz hearst algorithm
# extract entities from the text. This model will obviously extract entities
# that are not datasets, but it will give a good baseline for the recall for
# models that do binary entity classification and use the schwartz hearst
# algorithm to extract entities.

from typing import Any, Dict, List

import pandas as pd


from src.data.repository import Repository
from src.models.base_model import Model, Hyperparameters
from src.models.schwartz_hearst import extract_abbreviation_definition_pairs

class SchwartzHearstModel(Model):
    def train(self, repository: Repository, config: Hyperparameters) -> None:
        pass

    def inference_string(self, config: Dict[str, Any], text: str) -> str:
        raise NotImplementedError()

    def inference_dataframe(
        self, config: Dict[str, Any], df: pd.DataFrame
    ) -> pd.DataFrame:

        def infer_sample(text: List[Dict[str, str]]) -> str:
            predictions = []
            all_text = " ".join([s["text"].replace("\n", " ").strip() for s in text])
            extractions = extract_abbreviation_definition_pairs(doc_text=all_text)

            predictions = [e + "|" + extractions[e] for e in extractions]

            return "|".join(predictions)

        df["model_prediction"] = df["text"].progress_apply(infer_sample)

        return df