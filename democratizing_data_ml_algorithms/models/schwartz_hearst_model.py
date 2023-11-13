# This model is a naive baseline model that uses the schwartz hearst algorithm
# extract entities from the text. This model will obviously extract entities
# that are not datasets, but it will give a good baseline for the recall for
# models that do binary entity classification and use the schwartz hearst
# algorithm to extract entities.

from itertools import chain
from typing import Any, Dict, List

import pandas as pd


from democratizing_data_ml_algorithms.data.repository import Repository
from democratizing_data_ml_algorithms.models.base_model import Model
from democratizing_data_ml_algorithms.models.schwartz_hearst import (
    extract_abbreviation_definition_pairs,
)


class SchwartzHearstModel(Model):
    def train(self, repository: Repository, config: Dict[str, Any]) -> None:
        pass

    def inference_string(self, config: Dict[str, Any], text: str) -> str:
        raise NotImplementedError()

    def inference(self, config: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
        def infer_sample(text: List[Dict[str, str]]) -> str:
            extractions = list(chain.from_iterable(
                extract_abbreviation_definition_pairs(doc_text=text)
            ))



            return (
                "|".join(extractions),
                "|".join("-" for _ in range(len(extractions))),
                "|".join("1.0" for _ in range(len(extractions))),
            )

        df[
            ["model_prediction", "prediction_snippet", "prediction_confidence"]
        ] = df.apply(
            lambda x:infer_sample(
                x["text"],
            ),
            result_type="expand",
            axis=1,
        )

        return df
