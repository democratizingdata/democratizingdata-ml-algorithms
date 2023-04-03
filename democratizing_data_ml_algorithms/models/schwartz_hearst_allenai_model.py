# This model is a naive baseline model that uses the schwartz hearst algorithm
# extract entities from the text. This model will obviously extract entities
# that are not datasets, but it will give a good baseline for the recall for
# models that do binary entity classification and use the schwartz hearst
# algorithm to extract entities.

from functools import partial
from itertools import chain
from typing import Any, Dict, List

import pandas as pd
import spacy


from democratizing_data_ml_algorithms.data.repository import Repository
from democratizing_data_ml_algorithms.models.base_model import Model
from democratizing_data_ml_algorithms.models.schwartz_hearst_allenai import (
    AbbreviationDetector,
    extract_abbreviation_definition_pairs,
)

from tqdm import tqdm

from pandarallel import pandarallel


def infer_sample(nlp, char_limit: int, text: List[Dict[str, str]]) -> str:

    extractions = dict()

    all_text = ""
    for section in text:
        all_text += " " + section["text"].replace("\n", " ").strip()

        if len(all_text) > char_limit:
            extractions.update(
                extract_abbreviation_definition_pairs(nlp, text=all_text[:char_limit])
            )
            all_text = all_text[char_limit:]

    while all_text:
        extractions.update(
            extract_abbreviation_definition_pairs(nlp, text=all_text[:char_limit])
        )
        all_text = all_text[char_limit:]

    return "|".join(chain.from_iterable(extractions.items()))


class SchwartzHearstModel_AllenAI(Model):
    # you can get the model strings/packages https://allenai.github.io/scispacy/

    def __init__(self, model: str = "en_core_sci_lg"):
        self.nlp = spacy.load(model)
        self.nlp.add_pipe("abbreviation_detector")

    def train(self, repository: Repository, config: Dict[str, Any]) -> None:
        pass

    def inference_string(self, config: Dict[str, Any], text: str) -> str:
        raise NotImplementedError()

    def inference_dataframe(
        self, config: Dict[str, Any], df: pd.DataFrame
    ) -> pd.DataFrame:

        infer_f = partial(infer_sample, self.nlp, config["char_limit"])

        tqdm.pandas()
        pandarallel.initialize(progress_bar=True, use_memory_fs=False)
        df["model_prediction"] = df["text"].parallel_apply(infer_f)
        # df["model_prediction"] = df["text"].progress_apply(infer_f)

        return df
