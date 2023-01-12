# This model is a baseline that uses the a regext to extract entities from the
# text. This model will extract entitites that are not datasets, but the goal
# is high recall. This model could be combined with an entity classifier to
# increast the precision of the model.
import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import regex as re
from unidecode import unidecode

from src.data.repository import Repository
from src.models.base_model import Model, SupportsLogging
import src.evaluate.model as em

logger = logging.getLogger("RegexModel")


def validate_config(config: Dict[str, Any]) -> None:

    expected_keys = [
        "eval_path",
    ]

    for key in expected_keys:
        assert key in config, f"Missing key {key} in config"


def train(repository: Repository, config: Dict[str, Any], training_logger: Optional[SupportsLogging]=None) -> None:
    pass

def validate(repository: Repository, config: Dict[str, Any] = dict()) -> None:
    model = RegexModel(config)
    model_evaluation = em.evaluate_model(repository, model, config)

    print(model_evaluation)

    logger.info(f"Saving evaluation to {config['eval_path']}")
    with open(config["eval_path"], "w") as f:
        json.dump(model_evaluation.to_json(), f)


CONNECTING_WORDS = [
    "and",
    "for",
    "in",
    "of",
    "the",
]

# this makes the string r"|{val}\ " for each of the CONNECTING_WORDS
CONNECTING_PATTERN = r"|".join(list(c + r"\ " for c in CONNECTING_WORDS))


ENTITY_PATTERN = "".join([
    # start of main capturing group
    "(",
        # First part must start with a capital letter and be followed by at
        # least 2 lower case letters and a space
        r"([A-Z][a-z]{2,}\ )",
        # Second part must be one of the following:
        # - a capital letter followed by at least 2 lower case letters
        # - one of the CONNECTING_WORDS followed by a space
        # followed by an optional space
        # this patten can repeat 2 or more times
        r"(([A-Z][a-z]{2,}|" + CONNECTING_PATTERN + r")[\ ]?){2,}",
        # Third part is a logical lookbehind that exlcudes the last match from
        # the second part from being one of the CONNECTING_WORDS
        r"(?<!" + CONNECTING_PATTERN + r")",
        # Fourth part is an optional pattern that captures an opening and closing
        # parenthesis with at least 3 captial letters in between
        r"(\([A-Z]{3,}\))?",
    # end of main capturing group
    ")",
])

class RegexModel(Model):

    def __init__(self, config:Dict[str, str]) -> None:
        regex_pattern = config["regex_pattern"] if "regex_pattern" in config else ENTITY_PATTERN
        self.entity_pattern = re.compile(regex_pattern)


    def train(self, repository: Repository, config: Dict[str, Any], exp_logger:SupportsLogging) -> None:
        pass

    def inference(
        self, config: Dict[str, Any], df: pd.DataFrame
    ) -> pd.DataFrame:

        def infer_f(text: str) -> str:
            matches = set(match[0] for match in self.entity_pattern.findall(text))
            return "|".join(matches)

        df["model_prediction"] = df["text"].apply(infer_f)

        return df

