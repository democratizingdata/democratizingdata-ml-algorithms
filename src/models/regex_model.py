# This model is a baseline that uses the a regext to extract entities from the
# text. This model will extract entitites that are not datasets, but the goal
# is high recall. This model could be combined with an entity classifier to
# increast the precision of the model.
import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import regex as re
from pandarallel import pandarallel
from unidecode import unidecode
from tqdm import tqdm

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

CAPITALIZED_WORD = r"[A-Z][a-z]{2,}"
ACRONYM = r"[A-Z]{3,}"
APOSTROPHE_S = "'s"
OPTIONAL = lambda s : "(" + s + ")?"
OR = "|"
SPACE = r"\ "

# link to regex101: https://regex101.com/r/c903yg/1
ENTITY_PATTERN = "".join([
    # start of main capturing group
    "(",
        # First part must be one of the following:
        # - start with a capital letter and be followed by at least 2 lower case letters
        # - at least 3 captial letters
        # optionally an apostrophe and an s
        # ends with a space
        "(((" + CAPITALIZED_WORD + OR + ACRONYM + ")" + OPTIONAL(APOSTROPHE_S) + ")" + r"\ )",
        # Second part must be one of the following:
        # - a capital letter followed by at least 2 lower case letters
        # - one of the CONNECTING_WORDS followed by a space
        # - at least 3 captial letters
        # optionally end with an apostrophe and an s
        # followed by an optional space
        # this patten can repeat 2 or more times
        "((" + CAPITALIZED_WORD + OR + ACRONYM + OR + CONNECTING_PATTERN + ")" + OPTIONAL(APOSTROPHE_S) + OPTIONAL(SPACE) + r"){2,}",
        # Third part is a logical lookbehind that exlcudes the last match from
        # the second part from being one of the CONNECTING_WORDS
        r"(?<!" + CONNECTING_PATTERN + ")",
        # Fourth part is an optional pattern that captures an opening and closing
        # parenthesis with at least 3 captial letters in between
        r"(\([A-Z]{3,}\))?",
    # end of main capturing group
    ")",
])

class RegexModel(Model):

    def __init__(self, config:Dict[str, str]) -> None:
        regex_pattern = config["regex_pattern"] if "regex_pattern" in config else ENTITY_PATTERN
        keywords = config["keywords"] if "keywords" in config else []

        if keywords:
            keyword_pattern = r"|".join(list(map(RegexModel.regexify_keyword, keywords)))

            regex_pattern = "" if regex_pattern == "" else regex_pattern[:-1] + "|"
            keyword_pattern = "(" + keyword_pattern + ")"

            if regex_pattern == "":
                regex_pattern = keyword_pattern
            else:
                regex_pattern = regex_pattern + keyword_pattern + ")"

        self.entity_pattern = re.compile(regex_pattern)


    def train(self, repository: Repository, config: Dict[str, Any], exp_logger:SupportsLogging) -> None:
        pass

    def inference(
        self, config: Dict[str, Any], df: pd.DataFrame
    ) -> pd.DataFrame:

        def infer_f(text: str) -> str:
            # extract_f = lambda m: m[0] if len(m) > 0 else ""
            # matches = set(match[0] for match in self.entity_pattern.findall(text))

            matches = set(map(
                lambda m: m[0],
                filter(
                    lambda m: len(m) > 0,
                    self.entity_pattern.findall(text)
                ),
            ))

            matches_with_parens = list(filter(lambda m: "(" in m and ")" in m, matches))

            # Really Great Data (RGD) -> Really Great Data|RGD
            def split_parens(m: str) -> str:
                long_form, short_form = m.split("(")
                return long_form.strip() + "|" + short_form[:-1]
            split_matches_parens = list(map(split_parens, matches_with_parens))

            return "|".join(list(matches) + split_matches_parens)

        # tqdm.pandas()
        # pandarallel.initialize(progress_bar=True, use_memory_fs=False)
        # df["model_prediction"] = df["text"].parallel_apply(infer_f)
        df["model_prediction"] = df["text"].apply(infer_f)

        return df


    @staticmethod
    def regexify_char(c:str) -> str:
        if c.isalpha():
            return f"[{c.upper()}|{c.lower()}]"
        else:
            return c

    def regexify_first_char(c:str) -> str:
        if len(c) == 1:
            return RegexModel.regexify_char(c)
        else:
            return RegexModel.regexify_char(c[0]) + c[1:]

    def regexify_keyword(keyword:str) -> str:
        tokens = keyword.strip().split()

        sub_parens = lambda s: s.replace("(", r"\(").replace(")", r"\)")

        if len(tokens) == 1:
            # If this is a single word and its all caps, then we don't want to
            # regexify it, it is an acronym
            if keyword.isupper():
                return keyword
            else:
                return sub_parens("".join(list(map(RegexModel.regexify_char, keyword))))
        else:
            return sub_parens(" ".join(list(map(RegexModel.regexify_first_char, tokens))))
