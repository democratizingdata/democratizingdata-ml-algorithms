# BSD 3-Clause License

# Copyright (c) 2023, AUTHORS
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Extract entities from text using a regex.

This model is a baseline that uses the a regex to extract entities from the
text. This model will extract entitites that are not datasets, but the goal
is high recall. This model could be combined with an entity classifier to
increast the precision of the model.

Example:

    >>> import pandas as pd
    >>> import democratizing_data_ml_algorithms.models.regex_model as rm
    >>> df = pd.DataFrame({"text": ["This is a sentence with an entity in it."]})
    >>> config = {"regex_pattern": r"entity"}
    >>> df = rm.inference(config, df)
"""


import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import regex as re
from unidecode import unidecode
from tqdm import tqdm

import democratizing_data_ml_algorithms.models.base_model as bm
import democratizing_data_ml_algorithms.data.repository as repo

logger = logging.getLogger("RegexModel")

EXPECTED_KEYS = {}


def train(
    repository: repo.Repository,
    config: Dict[str, Any],
    training_logger: Optional[bm.SupportsLogging] = None,
) -> None:
    """Top level function for training. NOT IMPLEMENTED."""

    raise NotImplementedError("RegexModel does not support training")


def inference(config: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """Top level function for inference.

    Args:
        config (Dict[str, Any]): Dictionary of inference configuration
        df (pd.DataFrame): Dataframe to perform inference on. The dataframe
                           should have a column called "text" that contains
                           the text to perform inference on.

    Returns:
        pd.DataFrame: Dataframe with inference results. There will be two
                      new columns added to the dataframe: "model_prediction"
                      and "prediction_snippet". The "model_prediction" column
                      will contain the extracted entities separated by a "|".
                      The "prediction_snippet" column will contain the
                      sentence that the entity was extracted from.
    """

    bm.validate_config(EXPECTED_KEYS, config)
    model = RegexModel(config)
    return model.inference(config, df)


# Default regex pattern for extracting entities from text ======================
CONNECTING_WORDS = [
    "and",
    "for",
    "in",
    "of",
    "the",
    "on",
    "to",
]

# this makes the string r"|{val}\ " for each of the CONNECTING_WORDS
CONNECTING_PATTERN = r"|".join(list(c + r"\ " for c in CONNECTING_WORDS))

CAPITALIZED_WORD = r"[A-Z][a-z]{2,}"
ACRONYM = r"[A-Z]{3,}"
APOSTROPHE_S = "'s"
OPTIONAL = lambda s: "(" + s + ")?"
OR = "|"
SPACE = r"\ "

# link to regex101: https://regex101.com/r/c903yg/1
ENTITY_PATTERN = "".join(
    [
        # start of main capturing group
        "(",
        # First part must be one of the following:
        # - start with a capital letter and be followed by at least 2 lower case letters
        # - at least 3 captial letters
        # optionally an apostrophe and an s
        # ends with a space
        "((("
        + CAPITALIZED_WORD
        + OR
        + ACRONYM
        + ")"
        + OPTIONAL(APOSTROPHE_S)
        + ")"
        + r"\ )",
        # Second part must be one of the following:
        # - a capital letter followed by at least 2 lower case letters
        # - one of the CONNECTING_WORDS followed by a space
        # - at least 3 captial letters
        # optionally end with an apostrophe and an s
        # followed by an optional space
        # this patten can repeat 2 or more times
        "(("
        + CAPITALIZED_WORD
        + OR
        + ACRONYM
        + OR
        + CONNECTING_PATTERN
        + ")"
        + OPTIONAL(APOSTROPHE_S)
        + OPTIONAL(SPACE)
        + r"){2,}",
        # Third part is a logical lookbehind that exlcudes the last match from
        # the second part from being one of the CONNECTING_WORDS
        r"(?<!" + CONNECTING_PATTERN + ")",
        # Fourth part is an optional pattern that captures an opening and closing
        # parenthesis with at least 3 captial letters in between
        r"(\([A-Z]{3,}\))?",
        # end of main capturing group
        ")",
    ]
)
# Default regex pattern for extracting entities from text ======================


class RegexModel(bm.Model):
    """Extract entities from text using a regex.

    This class will extract entities from text using a regex pattern. The
    regex pattern can be configured using the "regex_pattern" key in the
    configuration dictionary. The default regex pattern is defined above
    as ENTITY_PATTERN. The regex pattern is compiled using the regex library
    and the compiled pattern is stored in the `entity_pattern` attribute.

    Optionally, the regex pattern can be configured using the "keywords" key
    in the configuration dictionary. The "keywords" key should be a list of
    strings. Each string in the list will be converted to a regex pattern
    that will be used to extract entities from the text. The regex pattern
    for each keyword is defined as follows:

    -   If the keyword is a single word and it is all caps, then the keyword
        will not be converted to a regex pattern, it will be treated as an
        acronym.
    -   If the keyword is a single word and it is not all caps, then the
        keyword will be converted to a regex pattern that matches the keyword,
        ignoring the case for the keyword.
    -   If the keyword is multiple words, then the keyword will be converted
        to a regex pattern that matches the keyword, ignoring the case for
        the first letter of each word in the keyword.

    Example:
        >>> import pandas as pd
        >>> import democratizing_data_ml_algorithms.models.regex_model as rm
        >>> df = pd.DataFrame({"text": ["This is a sentence with an entity in it."]})
        >>> config = {"regex_pattern": r"entity"}
        >>> df = rm.inference(config, df)

    Attributes:
        entity_pattern (re.Pattern): Compiled regex pattern for extracting entities

    """

    def __init__(self, config: Dict[str, str]) -> None:
        """Initializes the RegexModel using a regex pattern or keywords."""

        regex_pattern = config.get("regex_pattern", ENTITY_PATTERN)
        keywords = config.get("keywords", [])

        if keywords:
            keyword_pattern = r"|".join(
                list(map(RegexModel.regexify_keyword, keywords))
            )

            regex_pattern = "" if regex_pattern == "" else regex_pattern[:-1] + "|"
            keyword_pattern = "(" + keyword_pattern + ")"

            if regex_pattern == "":
                regex_pattern = keyword_pattern
            else:
                regex_pattern = regex_pattern + keyword_pattern + ")"

        self.entity_pattern = re.compile(regex_pattern)

    def train(
        self,
        repository: repo.Repository,
        config: Dict[str, Any],
        exp_logger: bm.SupportsLogging,
    ) -> None:
        raise NotImplementedError("RegexModel does not support training")

    def inference(self, config: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
        """Performs inference on a dataframe.

        Args:
            config (Dict[str, Any]): Dictionary of inference configuration
            df (pd.DataFrame): Dataframe to perform inference on there should
                               be a column called "text" that contains the text

        Returns:
            pd.DataFrame: Dataframe with inference results. There will be two
                          new columns added to the dataframe: "model_prediction"
                          and "prediction_snippet". The "model_prediction" column
                          will contain the extracted entities separated by a "|".
                          The "prediction_snippet" column will contain the
                          sentence that the entity was extracted from.
        """

        def infer_f(text: str) -> str:
            matches_snippets = list(
                map(
                    lambda match: (
                        match[0].strip(),
                        RegexModel.extract_context(text, match),
                    ),
                    filter(
                        lambda m: bool(m), self.entity_pattern.finditer(unidecode(text))
                    ),
                )
            )

            matches, snippets = zip(*matches_snippets) if matches_snippets else ([], [])
            return "|".join(matches), "|".join(snippets), "|".join("1.0" for _ in matches)

        df[["model_prediction", "prediction_snippet", "prediction_confidence"]] = df.apply(
            lambda x: infer_f(x["text"]),
            result_type="expand",
            axis=1,
        )

        return df

    @staticmethod
    def extract_context(text: str, match: re.Match) -> str:
        """Extracts the sentence that the match was found in."""
        sentence_boundary = ". "
        start, end = match.span()
        sent_start = text.rfind(sentence_boundary, 0, start)
        sent_end = text.find(sentence_boundary, end, len(text))
        return text[sent_start + len(sentence_boundary) : sent_end + 1].strip()

    @staticmethod
    def regexify_char(c: str) -> str:
        """Converts a character to a regex pattern that matches the character."""
        if c.isalpha():
            return f"[{c.upper()}|{c.lower()}]"
        else:
            return c

    @staticmethod
    def regexify_first_char(text: str) -> str:
        """Converts the first character of a string to a regex pattern that matches the character."""
        if len(text) == 1:
            return RegexModel.regexify_char(text)
        else:
            return RegexModel.regexify_char(text[0]) + text[1:]

    @staticmethod
    def regexify_keyword(keyword: str) -> str:
        """Converts a keyword to a regex pattern that matches the keyword."""
        tokens = keyword.strip().split()

        sub_parens = lambda s: s.replace("(", r"\(").replace(")", r"\)")

        if len(tokens) == 1:
            # If this is a single word and its all caps, then we don't want to
            # regexify it, it is an acronym
            if keyword.isupper():
                return sub_parens(keyword)
            else:
                return sub_parens("".join(list(map(RegexModel.regexify_char, keyword))))
        else:
            return sub_parens(
                " ".join(list(map(RegexModel.regexify_first_char, tokens)))
            )


if __name__ == "__main__":  # pragma: no cover
    bm.train = train
    bm.inference = inference
    bm.main()
