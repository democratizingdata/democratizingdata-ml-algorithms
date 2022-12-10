# Model notebook
# https://github.com/Coleridge-Initiative/rc-kaggle-models/blob/original_submissions/3rd%20Mikhail%20Arkhipov/3rd%20place%20coleridge.ipynb
# ==============================================================================
# Description of the model from the notebook: Brief Solution Description
#
# The solution is based on a simple heuristic: a capitalized sequence of words
# that includes a keyword and followed by parenthesis usually refer to a
# dataset. So, any sequence like
#
# ``` Xxx Xxx Keyword Xxx (XXX)```
#
# is a good candidate to be a dataset.
#
# All mentions of a given form are extracted to form a list of dataset names to
# look for. Each text in the test is checked for inclusion of the dataset name
# from the list. Every match is added to the prediction. Substring predictions
# are removed.
#
# Keywords list:
# - Study
# - Survey
# - Assessment
# - Initiative
# - Data
# - Dataset
# - Database
#
# Also, many data mentions refer to some organizations or systems. These
# mentions seem to be non-valid dataset names. To remove them the following list
# of stopwords is used:
# - lab
# - centre
# - center
# - consortium
# - office
# - agency
# - administration
# - clearinghouse
# - corps
# - organization
# - organisation
# - association
# - university
# - department
# - institute
# - foundation
# - service
# - bureau
# - company
# - test
# - tool
# - board
# - scale
# - framework
# - committee
# - system
# - group
# - rating
# - manual
# - division
# - supplement
# - variables
# - documentation
# - format
#
# To exclude mentions not related to data a simple count statistic is used:
#
#       N_{data}(str)
# F_d = -------------
#       N_{total}(str)
#
# where N_{data}(str) is the number of times the str occures with data
# word (parenthesis are dropped) and N_{total}(str) is the total number
# of times str present in texts. All mentions with  F_d<0.1  are dropped.

from collections import Counter, defaultdict
import re
from itertools import filterfalse
from typing import Any, Callable, Dict, Iterable, List, Set

import pandas as pd

from src.data.repository import Repository
from src.models.base_model import Model


class KaggleModel3(Model):
    """This class is based on the Kaggle model 3 notebook.

    Model 3 is a heuristic model, so it doesn't need to be "trained" in the
    deep learning sense. It will extract mentions that fit the format:

    ``` Xxx Xxx Keyword Xxx (XXX)```

    Where keyword is one of the keywords in the KaggleModel3.KEYWORDS list.
    """

    KEYWORDS = [
        "Study",
        "Survey",
        "Assessment",
        "Initiative",
        "Data",
        "Dataset",
        "Database",
    ]

    STOPWORDS_PAR = [
        " lab",
        "centre",
        "center",
        "consortium",
        "office",
        "agency",
        "administration",
        "clearinghouse",
        "corps",
        "organization",
        "organisation",
        "association",
        "university",
        "department",
        "institute",
        "foundation",
        "service",
        "bureau",
        "company",
        "test",
        "tool",
        "board",
        "scale",
        "framework",
        "committee",
        "system",
        "group",
        "rating",
        "manual",
        "division",
        "supplement",
        "variables",
        "documentation",
        "format",
    ]

    TOKENIZE_PAT = re.compile(r"[\w']+|[^\w ]")
    CAMEL_PAT = re.compile(r"(\b[A-Z]+[a-z]+[A-Z]\w+)")
    BR_PAT = re.compile(
        r"\s?\((.*)\)"
    )  # matches: whitespace, open bracket, anything, close bracket
    PREPS = {"from", "for", "of", "the", "in", "with", "to", "on", "and"}

    def train(self, repository: Repository, config: Dict[str, Any]) -> None:
        """Extracts dataset mentions and saves them to config["params"]

        Args:
            repository (Repository): Repository object
            config (Dict[str, Any]): Configuration dictionary

        Returns:
            None
        """

    def inference_string(self, config: Dict[str, Any], text: str) -> str:
        raise NotImplementedError()

    def inference_dataframe(
        self, config: Dict[str, Any], df: pd.DataFrame
    ) -> pd.DataFrame:
        raise NotImplementedError()

    @staticmethod
    def get_parenthesis(text: str, dataset: str) -> List[str]:
        # Get abbreviations in the brackets if there are any, ie dataset (abbr)
        cur_abbrs = re.findall(re.escape(dataset) + r"\s?(\([^\)]+\)|\[[^\]]+\])", text)
        # strip the parenthesis/brackets
        cur_abbrs = [abbr.strip("()[]").strip() for abbr in cur_abbrs]
        # this seems to remove an leftover brackets
        cur_abbrs = [re.split(r"[\(\[]", abbr)[0].strip() for abbr in cur_abbrs]
        # remove any ; or , at the end, assume the first part is the one we want
        cur_abbrs = [re.split("[;,]", abbr)[0].strip() for abbr in cur_abbrs]
        # also seems to be removing some leftover brackets
        cur_abbrs = [a for a in cur_abbrs if not any(ch in a for ch in "[]()")]
        # the acronym needs to consists of at least 2 captial letters
        cur_abbrs = [a for a in cur_abbrs if re.findall("[A-Z][A-Z]", a)]
        # this also removes any acronyms consisting of less than 3 letters
        cur_abbrs = [a for a in cur_abbrs if len(a) > 2]
        # removes any acronyms that are not all capital letters
        cur_abbrs = [
            a
            for a in cur_abbrs
            if not any(tok.islower() for tok in KaggleModel3.TOKENIZE_PAT.findall(a))
        ]

        # Remove an instances that captial letters followed by lower case letters
        # maybe this catches phrases that are within parenthesis
        fabbrs = list(
            filterfalse(
                lambda abbr: sum(
                    bool(re.findall("[A-Z][a-z]+", tok))
                    for tok in KaggleModel3.TOKENIZE_PAT.findall(abbr)
                )
                > 2,
                cur_abbrs,
            )
        )

        return fabbrs

    @staticmethod
    def _get_index(texts: List[str], words: List[str]) -> Dict[str, Set[int]]:
        # Returns a dictionary where words are keys and values are indices
        # of documents (sentences) in texts, in which the word present
        index = defaultdict(set)
        word_set = set(words)
        word_set = {
            w
            for w in word_set
            if w.lower() not in KaggleModel3.PREPS and re.sub("'", "", w).isalnum()
        }
        for n, text in enumerate(texts):
            tokens = KaggleModel3._tokenize(text)
            for tok in tokens:
                if tok in word_set:
                    index[tok].add(n)
        return index

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return KaggleModel3.TOKENIZE_PAT.findall(text)

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub("[^A-Za-z0-9]+", " ", str(text).lower()).strip()

    @staticmethod
    def _tokenized_extract(texts: List[str], keywords: List[str]) -> List[str]:
        # Exracts all mentions of the form
        # Xxx Xxx Keyword Xxx (XXX)
        connection_words = {"of", "the", "with", "for", "in", "to", "on", "and", "up"}
        datasets = []
        for text in texts:
            # ryanhausen: the code below was wrapped in try/except block, not sure why...
            # Skip texts without parenthesis or Xxx Xxx Keyword Xxx (XXX) keywords
            if "(" not in text or all(not kw in text for kw in keywords):
                continue

            toks = list(KaggleModel3.TOKENIZE_PAT.finditer(text))
            toksg = [tok.group() for tok in toks]

            # found = False # ryanhausen: this is set, but never used
            current_dss = set()
            for n in range(1, len(toks) - 2):
                is_camel = bool(KaggleModel3.CAMEL_PAT.findall(toksg[n + 1]))
                is_caps = toksg[n + 1].isupper()

                if toksg[n] == "(" and (is_caps or is_camel) and toksg[n + 2] == ")":
                    end = toks[n + 2].span()[1]
                    n_capi = 0
                    has_kw = False
                    start = 0
                    for tok, tokg in zip(toks[n - 1 :: -1], toksg[n - 1 :: -1]):
                        if tokg in keywords:
                            has_kw = True
                        if tokg[0].isupper() and tokg.lower() not in connection_words:
                            n_capi += 1
                            start = tok.span()[0]
                        elif tokg in connection_words or tokg == "-":
                            continue
                        else:
                            break
                    if n_capi > 1 and has_kw:
                        ds = text[start:end]
                        datasets.append(ds)
                        # found = True
                        current_dss.add(ds)

        return datasets


class MapFilter:
    """Base Class that applied a map and filter function to an iterable.

    The author of Model 3 originally had setup a series of steps that map
    and filter the input. This class is a way to encapsulate that
    """

    def __init__(
        self, map_f: Callable = lambda x: x, filter_f: Callable = lambda x: True
    ):
        self.map_f = map_f
        self.filter_f = filter_f

    def __call__(self, input: Iterable) -> Iterable:
        return map(self.map_f, filter(self.filter_f, input))


# ==============================================================================
# First stage map filters that can be applied batchwise and are independent of
# each other.
class MapFilter_AndThe(MapFilter):
    """Splits sentences on " and the " and returns the last part of the split."""

    def __init__(self):
        pat = re.compile(" and [Tt]he ")
        map_f: Callable[[str], str] = lambda ds: pat.split(ds)[-1]
        super().__init__(map_f=map_f)


class MapFilter_StopWords(MapFilter):
    """Filters out sentences that contain stopwords."""

    def __init__(self, stopwords: List[str], do_lower: bool = True):
        lower_f: Callable[[str], str] = lambda x: x.lower() if do_lower else x
        stopwords = list(map(lower_f, stopwords))

        def filter_f(ds: str) -> bool:
            ds_lower = lower_f(ds)
            return not any(sw in ds_lower for sw in stopwords)

        super().__init__(filter_f=filter_f)


class MapFilter_IntroSSAI(MapFilter):
    """I am not completely sure what this does, but it is used in Model 3.

    This rule seems hacky and should be refactored
    """

    def __init__(self, keywords: List[str], tokenize_pattern: re.Pattern):
        connection_words = {"of", "the", "with", "for", "in", "to", "on", "and", "up"}

        def map_f(ds: str) -> str:
            toks_spans = list(tokenize_pattern.finditer(ds))
            toks = [t.group() for t in toks_spans]

            start = 0
            if len(toks) > 3:
                if toks[1] == "the":
                    start = toks_spans[2].span()[0]
                elif (
                    toks[0] not in keywords
                    and toks[1] in connection_words
                    and len(toks) > 2
                    and toks[2] in connection_words
                ):
                    start = toks_spans[3].span()[0]
                elif toks[0].endswith("ing") and toks[1] in connection_words:
                    if toks[2] not in connection_words:
                        start_tok = 2
                    else:
                        start_tok = 3
                    start = toks_spans[start_tok].span()[0]
                return ds[start:]
            else:
                return ds

        super().__init__(map_f=map_f)


class MapFilter_IntroWords(MapFilter):
    """This replaces the first word and "the" or "to the" with '.'"""

    def __init__(self):
        miss_intro_pat = re.compile("^[A-Z][a-z']+ (?:the|to the) ")
        map_f: Callable[[str], str] = lambda ds: miss_intro_pat.sub("", ds)

        super().__init__(map_f)


class MapFilter_BRLessThanTwoWords(MapFilter):
    """This filters out strings that contain less thant two words, excluding phrases in parenthesis."""

    def __init__(self, br_pat: re.Pattern, tokenize_pat: re.Pattern):

        filter_f: Callable[[str], bool] = (
            lambda ds: len(tokenize_pat.findall(br_pat.sub("", ds))) > 2
        )

        super().__init__(filter_f=filter_f)


# ==============================================================================

# ==============================================================================
# Second stage map filters that can be applied batchwise and are dependent on
# on statistics from the first stage.
class MapFilter_PartialMatchDatasets(MapFilter):
    """This filters out subsets of strings that are between parenthesis and exist in the dataset."""

    def __init__(self, dataset: List[str], br_pat: re.Pattern):

        counter = Counter(dataset)
        abbrs_used = set()
        golden_ds_with_br = []

        for ds, _ in counter.most_common():
            abbr = br_pat.findall(ds)[0]

            if abbr not in abbrs_used:
                abbrs_used.add(abbr)
                golden_ds_with_br.append(ds)

        filter_f = lambda ds: not any(
            (ds in ds_) and (ds != ds_) for ds_ in golden_ds_with_br
        )

        super().__init__(filter_f=filter_f)


if __name__ == "__main__":

    input = (
        "This model was trained on the Really Great Dataset (RGD)"
        + " and the Really Bad Dataset (RBD) and it went well"
        + " (though not that well)."
    )
    dataset = "Really Great Dataset"
    KaggleModel3.get_parenthesis(text=input, dataset=dataset)
    assert KaggleModel3.get_parenthesis(text=input, dataset=dataset) == ["RGD"]
    assert KaggleModel3._tokenized_extract([input], KaggleModel3.KEYWORDS) == [
        "Really Great Dataset (RGD)",
        "Really Bad Dataset (RBD)",
    ]

    assert list(MapFilter_AndThe()([input])) == [
        "Really Bad Dataset (RBD) and it went well (though not that well)."
    ]
    assert list(MapFilter_StopWords(stopwords=["really", "bad"])([input])) == []
    assert list(MapFilter_StopWords(stopwords=["stopword"])([input])) == [input]

    assert list(
        MapFilter_IntroSSAI(
            keywords=KaggleModel3.KEYWORDS, tokenize_pattern=KaggleModel3.TOKENIZE_PAT
        )(["Really Great Dataset (RGD)", "Really Bad Dataset (RBD)"])
    ) == ["Really Great Dataset (RGD)", "Really Bad Dataset (RBD)"]

    assert list(MapFilter_IntroWords()(["Really to the data is great"])) == [
        "data is great"
    ]
    assert list(
        MapFilter_BRLessThanTwoWords(
            br_pat=KaggleModel3.BR_PAT, tokenize_pat=KaggleModel3.TOKENIZE_PAT
        )([input, "Hello World (HW)"])
    ) == [input]

    assert list(
        MapFilter_PartialMatchDatasets(dataset=[input], br_pat=KaggleModel3.BR_PAT)(
            [input]
        )
    ) == [input]
