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
import json
import logging
import os
import re
from collections import Counter, defaultdict
from itertools import chain, filterfalse
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import pandas as pd
from tqdm import tqdm

from src.data.repository import Repository
import src.models.base_model as bm
import src.evaluate.model as em

logger = logging.getLogger("kaggle_model3")


def validate_config(config: Dict[str, Any]) -> None:

    expected_keys = [
        "keywords",
        "min_train_count",
        "rel_freq_threshold",
        "model_path",
        "eval_path",
    ]

    for key in expected_keys:
        assert key in config, f"Missing key {key} in config"


def train(
    repository: Repository,
    config: Dict[str, Any],
    training_logger: Optional[bm.SupportsLogging] = None,
) -> None:
    """Trains the model and saves the results to config.model_path

    Args:
        repository (Repository): Repository object
        config (Dict[str, Any]): Configuration dictionary
        training_logger (SupportsLogging, optional): Training logger

    Returns:
        None
    """
    validate_config(config)

    model = KaggleModel3()
    model.train(repository, config)


def validate(repository: Repository, config: Dict[str, Any]) -> None:
    """Validates the model and saves the results to config.model_path

    Args:
        repository (Repository): Repository object
        config (Dict[str, Any]): Configuration dictionary

    Returns:
        None
    """
    validate_config(config)

    model = KaggleModel3()
    model_evaluation = em.evaluate_model(repository, model, config)

    print(model_evaluation)

    logger.info(f"Saving evaluation to {config['eval_path']}")
    with open(config["eval_path"], "w") as f:
        json.dump(model_evaluation.to_json(), f)


class KaggleModel3(bm.Model):
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
        """Extracts dataset mentions and saves them to config.model_path

        Args:
            repository (Repository): Repository object
            config (Dict[str, Any]): Configuration dictionary

        Returns:
            None
        """

        sentencizer = DotSplitSentencizer(True)
        # TODO: this currently only works with the KaggleRepository and pulls
        # the whole training set into memory. This should be changed to work
        # with on batches.
        df = next(repository.get_training_data_dataframe(10000000000))

        samples = {}
        for _, (
            idx,
            pub_title,
            dataset_title,
            dataset_label,
            cleaned_label,
            json_text,
        ) in df.iterrows():
            if idx not in samples:
                samples[idx] = {
                    "texts": [sec["text"] for sec in json_text],
                    "dataset_titles": [],
                    "dataset_labels": [],
                    "cleaned_labels": [],
                    "pub_title": pub_title,
                    "idx": idx,
                }
            samples[idx]["dataset_titles"].append(dataset_title)
            samples[idx]["dataset_labels"].append(dataset_label)
            samples[idx]["cleaned_labels"].append(cleaned_label)

        train_ids = []
        train_texts = []
        train_labels = []
        for sample_dict in samples.values():
            train_ids.append(sample_dict["idx"])
            texts = sample_dict["texts"]
            if sentencizer is not None:
                texts = list(chain(*[sentencizer(text) for text in texts]))
            train_texts.append(texts)
            train_labels.append(sample_dict["dataset_labels"])

        texts = list(chain(*train_texts))

        ssai_par_datasets = KaggleModel3._tokenized_extract(
            texts, KaggleModel3.KEYWORDS
        )
        words = list(chain(*[KaggleModel3._tokenize(ds) for ds in ssai_par_datasets]))

        mapfilters = [
            MapFilter_AndThe(),
            MapFilter_StopWords(KaggleModel3.STOPWORDS_PAR),
            MapFilter_IntroSSAI(KaggleModel3.KEYWORDS, KaggleModel3.TOKENIZE_PAT),
            MapFilter_IntroWords(),
            MapFilter_BRLessThanTwoWords(
                KaggleModel3.BR_PAT, KaggleModel3.TOKENIZE_PAT
            ),
        ]

        for mapfilter in mapfilters:
            ssai_par_datasets = mapfilter(ssai_par_datasets)

        mapfilters = [
            MapFilter_PartialMatchDatasets(ssai_par_datasets, KaggleModel3.BR_PAT),
            MapFilter_TrainCounts(
                texts,
                ssai_par_datasets,
                KaggleModel3._get_index(texts, set(words)),
                config["keywords"],
                config["min_train_count"],
                config["rel_freq_threshold"],
                KaggleModel3.TOKENIZE_PAT,
            ),
            MapFilter_BRPatSub(KaggleModel3.BR_PAT),
        ]

        for mapfilter in mapfilters:
            ssai_par_datasets = mapfilter(ssai_par_datasets)

        train_labels_set = set(chain(*train_labels))
        # This line is in the original notebook, but doesn't seem to do anything
        train_datasets = [
            KaggleModel3.BR_PAT.sub("", ds).strip() for ds in train_labels_set
        ]
        train_datasets = [
            ds for ds in train_labels_set if sum(ch.islower() for ch in ds) > 0
        ]
        datasets = set(ssai_par_datasets) | set(train_datasets)

        logger.info(f"Saving {len(datasets)} datasets to {config['model_path']}")
        os.makedirs(os.path.dirname(config["model_path"]), exist_ok=True)
        with open(config["model_path"], "w") as f:
            f.write("\n".join(datasets))

    def inference(self, config: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
        """Inference method for the KaggleModel3

        Args:
            config (Dict[str, Any]): Configuration dictionary
            df (pd.DataFrame): Dataframe containing the texts to be classified
                               should have columns "Id", "text"


        Returns:
            pd.DataFrame: Dataframe with an additional column "model_prediction"
                          containing the inferred datasets
        """

        with open(config["model_path"]) as f:
            datasets = set([l.strip() for l in f.readlines()])

        def infer_sample(text: str) -> str:
            predictions = []
            for sent in re.split(r"[\.]", text):
                for ds in datasets:
                    if (ds in sent) and (ds not in predictions):
                        predictions.append(ds)
                        predictions.extend(KaggleModel3.get_parenthesis(sent, ds))
            return "|".join(predictions)

        df["model_prediction"] = df["text"].apply(infer_sample)

        return df

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
    def _get_index(texts: List[str], words: Set[str]) -> Dict[str, Set[int]]:
        # Returns a dictionary where words are keys and values are indices
        # of documents (sentences) in texts, in which the word present
        index = defaultdict(set)
        words = {
            w
            for w in words
            if w.lower() not in KaggleModel3.PREPS and re.sub("'", "", w).isalnum()
        }
        for n, text in tqdm(enumerate(texts), total=len(texts), desc="Indexing"):
            tokens = KaggleModel3._tokenize(text)
            for tok in tokens:
                if tok in words:
                    index[tok].add(n)
        return index

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return KaggleModel3.TOKENIZE_PAT.findall(text)

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub("[^A-Za-z0-9]+", " ", str(text).lower()).strip()

    @staticmethod
    def _tokenized_extract(texts: List[str], keywords: List[str]) -> Iterable[str]:
        # Exracts all mentions of the form
        # Xxx Xxx Keyword Xxx (XXX)
        connection_words = {"of", "the", "with", "for", "in", "to", "on", "and", "up"}
        datasets = []
        for text in tqdm(texts, total=len(texts), desc="Tokenizing"):
            try:
                # Skip texts without parenthesis orXxx Xxx Keyword Xxx (XXX) keywords
                if "(" not in text or all(not kw in text for kw in keywords):
                    continue

                toks = list(KaggleModel3.TOKENIZE_PAT.finditer(text))
                toksg = [tok.group() for tok in toks]

                found = False
                current_dss = set()
                for n in range(1, len(toks) - 2):
                    is_camel = bool(KaggleModel3.CAMEL_PAT.findall(toksg[n + 1]))
                    is_caps = toksg[n + 1].isupper()

                    if (
                        toksg[n] == "("
                        and (is_caps or is_camel)
                        and toksg[n + 2] == ")"
                    ):
                        end = toks[n + 2].span()[1]
                        n_capi = 0
                        has_kw = False
                        for tok, tokg in zip(toks[n - 1 :: -1], toksg[n - 1 :: -1]):
                            if tokg in keywords:
                                has_kw = True
                            if (
                                tokg[0].isupper()
                                and tokg.lower() not in connection_words
                            ):
                                n_capi += 1
                                start = tok.span()[0]
                            elif tokg in connection_words or tokg == "-":
                                continue
                            else:
                                break
                        if n_capi > 1 and has_kw:
                            ds = text[start:end]
                            datasets.append(ds)
                            found = True
                            current_dss.add(ds)
            except:
                print(text)

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

    def __call__(self, input: Iterable[str]) -> Iterable[str]:
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

    def __repr__(self) -> str:
        return "MapFilter_AndThe"


class MapFilter_StopWords(MapFilter):
    """Filters out sentences that contain stopwords."""

    def __init__(self, stopwords: List[str], do_lower: bool = True):
        lower_f: Callable[[str], str] = lambda x: x.lower() if do_lower else x
        stopwords = list(map(lower_f, stopwords))

        def filter_f(ds: str) -> bool:
            ds_lower = lower_f(ds)
            return not any(sw in ds_lower for sw in stopwords)

        super().__init__(filter_f=filter_f)

    def __repr__(self) -> str:
        return "MapFilter_StopWords"


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

    def __repr__(self) -> str:
        return "MapFilter_IntroSSAI"


class MapFilter_IntroWords(MapFilter):
    """This replaces the first word and "the" or "to the" with '.'"""

    def __init__(self):
        miss_intro_pat = re.compile("^[A-Z][a-z']+ (?:the|to the) ")
        map_f: Callable[[str], str] = lambda ds: miss_intro_pat.sub("", ds)

        super().__init__(map_f)

    def __repr__(self) -> str:
        return "MapFilter_IntroWords"


class MapFilter_BRLessThanTwoWords(MapFilter):
    """This filters out strings that contain less than two words, excluding phrases in parenthesis."""

    def __init__(self, br_pat: re.Pattern, tokenize_pat: re.Pattern):

        filter_f: Callable[[str], bool] = (
            lambda ds: len(tokenize_pat.findall(br_pat.sub("", ds))) > 2
        )

        super().__init__(filter_f=filter_f)

    def __repr__(self) -> str:
        return "MapFilter_BRLessThanTwoWords"


# ==============================================================================

# ==============================================================================
# Second stage map filters that can be applied batchwise and applied after
# aggregate data is collected over the whole dataset.
class MapFilter_PartialMatchDatasets(MapFilter):
    """This filters out subsets of strings that are between parenthesis and exist in the dataset."""

    def __init__(
        self,
        dataset: Iterable[str],
        br_pat: re.Pattern,
        n_most_common: Optional[int] = None,
    ):

        counter = Counter(dataset)
        abbrs_used = set()
        golden_ds_with_br = []

        # REFACTOR: will have to save to disk/restore/merge counts
        for ds, _ in counter.most_common(n=n_most_common):
            abbr = br_pat.findall(ds)[0]

            if abbr not in abbrs_used:
                abbrs_used.add(abbr)
                golden_ds_with_br.append(ds)

        filter_f: Callable[[str], bool] = lambda ds: not any(
            (ds in ds_) and (ds != ds_) for ds_ in golden_ds_with_br
        )

        super().__init__(filter_f=filter_f)

    def __repr__(self) -> str:
        return "MapFilter_PartialMatchDatasets"


# This class might need to be refactored to consider taking a second pass
# over the dataset to collect more information.
class MapFilter_TrainCounts(MapFilter):
    def __init__(
        self,
        texts,  # REFACTOR: this is ALL the texts
        datasets,  # this is the curried set of datasets
        index,
        kw,  # this is a selected keyword orginally "data"
        min_train_count,  # min occurences in dataset defualt 2
        rel_freq_threshold,  # default 0.1
        tokenize_pat,  # tokenization pattern
    ):
        # Filter by relative frequency (no parenthesis)
        # (check the formula in the first cell)
        (
            tr_counts,
            data_counts,
        ) = MapFilter_TrainCounts.get_train_predictions_counts_data(
            texts,
            MapFilter_TrainCounts.extend_paranthesis(set(datasets)),
            index,
            kw,
            tokenize_pat,
        )
        stats = {}

        for ds, count in Counter(datasets).most_common():
            stats[ds] = [
                count,
                tr_counts[ds],
                tr_counts[re.sub(r"[\s]?\(.*\)", "", ds)],
                data_counts[ds],
                data_counts[re.sub(r"[\s]?\(.*\)", "", ds)],
            ]

        def filter_f(ds):
            count, tr_count, tr_count_no_br, dcount, dcount_nobr = stats[ds]
            return (tr_count_no_br > min_train_count) and (
                dcount_nobr / tr_count_no_br > rel_freq_threshold
            )

        super().__init__(filter_f=filter_f)

    @staticmethod
    def extend_paranthesis(datasets: List[str]) -> List[str]:
        # Return each instance of dataset from datasets +
        # the same instance without parenthesis (if there are some)
        pat = re.compile(r"\(.*\)")
        extended_datasets = []
        for ds in datasets:
            ds_no_parenth = pat.sub("", ds).strip()
            if ds != ds_no_parenth:
                extended_datasets.append(ds_no_parenth)
            extended_datasets.append(ds)
        return extended_datasets

    @staticmethod
    def get_train_predictions_counts_data(
        texts: List[str],
        datasets: List[str],
        index: Dict[str, Set[int]],
        kw: Union[str, List[str]],
        tokenize_pat: re.Pattern,
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        # Original author notes:
        # Returns N_data and N_total counts dictionary
        # (check the formulas in the first cell)
        pred_count: Dict[str, int] = Counter()
        data_count: Dict[str, int] = Counter()
        if isinstance(kw, str):
            kw = [kw]

        for ds in tqdm(datasets):
            first_tok, *toks = tokenize_pat.findall(ds)
            to_search: Set[int] = set()
            for tok in [first_tok] + toks:
                if index.get(tok):
                    if len(to_search) == 0:
                        to_search = set(index[tok])
                    else:
                        to_search &= index[tok]
                else:
                    pass
            for doc_idx in to_search:
                text = texts[doc_idx]
                if ds in text:
                    pred_count[ds] += 1
                    data_count[ds] += int(any(w in text.lower() for w in kw))
        return pred_count, data_count

    @staticmethod
    def get_index(texts: List[str], input_words: List[str]) -> Dict[str, Set[int]]:
        """Returns a Dictionary that maps input_words to the indices of the documents they apeared in."""

        # Original author notes:
        # Returns a dictionary where words are keys and values are indices
        # of documents (sentences) in texts, in which the word present
        index = defaultdict(set)
        # words = set(words)
        # words = {
        #     w
        #     for w in words
        #     if w.lower() not in KaggleModel3.PREPS and re.sub("'", "", w).isalnum()
        # }
        words = set(
            filter(
                lambda w: w.lower() not in KaggleModel3.PREPS
                and re.sub("'", "", w).isalnum(),
                input_words,
            )
        )

        for n, text in enumerate(texts):
            # this tokenizes entire documents
            tokens = KaggleModel3._tokenize(text)
            for tok in tokens:
                if tok in words:
                    index[tok].add(n)
        return index

    def __repr__(self) -> str:
        return f"MapFilter_TrainCounts"


class MapFilter_BRPatSub(MapFilter):
    """This removes the strings between parenthesis."""

    def __init__(self, br_pat):

        map_f: Callable[[str], str] = lambda ds: br_pat.sub("", ds)

        super().__init__(map_f=map_f)

    def __repr__(self) -> str:
        return f"MapFilter_BRPatSub"


class Sentencizer:
    def __init__(self, sentencize_fun: Callable, split_by_newline: bool = True) -> None:
        self.sentencize = sentencize_fun
        self.split_by_newline = split_by_newline

    def __call__(self, text: str) -> List[str]:
        if self.split_by_newline:
            texts = text.split("\n")
        else:
            texts = [text]
        sents = []
        for text in texts:
            sents.extend(self.sentencize(text))
        return sents


class DotSplitSentencizer(Sentencizer):
    def __init__(self, split_by_newline: bool) -> None:
        def _sent_fun(text: str) -> List[str]:
            return [sent.strip() for sent in text.split(".") if sent]

        super().__init__(_sent_fun, split_by_newline)


if __name__ == "__main__":
    bm.train = train
    bm.validate = validate
    bm.main()


# if __name__ == "__main__":

#     input = (
#         "This model was trained on the Really Great Dataset (RGD)"
#         + " and the Really Bad Dataset (RBD) and it went well"
#         + " (though not that well)."
#     )
#     dataset = "Really Great Dataset"
#     KaggleModel3.get_parenthesis(text=input, dataset=dataset)
#     assert KaggleModel3.get_parenthesis(text=input, dataset=dataset) == ["RGD"]
#     assert KaggleModel3._tokenized_extract([input], KaggleModel3.KEYWORDS) == [
#         "Really Great Dataset (RGD)",
#         "Really Bad Dataset (RBD)",
#     ]

#     assert list(MapFilter_AndThe()([input])) == [
#         "Really Bad Dataset (RBD) and it went well (though not that well)."
#     ]
#     assert list(MapFilter_StopWords(stopwords=["really", "bad"])([input])) == []
#     assert list(MapFilter_StopWords(stopwords=["stopword"])([input])) == [input]

#     assert list(
#         MapFilter_IntroSSAI(
#             keywords=KaggleModel3.KEYWORDS, tokenize_pattern=KaggleModel3.TOKENIZE_PAT
#         )(["Really Great Dataset (RGD)", "Really Bad Dataset (RBD)"])
#     ) == ["Really Great Dataset (RGD)", "Really Bad Dataset (RBD)"]

#     assert list(MapFilter_IntroWords()(["Really to the data is great"])) == [
#         "data is great"
#     ]
#     assert list(
#         MapFilter_BRLessThanTwoWords(
#             br_pat=KaggleModel3.BR_PAT, tokenize_pat=KaggleModel3.TOKENIZE_PAT
#         )([input, "Hello World (HW)"])
#     ) == [input]

#     assert list(
#         MapFilter_PartialMatchDatasets(dataset=[input], br_pat=KaggleModel3.BR_PAT)(
#             [input]
#         )
#     ) == [input]

#     assert MapFilter_TrainCounts.get_index(
#         [input, input[: input.index("Dataset")]], ["Really", "Great", "Dataset"]
#     ) == {"Really": {0, 1}, "Great": {0, 1}, "Dataset": {0}}

#     assert MapFilter_TrainCounts.extend_paranthesis(["Really Great Dataset (RGD)"]) == [
#         "Really Great Dataset",
#         "Really Great Dataset (RGD)",
#     ]

#     assert MapFilter_TrainCounts.get_train_predictions_counts_data(
#         texts=[input, input[: input.index("Dataset")]],
#         datasets=["Really Great Dataset (RGD)", "Really Bad Dataset (RBD)"],
#         index=MapFilter_TrainCounts.get_index(
#             [input, input[: input.index("Dataset")]],
#             ["Really", "Great", "Dataset", "Really", "Bad", "Dataset"],
#         ),
#         kw="data",
#         tokenize_pat=KaggleModel3.TOKENIZE_PAT,
#     ) == (
#         Counter({"Really Great Dataset (RGD)": 1, "Really Bad Dataset (RBD)": 1}),
#         Counter({"Really Great Dataset (RGD)": 1, "Really Bad Dataset (RBD)": 1}),
#     )

#     config = Model3Hyperparameters(
#         model_path="models/kagglemodel3/baseline/params.txt",
#         keywords=["data"],
#         min_train_count=2,
#         rel_freq_threshold=0.1,
#         batch_size=-1,
#     )

#     logging.basicConfig(level=logging.INFO)
#     KaggleModel3().train(KaggleRepository(), config)
