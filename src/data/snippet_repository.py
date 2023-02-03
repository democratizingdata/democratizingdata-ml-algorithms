# Based on the dataset used by kaggle model 1
# https://github.com/Coleridge-Initiative/rc-kaggle-models/blob/main/1st%20ZALO%20FTW/notebooks/get_candidate_labels.ipynb

from enum import Enum
from functools import partial
from itertools import starmap
import itertools
import json
import logging
import os
from unidecode import unidecode
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import regex as re
import spacy
from imblearn.over_sampling import RandomOverSampler
from pandarallel import pandarallel
from sklearn.model_selection import train_test_split
from spacy import displacy
from spacy.tokens import Doc
from tqdm import tqdm

from src.data.repository import Repository
from src.models.regex_model import RegexModel

import warnings

warnings.filterwarnings("ignore")  # setting ignore as a parameter

logger = logging.getLogger("snippet_repository")


def get_text_per_row(path_dir: str, row: pd.DataFrame) -> str:
    doc_id: str = row["id"]
    with open(os.path.join(path_dir, doc_id + ".json"), "r") as f:
        text = unidecode(
            " ".join(
                [
                    sec["text"].replace("\t", " ").replace("\n", " ")
                    for sec in json.load(f)
                ]
            )
        )
    return text


def extract_extra_candidates(
    model: RegexModel, keywords: List[str], row: pd.DataFrame
) -> str:
    text = row["text"]
    all_entities = model.inference({}, pd.DataFrame({"text": [text]}))[
        "model_prediction"
    ]
    stripped_and_split = [v.strip() for v in all_entities.values[0].split("|")]

    contains_keyword = list(
        filter(
            lambda x: any(map(lambda y: y.lower() in x.lower(), keywords)),
            stripped_and_split,
        )
    )

    return "|".join(contains_keyword)


def tag_sentence(
    nlp: spacy.language.Language,
    regex_labels: List[re.Pattern],
    sentence: spacy.tokens.span.Span,
) -> Tuple[List[str], List[str], List[str]]:

    # find matches for each label and sort by length, longest first
    # shorter matches might be a subset of a longer match. So, we'll
    # prefer the longer matches first.
    match_lists = sorted(
        SnippetRepository.detect_labels(regex_labels, sentence.text),
        key=lambda x: max(map(len, x)),
        reverse=True,
    )

    tokens = [token.text for token in sentence]
    tags = [token.tag_ for token in sentence]
    ner_tags = ["O"] * len(sentence)  # assume no match

    for matches in match_lists:
        for match in matches:
            try:
                label_tokens = nlp(match)
                if len(label_tokens) == 0:
                    raise ValueError(label_tokens)
                start_idx = tokens.index(label_tokens[0].text)
                idxs = list(range(start_idx, start_idx + len(label_tokens)))
            except ValueError:
                # print(f"Could not find {str(match)} in sentence: ",  sentence.text)
                continue

            first_tag = ner_tags[start_idx]
            prev_tag = ner_tags[start_idx - 1] if start_idx > 0 else "O"
            # If there are any tokens that are already marked then this match
            # could be a subset of another match
            if not any(
                map(
                    lambda x: x != "O",
                    ner_tags[start_idx : start_idx + len(label_tokens)],
                )
            ):
                if prev_tag == "O":
                    ner_tags[start_idx] = "I-DAT"
                else:
                    ner_tags[start_idx] = "B-DAT"

                for idx in idxs[1:]:
                    ner_tags[idx] = "I-DAT"

    return tokens, tags, ner_tags


def convert_document_to_samples(nlp: spacy.language.Language, row: pd.DataFrame):
    doc = row["tokenized"]
    samples = []
    labels = list(
        map(re.compile, map(RegexModel.regexify_keyword, row["label"].split("|")))
    )

    candidate_labels = list(
        map(
            re.compile, map(RegexModel.regexify_keyword, row["extra_labels"].split("|"))
        )
    )
    for sentence in filter(lambda s: len(s)>5, doc.sents):
        tokens, tags, label_ner_tags = tag_sentence(nlp, labels, sentence)

        contains_label = any(map(lambda x: x != "O", label_ner_tags))

        if not contains_label:
            _, _, candidate_ner_tags = tag_sentence(nlp, candidate_labels, sentence)
        else:
            candidate_ner_tags = ["O"] * len(sentence)

        contains_candidate = any(map(lambda x: x != "O", candidate_ner_tags))
        # if there are only tags form the labels, then
        # we'll use it training as a positive sample.
        # if there are tags from the candidate labels,
        # then we'll use it in the validation set as a
        # positive sample.
        # otherwise, it will be a negative sample
        ner_tags = label_ner_tags if contains_label else candidate_ner_tags

        # tagged_formatted_tokens = "".join(starmap(
        #     lambda token, tag, ner_tag: f"{token}\t{tag}\t{ner_tag}\n",
        #     zip(tokens, tags, ner_tags)
        # ))
        clean_tokens, clean_tags, clean_ner_tags = list(zip(*filter(
            lambda x: x[0] != " ",
            zip(tokens, tags, ner_tags)
        )))

        tagged_formatted_tokens = (
            "\t".join(
                [
                    str(int(contains_label or contains_candidate)),
                    " ".join(clean_tokens),
                    " ".join(clean_tags),
                    " ".join(clean_ner_tags),
                ]
            )
            + "\n"
        )

        samples.append(
            {
                "sample": tagged_formatted_tokens,
                "is_validation": contains_candidate,
            }
        )

    if len(samples) == 0:
        print("No samples found for document: ", row["id"])

    return samples


def save_samples(
    train_location: str, validation_location: str, row: pd.DataFrame
) -> None:
    save_train_path = os.path.join(train_location, row["id"] + ".tsv")

    save_validation_path = os.path.join(validation_location, row["id"] + ".tsv")

    try:
        for sample in row["tokenized_samples"]:
            save_path = (
                save_validation_path if sample["is_validation"] else save_train_path
            )
            with open(save_path, "a") as f:
                f.write(sample["sample"])
    except:
        print("Row could not be saved", row)


def get_snippet_labels(root_path: str, row: pd.DataFrame) -> np.ndarray:
    snippet_path = os.path.join(root_path, row["id"] + ".tsv")
    if os.path.exists(snippet_path):
        with open(snippet_path, "r") as f:
            lines = f.readlines()
        labels = list(map(lambda x: int(x.split("\t")[0]), lines))
    else:
        labels = []

    return np.array(labels)

def get_sample_snippet(path:str, id:str, index:int)-> Tuple[str, str, str]:
    with open(os.path.join(path, id + ".tsv"), "r") as f:
        lines = f.readlines()
    return lines[index].strip().split("\t")[1:]

def snippet_to_classification_sample(path:str , row:pd.DataFrame) -> pd.DataFrame:
    tokens, _, _ = get_sample_snippet(path, row["id"], row["snippet_index"])

    return pd.DataFrame({
        "text": [tokens],
        "label": [row["snippet_label"]]
    })

def snippet_to_masked_lm_sample(path:str, row:pd.DataFrame) -> pd.DataFrame:
    tokens, _, ner_tags = get_sample_snippet(path, row["id"], row["snippet_index"])

    return pd.DataFrame({
        "text": [tokens.split()],
        "mask": [list(map(lambda x: x != "O", ner_tags.split()))],
        "label": [row["snippet_label"]]
    })

def snippet_to_ner_sample(path:str, row:pd.DataFrame) -> pd.DataFrame:
    tokens, tags, ner_tags = get_sample_snippet(path, row["id"], row["snippet_index"])

    return pd.DataFrame({
        "text": [tokens.split()],
        "tags": [tags.split()],
        "ner_tags": [ner_tags.split()]
    })


def extract_entities(text:List[str], ner_tags:List[str]) -> List[Dict[str, Union[int,str]]]:
    """Extracts entities from a list of tokens and their corresponding NER tags.
    Args:
        text (List[str]): The list of tokens.
        ner_tags (List[str]): The list of NER tags.
        Returns:
            List[Dict[str, Any]]: A list of entities, each entity is a dictionary
                                  with the following keys:
                                    - start: The start index of the entity.
                                    - end: The end index of the entity.
                                    - label: The label of the entity.
    """
    entities = []
    start = 0
    end = 0
    in_ent = False
    for token, tag in zip(text, ner_tags):
        token_length = len(token) + 1 # plus a space
        if tag == "I-DAT":
            in_ent = True
            end += token_length
        elif tag == "O" and in_ent:
            entities.append({
                "start": start,
                "end": end,
                "label": "Dataset"
            })
            in_ent = False
            start = end
        elif tag == "B-DAT":
            entities.append({
                "start": start,
                "end": end,
                "label": "Dataset"
            })
            start = end
            end += token_length
        else:
            # just an O tag
            start += token_length
            end = start

    return entities

def visualize_ner_tags(
    text: str, ner_tags: List[str]
) -> Tuple[str, str, str]:
    """Visualize NER tags in a text."""
    entities = extract_entities(text, ner_tags)

    ex = [{
        "text": " ".join(text),
        "ents": entities,
        "title": None
    }]

    displacy.render(
        ex,
        style="ent",
        manual=True,
        colors={"Dataset": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
    )

class SnippetRepositoryMode(Enum):
    NER = "ner"
    CLASSIFICATION = "classification"
    MASKED_LM = "masked_lm"


class SnippetRepository(Repository):
    """Repository for serving training snippets."""

    def __init__(
        self,
        mode: SnippetRepositoryMode,
        build_options: Optional[Dict[str, Any]] = dict(),
    ) -> None:
        self.mode = mode
        self.local = os.path.dirname(__file__)
        with_local_path = partial(os.path.join, self.local)

        self.train_labels_location = with_local_path("../../data/kaggle/train.csv")

        self.train_dataframe = with_local_path("../../data/snippets/snippets_train.csv")
        self.test_dataframe = with_local_path("../../data/snippets/snippets_test.csv")
        self.validation_dataframe = with_local_path(
            "../../data/snippets/snippets_validation.csv"
        )
        self.train_balanced_dataframe = with_local_path(
            "../../data/snippets/snippets_train_balanced.csv"
        )

        self.all_ids = pd.read_csv(self.train_labels_location)["Id"].values
        self.train_ids, self.test_ids = train_test_split(
            self.all_ids, test_size=0.2, random_state=42
        )

        self.train_files_location = with_local_path(
            "../../data/snippets/kaggle_snippets_train"
        )
        self.validation_files_location = with_local_path(
            "../../data/snippets/kaggle_snippets_validation"
        )

        make_dir_f = partial(os.makedirs, exist_ok=True)
        list(
            map(make_dir_f, [self.train_files_location, self.validation_files_location])
        )

        if (
            len(os.listdir(self.train_files_location)) == 0
            or len(os.listdir(self.validation_files_location)) == 0
            or len(build_options)
        ):
            self.nlp = spacy.load("en_core_web_sm")
            self.build(build_options)

    def transform_df(self, is_validation:bool, df: pd.DataFrame) -> pd.DataFrame:
        # this dataframe has the columns id, snippet_label, snippet_index
        # we need to retrieve the snippet and transform it depending on the
        # selected mode
        path = self.validation_files_location if is_validation else self.train_files_location
        if self.mode == SnippetRepositoryMode.CLASSIFICATION:
            # If mode is classification, we need to transform the rows to return
            # the snippet and the label
            return df.apply(
                lambda row: snippet_to_classification_sample(path, row), axis=1
            )
        elif self.mode == SnippetRepositoryMode.NER:
            # If mode is NER, we need to transform the rows to return the snippet
            # as a token list and the label as a list of NER tags
            return df.apply(
                lambda row: snippet_to_ner_sample(path, row), axis=1
            )
        elif self.mode == SnippetRepositoryMode.MASKED_LM:
            # If mode is MASKED_LM, we need to transform the rows to return the
            # snippet as a token list and the label as a list of "mask" tokens
            return df.apply(
                lambda row: snippet_to_masked_lm_sample(path, row), axis=1
            )

        return df

    def get_iter_or_df(
        self,
        path:str,
        transform_f:Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x,
        batch_size: Optional[int] = None,
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        def iter_f():
            for batch in pd.read_csv(path, chunksize=batch_size):
                yield batch

        if batch_size:
            return map(transform_f, iter_f())
        else:
            df = pd.read_csv(path)
            return transform_f(df)


    def get_training_data(
        self, batch_size: Optional[int] = None, balance_labels: bool = False
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        path = (
            self.train_balanced_dataframe
            if balance_labels
            else self.train_dataframe
        )

        transform_f = partial(self.transform_df, False)
        aggregate_f = lambda x: pd.concat(x.values, ignore_index=True)
        transform_aggregate_f = lambda x: aggregate_f(transform_f(x))
        return self.get_iter_or_df(path, transform_aggregate_f,  batch_size)


    def get_test_data(
        self, batch_size: Optional[int] = None
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        return self.get_iter_or_df(self.test_dataframe, partial(self.transform_df, False), batch_size)

    def get_validation_data(
        self, batch_size: Optional[int] = None
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        return self.get_iter_or_df(self.validation_dataframe, partial(self.transform_df, True), batch_size)

    @staticmethod
    def detect_labels(labels: List[re.Pattern], sentence: str) -> List[List[str]]:
        return list(
            map(
                lambda match: match.captures(),  # It's possible to have more than one match
                filter(bool, map(lambda rl: rl.search(sentence), labels)),
            )
        )

    def build(self, build_options: Dict[str, Any]) -> None:
        # get training data from kaggle
        process_n = build_options.get("process_n", 100)

        print("Loading Kaggle training labels...")
        all_kaggle_train = pd.read_csv(self.train_labels_location)#.iloc[:300, :]
        split_training_data = np.array_split(
            all_kaggle_train, len(all_kaggle_train) // process_n
        )

        for kaggle_train in tqdm(split_training_data, desc="Total Progress"):

            def aggregate_clean_label(row: pd.DataFrame):
                labels = list(
                    map(lambda x: x.lower().strip(), row["dataset_label"].unique())
                )
                return "|".join(labels)

            existing_files = list(
                map(lambda x: x.split(".")[0], os.listdir(self.train_files_location))
            )

            existing_files.extend(
                list(
                    map(
                        lambda x: x.split(".")[0],
                        os.listdir(self.validation_files_location),
                    )
                )
            )

            model = RegexModel(config=dict())
            extract_extra_candidates_f = partial(
                extract_extra_candidates, model, build_options["keywords"]
            )

            unique_labels = kaggle_train.groupby("Id").apply(aggregate_clean_label)

            ids_to_work_on = list(
                filter(lambda x: x not in existing_files, kaggle_train["Id"].unique())
            )

            all_df = pd.DataFrame({"id": ids_to_work_on})
            all_df["label"] = all_df["id"].apply(lambda x: unique_labels[x])

            print("Getting text files")
            get_text_f = partial(
                get_text_per_row, os.path.join(self.local, "../../data/kaggle/train")
            )
            pandarallel.initialize(progress_bar=True)
            all_df["text"] = all_df.parallel_apply(get_text_f, axis=1)

            # export OPENBLAS_NUM_THREADS=1, sometimes this helps with parallelization
            # for some reason spacy's large model doesn't work with parallelization
            # so for now initialize with the small model
            print("\nRunning spaCy tokenizer/tagger")
            all_texts = all_df["text"].tolist()
            all_ids = all_df["id"].tolist()
            expanded_texts = []
            expanded_ids = []
            max_tokens = 10_000
            for text, id in zip(all_texts, all_ids):
                tokens = text.split()

                if len(tokens) > max_tokens:
                    token_sequences = [
                        " ".join(tokens[i : i + max_tokens])
                        for i in range(0, len(tokens), max_tokens)
                    ]
                    expanded_texts.extend(token_sequences)
                    expanded_ids.extend([id for _ in range(len(token_sequences))])
                else:
                    expanded_texts.append(text)
                    expanded_ids.append(id)

            assert len(expanded_texts) == len(
                expanded_ids
            ), f"Something went wrong with the expansion of texts and ids {len(expanded_texts)} vs {len(expanded_ids)}"
            pipeline_generator = self.nlp.pipe(
                expanded_texts,
                batch_size=5,
                n_process=4,
                disable=["lemmatizer", "ner", "textcat"],
            )

            combined_tokens = []
            combined_ids = []
            for id, docs in itertools.groupby(
                zip(expanded_ids, tqdm(pipeline_generator, total=len(expanded_texts))),
                lambda x: x[0],
            ):
                docs = list(map(lambda x: x[1], docs))
                combined_ids.append(id)
                combined_tokens.append(Doc.from_docs(docs))

            all_df["tokenized"] = combined_tokens
            # pipeline_generator = self.nlp.pipe(
            #     all_texts,
            #     batch_size=5,
            #     n_process=4,
            #     disable=["lemmatizer", "ner", "textcat"]
            # )
            # all_df["tokenized"] = [d for d in tqdm(pipeline_generator, total=len(all_texts))]
            # pandarallel.initialize(progress_bar=True)
            # all_df["tokenized"] = all_df["text"].parallel_apply(self.nlp)
            del (
                all_texts,
                all_ids,
                expanded_texts,
                expanded_ids,
                combined_tokens,
                combined_ids,
            )

            print("\nGetting candidate labels")
            pandarallel.initialize(progress_bar=True)
            all_df["extra_labels"] = all_df.parallel_apply(
                extract_extra_candidates_f, axis=1
            )

            print("\nConverting documents to samples")
            convert_document_to_samples_f = partial(
                convert_document_to_samples, self.nlp
            )
            pandarallel.initialize(progress_bar=True)
            all_df["tokenized_samples"] = all_df.parallel_apply(
                convert_document_to_samples_f, axis=1
            )
            # tqdm.pandas()
            # all_df["tokenized_samples"] = all_df.progress_apply(convert_document_to_samples_f, axis=1)

            print("\nSaving samples")
            pandarallel.initialize(progress_bar=True, use_memory_fs=False)
            save_samples_f = partial(
                save_samples, self.train_files_location, self.validation_files_location
            )
            all_df.parallel_apply(save_samples_f, axis=1)

        # the training dataframe list samples by id, index, and label. In this
        # way we can also balance the dataset by label. Validation samples are
        # not included in the training set, following the practice from the
        # first place submission to the Kaggle competition.

        document_keys = all_kaggle_train.rename(columns={"Id": "id"}).loc[:, ["id"]]

        train_samples = document_keys.copy()
        train_samples["snippet_label"] = train_samples.apply(
            partial(get_snippet_labels, self.train_files_location), axis=1
        )
        train_samples["snippet_index"] = train_samples["snippet_label"].apply(
            lambda x: [i for i in range(len(x))]
        )
        train_samples = train_samples.explode(["snippet_label", "snippet_index"])

        validation_samples = document_keys.copy()
        validation_samples["snippet_label"] = validation_samples.apply(
            partial(get_snippet_labels, self.validation_files_location), axis=1
        )
        validation_samples = validation_samples.loc[validation_samples.apply(lambda x: len(x["snippet_label"]) > 0, axis=1), :]

        validation_samples["snippet_index"] = validation_samples["snippet_label"].apply(
            lambda x: [i for i in range(len(x))]
        )
        validation_samples = validation_samples.explode(["snippet_label", "snippet_index"])

        train_df, test_df = train_test_split(
            train_samples, test_size=0.2, random_state=42
        )

        train_df.to_csv(self.train_dataframe)
        test_df.to_csv(self.test_dataframe)

        validation_samples.to_csv(self.validation_dataframe)

        ros = RandomOverSampler(random_state=42, sampling_strategy=1)
        train_df, train_df["snippet_label"] = ros.fit_resample(
            train_df.drop(columns=["snippet_label"]), train_df["snippet_label"].astype(int)
        )
        train_df.to_csv(self.train_balanced_dataframe, index=False)


if __name__ == "__main__":
    keywords = [
        "Study",
        "Studies",
        "Survey",
        "Surveys",
        "Dataset",
        "Datasets",
        "Database",
        "Databases",
        "Data Set",
        "Data System",
        "Data Systems",
    ]

    repo = SnippetRepository(
        mode=SnippetRepositoryMode.NER, build_options=dict(keywords=keywords)
    )
