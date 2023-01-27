# Based on the dataset used by kaggle model 1
# https://github.com/Coleridge-Initiative/rc-kaggle-models/blob/main/1st%20ZALO%20FTW/notebooks/get_candidate_labels.ipynb

from enum import Enum
from functools import partial
from itertools import starmap
import json
import logging
import os
from unidecode import unidecode
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import pandas as pd
import regex as re
import spacy
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.repository import Repository
from src.models.regex_model import RegexModel

logger = logging.getLogger("snippet_repository")

class SnippetRepositoryMode(Enum):
    NER = "ner"
    CLASSIFICATION = "classification"
    MASKED_LM = "masked_lm"

class SnippetRepository(Repository):
    """Repository for serving training snippets.



    """

    def __init__(self, mode:SnippetRepositoryMode, build_options:Optional[Dict[str, Any]]=None) -> None:
        self.mode = mode
        self.local = os.path.dirname(__file__)
        with_local_path = partial(os.path.join, self.local)

        self.train_labels_location = with_local_path("../../data/kaggle/train.csv")

        self.train_files_location = with_local_path("../../data/snippets/kaggle_snippets_train")
        self.test_files_location = with_local_path("../../data/snippets/kaggle_snippets_test")
        self.validation_files_location = with_local_path("../../data/snippets/kaggle_snippets_validation")

        make_dir_f = partial(os.makedirs, exist_ok=True)
        list(map(make_dir_f, [self.train_files_location, self.test_files_location, self.validation_files_location]))

        if len(os.listdir(self.train_files_location))==0:
            self.nlp = spacy.load("en_core_web_trf")
            self.build(build_options)


    def get_training_data(self, batch_size: Optional[int] = None, rebalance: Optional[bool] = False) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        ...

    def get_test_data(self, batch_size: Optional[int] = None) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        ...

    def get_validation_data(self, batch_size: Optional[int] = None) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        ...

    @staticmethod
    def detect_labels(labels:List[re.Pattern], sentence:str) -> List[List[str]]:
        return list(map(
            lambda match: match.captures(), # It's possible to have more than one match
            filter(
                bool,
                map(
                    lambda rl: rl.search(sentence),
                    labels
                )
            )
        ))


    def tag_sentence(
        self,
        regex_labels:List[re.Pattern],
        sentence:spacy.tokens.span.Span
    ) -> Tuple[List[str], List[str], List[str]]:

        # find matches for each label and sort by length, longest first
        # shorter matches might be a subset of a longer match. So, we'll
        # prefer the longer matches first.
        match_lists = sorted(
            SnippetRepository.detect_labels(regex_labels, sentence.text),
            key=lambda x: max(map(len, x)),
            reverse=True
        )

        tokens = [token.text for token in sentence]
        tags = [token.tag_ for token in sentence]
        ner_tags = ["O"] * len(sentence) # assume no match

        for matches in match_lists:
            for match in matches:
                try:
                    label_tokens = self.nlp(match)
                    start_idx = tokens.index(label_tokens[0].text)
                    idxs = list(range(start_idx, start_idx + len(label_tokens)))
                except Exception:
                    continue

                first_tag = ner_tags[start_idx]
                prev_tag = ner_tags[start_idx - 1] if start_idx > 0 else "O"
                # If there are any tokens that are already marked then this match
                # could be a subset of another match
                if not any(map(lambda x: x!="O", ner_tags[start_idx: start_idx + len(label_tokens)])):
                    if prev_tag=="O":
                        ner_tags[start_idx] = "I-DAT"
                    else:
                        ner_tags[start_idx] = "B-DAT"

                    for idx in idxs[1:]:
                        ner_tags[idx] = "I-DAT"

        return tokens, tags, ner_tags

    def build(self, build_options:Dict[str, Any]) -> None:
        # get training data from kaggle
        print("Loading Kaggle training labels...")
        kaggle_train = pd.read_csv(self.train_labels_location)

        def aggregate_clean_label(row: pd.DataFrame):
            labels = list(map(lambda x: x.lower().strip(), row["dataset_label"].unique()))
            return "|".join(labels)

        def get_text(doc_id:str) -> str:
            with open(os.path.join(self.local, "../../data/kaggle/train", doc_id + ".json"), "r") as f:
                text = unidecode(" ".join([sec["text"].replace("\n", " ") for sec in json.load(f)]))
            return text

        model = RegexModel({"keywords": build_options["keywords"]})
        def extract_extra_candidates(text:str) -> str:
            return model.inference({}, pd.DataFrame({"text": [text]}))["model_prediction"].values[0]

        unique_labels = kaggle_train.groupby("Id").apply(aggregate_clean_label)

        all_df = pd.DataFrame({"id": kaggle_train["Id"].unique()})
        all_df["label"] = all_df["id"].apply(lambda x: unique_labels[x])
        print("Getting text files")
        tqdm.pandas()
        all_df["text"] = all_df["id"].progress_apply(get_text)
        print("Getting candidate labels")
        tqdm.pandas()
        all_df["extra_labels"] = all_df["text"].progress_apply(extract_extra_candidates)


        def convert_document_to_samples(row:pd.DataFrame):
            text = row["text"]
            doc = self.nlp(text)
            samples = []
            labels = list(
                map(
                    re.compile,
                    map(
                        RegexModel.regexify_keyword,
                        row["label"].split("|")
                    )
                )
            )

            candidate_labels = list(
                map(
                    re.compile,
                    map(
                        RegexModel.regexify_keyword,
                        row["extra_labels"].split("|")
                    )
                )
            )
            for sentence in doc.sents:
                tokens, tags, label_ner_tags = self.tag_sentence(
                    labels,
                    sentence
                )
                tokens, tags, candidate_ner_tags = self.tag_sentence(
                    candidate_labels,
                    sentence
                )

                # if there are only tags form the labels, then
                # we'll use it training as a positive sample.
                # if there are tags from the candidate labels,
                # then we'll use it in the validation set as a
                # positive sample.
                # otherwise, it will be a negative sample
                contains_candidate = any(map(lambda x: x!="O", candidate_ner_tags))
                ner_tags = candidate_ner_tags if contains_candidate else label_ner_tags

                tagged_formatted_tokens = "\n".join(starmap(
                    lambda token, tag, ner_tag: f"{token}\t{tag}\t{ner_tag}",
                    zip(tokens, tags, ner_tags)
                ))

                samples.append({
                    "sample": tagged_formatted_tokens,
                    "is_validation": contains_candidate,
                })
            return samples

        print("Converting documents to samples")
        tqdm.pandas()
        all_df["tokenized_samples"] = all_df.progress_apply(convert_document_to_samples, axis=1)

        def save_samples(row:pd.DataFrame) -> None:
            save_train_path = os.path.join(
                self.train_files_location,
                row["id"] + ".tsv"
            )

            save_validation_path = os.path.join(
                self.train_files_location,
                row["id"] + ".tsv"
            )

            for sample in row["tokenized_samples"]:
                save_path = save_validation_path if sample["is_validation"] else save_train_path
                with open(save_path, "a") as f:
                    f.write(sample["sample"])

        tqdm.pandas()
        all_df.progress_apply(save_samples, axis=1)


if __name__=="__main__":
    repo = SnippetRepository(mode=SnippetRepositoryMode.NER, build_options=dict(
        keywords=["dataset", "data set", "data sets", "datasets"]
    ))