# Based on the dataset used by kaggle model 1
# https://github.com/Coleridge-Initiative/rc-kaggle-models/blob/main/1st%20ZALO%20FTW/notebooks/get_candidate_labels.ipynb


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

from src.data.repository import Repository
from src.models.regex_model import RegexModel

logger = logging.getLogger("snippet_repository")

class SnippetRepository(Repository):
    """Repository for serving training snippets.



    """

    def __init__(self, mode, build_kwargs:Optional[Dict[str, Any]]=None) -> None:
        self.local = os.path.dirname(__file__)
        self.train_labels_location = os.path.join(
            self.local, "../../data/kaggle/train.csv"
        )
        self.train_files_location = os.path.join(self.local, "../../data/kaggle/train")
        self.validation_files_location = os.path.join(self.local, "../../data/kaggle/validation")

        self.train_dataframe_location = os.path.join(
            self.local, "../../data/kaggle/train_snippet_dataframe.csv"
        )

        self.test_dataframe_location = os.path.join(
            self.local, "../../data/kaggle/test_snippet_dataframe.csv"
        )

        self.validation_dataframe_location = os.path.join(
            self.local, "../../data/kaggle/validation_labels.csv"
        )


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
                label_tokens = self.nlp(match)
                start_idx = tokens.index(label_tokens[0].text)
                idxs = list(range(start_idx, start_idx + len(label_tokens)))


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

    def build(self, filter_keywords:List[str]) -> None:
        # get training data from kaggle
        kaggle_train = pd.read_csv(self.train_labels_location)

        def aggregate_clean_label(row: pd.DataFrame):
            labels = list(map(lambda x: x.lower().strip(), row["dataset_label"].unique()))
            return "|".join(labels)

        def get_text(doc_id:str) -> str:
            with open(os.path.join("../data/kaggle/train", doc_id + ".json"), "r") as f:
                text = unidecode(" ".join([sec["text"].replace("\n", " ") for sec in json.load(f)]))
            return text

        model = RegexModel({"keywords": filter_keywords})
        def extract_extra_candidates(text:str) -> str:
            return model.inference({}, pd.DataFrame({"text": [text]}))["model_predictions"].values[0]

        unique_labels = kaggle_train.groupby("Id").apply(aggregate_clean_label)

        all_df = pd.DataFrame({"id": kaggle_train["Id"].unique()})
        all_df["label"] = all_df["id"].apply(lambda x: unique_labels[x])
        all_df["text"] = all_df["id"].apply(get_text)
        all_df["extra_labels"] = all_df["text"].apply(extract_extra_candidates)


        def convert_document_to_samples(row:pd.DataFrame):
            text = row["text"]
            doc = self.nlp(text)
            samples = []
            labels = list(map(RegexModel.regexify_keyword, row["label"].split("|")))
            candidate_labels = list(map(RegexModel.regexify_keyword, row["extra_labels"].split("|")))
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

        all_df["tokenized_samples"] = all_df.apply(convert_document_to_samples, axis=1)

        # flatten the list of samples per document and save to splitting out the
        # candidate labels into a separate set, and further splitting the training
        # set into a training and test set.


    @staticmethod
    def preprocess_text(text:List[Dict[str, str]]) -> str:
        """Clean text for comparison."""
        full_text = " ".join([section["text"].strip() for section in text])
        return unidecode(full_text)

    @staticmethod
    def tokenize_text(text:str) -> List[str]:
        """Tokenize text for comparison."""
        return text.split()

    @staticmethod
    def gen_sliding_window(sample_len:int, win_size:int, step_size:int) -> List[Tuple[int, int]]:
        """Generate sliding windows indicies for extracting subsets of the text."""

        starts = filter(
            lambda x: x + win_size <= sample_len,
            range(0, sample_len, step_size)
        )
        windows = list(map(
            lambda x: [x, x + win_size],
            starts
        ))

        if windows[-1][1] != sample_len:
            windows.append((sample_len - win_size, sample_len))

        return windows


    @staticmethod
    def extract_snippets(
        location:str,
        doc_id:str,
        window_size:int=30,
        step_size:Optional[int]=None
    ) -> pd.DataFrame:
        """Extract snippets from a given document."""

        step_size = step_size or window_size // 2

        with open(os.path.join(location, f"{doc_id}.json")) as f:
            document = json.load(f)

        full_text = SnippetRepository.preprocess_text(document)
        tokens = SnippetRepository.tokenize_text(full_text)

        windows = SnippetRepository.gen_sliding_window(
            len(tokens),
            window_size,
            step_size,
        )

        snippets = list(starmap(
            lambda start, end: " ".join(tokens[start:end]),
            windows
        ))
        n = len(snippets)
        return pd.DataFrame({
            "document_id": [doc_id] * n,
            "text": snippets,
            "window": windows,
            "label": ["unknown"] * n,
        })




if __name__=="__main__":