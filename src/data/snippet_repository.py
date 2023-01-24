# Based on the dataset used by kaggle model 1
# https://github.com/Coleridge-Initiative/rc-kaggle-models/blob/main/1st%20ZALO%20FTW/notebooks/get_candidate_labels.ipynb


from functools import partial
from itertools import starmap
import json
import logging
import os
from unidecode import unidecode
from typing import Dict, Iterator, List, Optional, Tuple, Union

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

    def __init__(self, mode, build_kwargs:Optional[Dict[str, A]]=None) -> None:
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

    def classify_text(self, nlp, text: str) -> Tuple[List[str], List[str], List[str]]:

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

        model = RegexModel({"keywords": filter_keywords})
        def extract_extra_candidates(doc_id:str) -> str:
            with open(os.path.join("../data/kaggle/train", doc_id + ".json"), "r") as f:
                text = " ".join([sec["text"].replace("\n", " ") for sec in json.load(f)])

            return model.inference({}, pd.DataFrame({"text": [text]}))["model_predictions"].values[0]

        unique_labels = kaggle_train.groupby("Id").apply(aggregate_clean_label)

        all_df = pd.DataFrame({"id": kaggle_train["Id"].unique()})
        all_df["label"] = all_df["id"].apply(lambda x: unique_labels[x])
        all_df["extra_labels"] = all_df["id"].apply(extract_extra_candidates)



        # get additional candidate labels using algorithm from kaggle model 1
        # run spacy on text to get tokens/sentences
        # tag sentences
        # filter out sentences that are additional candidates from kaggle model 1 to a the validation set
        # save to tsv

        pass


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
    results = SnippetRepository.extract_snippets(
        "data/kaggle/train",
        "d0fa7568-7d8e-4db9-870f-f9c6f668c17b",
        window_size=30,
        step_size=15
    )

    print(results)