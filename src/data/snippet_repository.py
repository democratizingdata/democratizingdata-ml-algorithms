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
from pandarallel import pandarallel
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.repository import Repository
from src.models.regex_model import RegexModel

import warnings
warnings.filterwarnings('ignore') # setting ignore as a parameter

logger = logging.getLogger("snippet_repository")



def get_text_per_row(path_dir:str, row:pd.DataFrame) -> str:
    doc_id:str = row["id"]
    with open(os.path.join(path_dir, doc_id + ".json"), "r") as f:
        text = unidecode(" ".join([sec["text"].replace("\n", " ") for sec in json.load(f)]))
    return text


def extract_extra_candidates(model:RegexModel, keywords:List[str], row:pd.DataFrame) -> str:
    text = row["text"]
    all_entities = model.inference({}, pd.DataFrame({"text": [text]}))["model_prediction"]
    stripped_and_split = [v.strip() for v in all_entities.values[0].split("|")]

    contains_keyword = list(filter(
        lambda x: any(map(lambda y: y.lower() in x.lower(), keywords)),
        stripped_and_split
    ))

    return "|".join(contains_keyword)


def tag_sentence(
    nlp:spacy.language.Language,
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
                label_tokens = nlp(match)
                start_idx = tokens.index(label_tokens[0].text)
                idxs = list(range(start_idx, start_idx + len(label_tokens)))
            except ValueError:
                # print(f"Could not find {str(match)} in sentence: ",  sentence.text)
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


def convert_document_to_samples(nlp:spacy.language.Language, row:pd.DataFrame):
    text = row["text"]
    doc = nlp(text)
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
        tokens, tags, label_ner_tags = tag_sentence(
            nlp,
            labels,
            sentence
        )

        contains_label = any(map(lambda x: x!="O", label_ner_tags))

        if not contains_label:
            _, _, candidate_ner_tags = tag_sentence(
                nlp,
                candidate_labels,
                sentence
            )
        else:
            candidate_ner_tags = ["O"] * len(sentence)


        contains_candidate = any(map(lambda x: x!="O", candidate_ner_tags))
        # if there are only tags form the labels, then
        # we'll use it training as a positive sample.
        # if there are tags from the candidate labels,
        # then we'll use it in the validation set as a
        # positive sample.
        # otherwise, it will be a negative sample
        ner_tags = label_ner_tags if contains_label else candidate_ner_tags

        tagged_formatted_tokens = "".join(starmap(
            lambda token, tag, ner_tag: f"{token}\t{tag}\t{ner_tag}\n",
            zip(tokens, tags, ner_tags)
        ))

        samples.append({
            "sample": tagged_formatted_tokens,
            "is_validation": contains_candidate,
        })

    if len(samples) == 0:
        print("No samples found for document: ", row["id"])

    return samples

def save_samples(train_location:str, validation_location:str, row:pd.DataFrame) -> None:
    save_train_path = os.path.join(
        train_location,
        row["id"] + ".tsv"
    )

    save_validation_path = os.path.join(
        validation_location,
        row["id"] + ".tsv"
    )

    try:
        for sample in row["tokenized_samples"]:
            save_path = save_validation_path if sample["is_validation"] else save_train_path
            with open(save_path, "a") as f:
                f.write(sample["sample"])
    except:
        print(row)



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

        # if len(os.listdir(self.train_files_location))==0:
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




    def build(self, build_options:Dict[str, Any]) -> None:
        # get training data from kaggle
        print("Loading Kaggle training labels...")
        kaggle_train = pd.read_csv(self.train_labels_location)

        def aggregate_clean_label(row: pd.DataFrame):
            labels = list(map(lambda x: x.lower().strip(), row["dataset_label"].unique()))
            return "|".join(labels)

        existing_files = list(map(
            lambda x: x.split(".")[0],
            os.listdir(self.train_files_location)
        ))

        existing_files.extend(list(map(
            lambda x: x.split(".")[0],
            os.listdir(self.validation_files_location)
        )))

        model = RegexModel(config=dict())
        extract_extra_candidates_f = partial(extract_extra_candidates, model, build_options["keywords"])

        unique_labels = kaggle_train.groupby("Id").apply(aggregate_clean_label)

        ids_to_work_on = list(filter(
            lambda x: x not in existing_files,
            kaggle_train["Id"].unique()
        ))

        all_df = pd.DataFrame({"id": ids_to_work_on}).iloc[:100, :]
        all_df["label"] = all_df["id"].apply(lambda x: unique_labels[x])


        print("Getting text files")
        pandarallel.initialize(progress_bar=True)
        get_text_f = partial(get_text_per_row, os.path.join(self.local, "../../data/kaggle/train"))
        all_df["text"] = all_df.parallel_apply(get_text_f, axis=1)


        print("Getting candidate labels")
        pandarallel.initialize(progress_bar=True)
        all_df["extra_labels"] = all_df.parallel_apply(extract_extra_candidates_f, axis=1)


        print("Converting documents to samples")
        convert_document_to_samples_f = partial(convert_document_to_samples, self.nlp)
        # pandarallel.initialize(progress_bar=True)
        # all_df["tokenized_samples"] = all_df.parallel_apply(convert_document_to_samples_f, axis=1)
        tqdm.pandas()
        all_df["tokenized_samples"] = all_df.progress_apply(convert_document_to_samples_f, axis=1)


        print("Saving samples")
        pandarallel.initialize(progress_bar=True, use_memory_fs=False)
        save_samples_f = partial(save_samples, self.train_files_location, self.validation_files_location)
        all_df.parallel_apply(save_samples_f, axis=1)




if __name__=="__main__":
    keywords = [
        "Study", "Studies", "Survey", "Surveys", "Dataset", "Datasets",
        "Database", "Databases", "Data Set", "Data System", "Data Systems"
    ]

    repo = SnippetRepository(mode=SnippetRepositoryMode.NER, build_options=dict(
        keywords=keywords
    ))