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
from sklearn.model_selection import train_test_split

from src.data.repository import Repository

logger = logging.getLogger("snippet_repository")

class SnippetRepository(Repository):
    """Repository for serving training snippets.

    Based on the 1st place kaggle solution:
    https://github.com/Coleridge-Initiative/rc-kaggle-models/blob/main/1st%20ZALO%20FTW/notebooks/get_candidate_labels.ipynb

    We want to aggressively extract positive labels. If the label isn't similar
    to any of the training labels, then don't use it for training. Instead,
    use it for testing/validation. We want to avoid accidently including samples
    as True Negatives that are actually positive samples.

    """

    def __init__(self) -> None:
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


    def get_training_data(self, batch_size: Optional[int] = None) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        ...

    def get_test_data(self, batch_size: Optional[int] = None) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        ...

    def get_validation_data(self, batch_size: Optional[int] = None) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        ...

    def build() -> None:
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