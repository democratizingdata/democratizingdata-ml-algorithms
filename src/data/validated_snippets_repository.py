import os

from enum import Enum
from functools import partial
from typing import Callable, Iterator, List, Optional, Union
import logging

import regex as re
import pandas as pd
from src.data.repository import Repository, SnippetRepositoryMode
import spacy


# TODO: this should be defined in a config file somewhere
VALIDATED_SNIPPET_PATH: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "data",
    "snippets",
    "ncses_run_21_model1_snippets.4.18.2023.csv",
)

logger = logging.getLogger("ValidatedSnippetsRepository")

class ValidatedSnippetsRepository(Repository):
    def __init__(self, mode: SnippetRepositoryMode):
        self.mode = mode
        self.nlp = spacy.load("en_core_web_sm")
        self.path = VALIDATED_SNIPPET_PATH
        self.train_frac = 0.8
        n_rows = sum(1 for _ in open(VALIDATED_SNIPPET_PATH, "r"))
        self.n_train_rows = int(n_rows * self.train_frac)
        self.skiprows = self.n_train_rows

    def transform_df_tokenize(self, row: pd.DataFrame) -> List[str]:
        return list(
            filter(
                lambda s: s.strip() != "",
                map(
                    lambda s: str(s),
                    self.nlp(
                        re.subn(
                            r"[\\][\\n]",
                            " ",
                            row["snippet"],
                        )[0].strip(),
                    ).doc,
                )
            )
        )

    def transform_df_ner(self, row: pd.DataFrame) -> pd.DataFrame:
        tokens = self.transform_df_tokenize(row)
        lbl_tokens = row["mention_candidate"].split()
        lbl = ["B-DAT"] + ["I-DAT"] * (len(lbl_tokens) - 1)
        token_lbls = ["O"] * len(tokens)

        try:
            lower_tokens = list(map(lambda t: t.lower(), tokens))
            start = next(filter(
                lambda i: lower_tokens[i].startswith(lbl_tokens[0]),
                range(len(lower_tokens))
            ))
            token_lbls[start : start + len(lbl_tokens)] = lbl
        except Exception as e:
            print(lbl_tokens[0], "not in", lower_tokens)
            pass

        return pd.DataFrame(
            {
                "text": [tokens],
                "tags": [list(map(lambda t: "UNK", tokens))],
                "ner_tags": [token_lbls],
            }
        )

    def transform_df_classification(self, row: pd.DataFrame) -> pd.DataFrame:
        tokens = self.transform_df_tokenize(row)

        return pd.DataFrame({"text": [tokens], "label": [1.0]})

    def transform_df_masked_lm(self, row: pd.DataFrame) -> pd.DataFrame:
        row = self.transform_df_ner(row)

        row["mask"] = row["ner_tags"].apply(
            lambda x: [t in ["B-DAT", "I-DAT"] for t in x]
        )
        row["label"] = 1.0

        return row.loc[:, ["text", "mask", "tags", "label"]]

    def transform_df(self, is_test: bool, df: pd.DataFrame) -> pd.DataFrame:
        def double_check(
            val: Union[SnippetRepositoryMode, str],
            mode: SnippetRepositoryMode,
        ) -> bool:
            return val == mode or val == mode.value

        if double_check(self.mode, SnippetRepositoryMode.NER):
            return df.apply(self.transform_df_ner, axis=1)
        elif double_check(self.mode, SnippetRepositoryMode.CLASSIFICATION):
            return df.apply(self.transform_df_classification, axis=1)
        elif double_check(self.mode, SnippetRepositoryMode.MASKED_LM):
            return df.apply(self.transform_df_masked_lm, axis=1)
        else:
            raise ValueError(f"Unknown mode {self.mode}")

    def get_iter_or_df(
        self,
        path: str,
        is_test: bool,
        transform_f: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x,
        batch_size: Optional[int] = None,
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:

        if is_test:
            extras = dict(
                skiprows = self.n_train_rows,
                names = ['Unnamed: 0', 'dyad_id', 'm1_score', 'mention_candidate', 'snippet']
            )
        else:
            extras = dict(nrows = self.n_train_rows)

        def iter_f():
            for batch in pd.read_csv(path, chunksize=batch_size, **extras):
                yield batch

        if batch_size:
            return map(transform_f, iter_f())
        else:
            df = pd.read_csv(path, **extras)
            return transform_f(df)

    def get_training_data(
        self, batch_size: Optional[int] = None, balance_labels: bool = False,
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:

        if balance_labels:
            logger.warning(
                "Balancing labels is not implemented for ValidatedSnippetsRepository, ignoring"
            )

        transform_f = partial(self.transform_df, False)
        aggregate_f = lambda x: pd.concat(x.values, ignore_index=True)
        transform_aggregate_f = lambda x: aggregate_f(transform_f(x))
        return self.get_iter_or_df(self.path, False, transform_aggregate_f, batch_size)


    def get_validation_data(
        self, batch_size: Optional[int] = None
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        transform_f = partial(self.transform_df, False)
        aggregate_f = lambda x: pd.concat(x.values, ignore_index=True)
        transform_aggregate_f = lambda x: aggregate_f(transform_f(x))
        return self.get_iter_or_df(self.path, True, transform_aggregate_f, batch_size)


if __name__ == "__main__":
    print("NER ==================================================")
    print("Training")
    repo = ValidatedSnippetsRepository(SnippetRepositoryMode.NER)
    # print(next(repo.get_training_data(batch_size=5)))

    print("Validation")
    print(next(repo.get_validation_data(batch_size=5)))

    print("CLASSIFICATION =======================================")
    repo = ValidatedSnippetsRepository(SnippetRepositoryMode.CLASSIFICATION)
    print("Training")
    print(next(repo.get_training_data(batch_size=5)))

    print("Validation")
    print(next(repo.get_validation_data(batch_size=5)))

    print("MASKED LM ============================================")
    repo = ValidatedSnippetsRepository(SnippetRepositoryMode.MASKED_LM)
    print("Training")
    print(next(repo.get_training_data(batch_size=5)))

    print("Validation")
    print(next(repo.get_validation_data(batch_size=5)))
