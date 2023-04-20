import dataclasses as dc
from functools import partial
import glob
from itertools import filterfalse
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import spacy
import tensorflow as tf
from tqdm import tqdm
import transformers as tfs

from democratizing_data_ml_algorithms.data.repository import Repository
import democratizing_data_ml_algorithms.models.base_model as bm

from democratizing_data_ml_algorithms.models.kaggle_model1_support.model_1_QueryDataLoader import (
    QueryDataLoader,
)
from democratizing_data_ml_algorithms.models.kaggle_model1_support.model_1_SupportQueryDataLoader import (
    SupportQueryDataLoader,
)
from democratizing_data_ml_algorithms.models.kaggle_model1_support.model_1_MetricLearningModel_static import (
    MetricLearningModel,
)


def train(
    repository: Repository,
    config: Dict[str, Any],
    training_logger: Optional[bm.SupportsLogging] = None,
) -> None:
    pass


def validate(repository: Repository, config: Dict[str, Any]) -> None:
    pass


class KaggleModel1(bm.Model):
    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_sm")
        self.model = None

    def train(
        self,
        repository: Repository,
        config: Dict[str, Any],
        exp_logger: bm.SupportsLogging,
    ) -> None:
        raise NotImplementedError(
            "This class only runs pretrained models from the competition."
        )

    def sentencize_text(self, text: str) -> List[str]:
        max_tokens = 10_000

        tokens = text.split()
        if len(tokens) > max_tokens:
            texts = [
                " ".join(tokens[i : i + max_tokens])
                for i in range(0, len(tokens), max_tokens)
            ]
            tokens = tokens[:max_tokens]
        else:
            texts = [text]

        process_generator = self.nlp.pipe(
            texts,
            disable=["lemmatizer", "ner", "textcat"],
        )

        sents = []
        for doc in process_generator:
            sents.extend([sent.text for sent in doc.sents])

        return sents

    def get_support_mask_embed(self, path: str, n_samples: int) -> np.ndarray:

        support_tokens = np.load(path)  # [n, embed_dim]

        # print("getting", n_samples, "from", path, support_tokens.shape)

        # sample_idxs = np.random.choice(
        #     np.arange(support_tokens.shape[0]),
        #     n_samples
        # )

        # support_tokens = np.mean(
        #     support_tokens[sample_idxs, :], # [n, embed_dim]
        #     axis=0,
        #     keepdims=True,
        # )[np.newaxis, ...].astype(np.float32) # [1, 1, embed_dim]

        return support_tokens

    def slice_up_samples(self, seq_len: int, row: pd.DataFrame) -> Dict[str, str]:
        tokens = row["text"].replace("\n", " ").split()

        split_up = [
            " ".join(tokens[i * seq_len : (i + 1) * seq_len])
            for i in range(len(tokens) // seq_len)
        ]

        return dict(
            id=[row["id"]] * len(split_up),
            text=split_up,
        )

    def _slice_up_text(self, seq_len: int, text: str) -> List[str]:
        tokens = text.replace("\n", " ").split()

        split_up = [
            " ".join(tokens[i * seq_len : (i + 1) * seq_len])
            for i in range(len(tokens) // seq_len)
        ]

        return split_up

    def is_special_token(token):
        return (
            token.startswith("[")
            and token.endswith("]")
            or token.startswith("<")
            and token.endswith(">")
        )

    def merge_tokens_w_classifications(
        self,
        tokens: List[str],
        token_should_be_merged: List[bool],
        classifications: List[float],
    ) -> List[Tuple[str, float]]:
        merged = []
        for token, do_merge, classification in zip(
            tokens, token_should_be_merged, classifications
        ):
            if do_merge:
                merged[-1] = (merged[-1][0] + token, merged[-1][1])
            else:
                merged.append((token, classification))
        return merged

    def high_probablity_token_groups(
        self,
        tokens_classifications: List[Tuple[str, float]],
        threshold: float = 0.9,
    ) -> List[List[Tuple[str, float]]]:

        datasets = []
        dataset = []
        for token, score in tokens_classifications:
            # REFACTOR HERE TO RECOVER SCORES
            if score >= threshold:
                dataset.append(token)
            else:
                if len(dataset) > 0:
                    datasets.append(" ".join(dataset))
                    dataset = []
        if len(dataset) > 0:
            datasets.append(" ".join(dataset))

        return datasets

    def inference(self, config: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:

        if self.model is None:
            model_config = tfs.AutoConfig.from_pretrained(
                config["model_tokenizer_name"]
            )
            model_config.output_attentions = True
            model_config.output_hidden_states = True

            ll_model = tfs.TFAutoModel.from_config(config=model_config)
            model = MetricLearningModel(
                config=model_config, name="metric_learning_model"
            )
            model.main_model = ll_model
            model.K = 3

            self.tokenizer = tfs.AutoTokenizer.from_pretrained(
                config["model_tokenizer_name"]
            )
            self.tokenizer_special_tokens = set(
                self.tokenizer.special_tokens_map.values()
            )
            mask_embedding = self.get_support_mask_embed(
                config["support_mask_embedding_path"],
                config["n_support_samples"],
            )  # [n,     embed_dim]

            no_mask_embedding = self.get_support_mask_embed(
                config["support_no_mask_embedding_path"],
                config["n_support_samples"],
            )  # [n,     embed_dim]

            # empty mock values are passed to the model to build the tensorflow
            # graph. We don't need the outputs
            mock_input_ids = np.zeros(
                [config["batch_size"], config["seq_len"]], dtype=np.int32
            )
            mock_attention_mask = np.zeros(
                [config["batch_size"], config["seq_len"]], dtype=np.int32
            )
            _ = model(
                [
                    mock_input_ids,
                    mock_attention_mask,
                ],
                training=True,
                sequence_labels=None,
                mask_embeddings=mask_embedding[:1, :],
                nomask_embeddings=no_mask_embedding[:1, :],
            )

            weights_path = glob.glob(os.path.join(config["weights_path"], "*.h5"))[0]
            model.load_weights(weights_path, by_name=True)
            self.model = tf.function(model, experimental_relax_shapes=True)

        if config.get("is_roberta", False):
            print("Merging tokens based on Roberta tokenizer")
            should_merge = lambda t: not t.startswith("Ġ") and not t.startswith("<")
            clean = lambda t: t.replace("Ġ", "")
        else:
            print("Merging tokens based on BERT tokenizer")
            should_merge = lambda t: t.startswith("##")
            clean = lambda t: t.replace("##", "")

        def infer_sample(text: str) -> str:
            sents = self._slice_up_text(config["seq_len"], text)

            datasets = []
            for batch in spacy.util.minibatch(sents, config["batch_size"]):
                # batch is a list of string, the tokenizer will return a
                # dictionary with at least the keys "input_ids" and
                # "attention_mask" where the values are tensorflow tensors
                tokenized_batch = self.tokenizer(
                    batch,
                    return_tensors="tf",
                    max_length=config["seq_len"],
                    truncation=True,
                    padding=True,
                )

                # the embeddings are randomly sampled from the sets of embeddings
                batch_mask_embeddings = mask_embedding[
                    np.random.choice(
                        np.arange(mask_embedding.shape[0]),
                        config["n_support_samples"],
                    ),
                    ...,
                ].mean(
                    axis=0, keepdims=True
                )  # [1, embed_dim]

                batch_no_mask_embeddings = no_mask_embedding[
                    np.random.choice(
                        np.arange(no_mask_embedding.shape[0]),
                        config["n_support_samples"],
                    ),
                    ...,
                ].mean(
                    axis=0, keepdims=True
                )  # [1, embed_dim]

                (_, _, _, attention_values) = self.model(
                    [
                        tokenized_batch["input_ids"],
                        tokenized_batch["attention_mask"],
                    ],
                    training=False,
                    sequence_labels=None,
                    mask_embeddings=batch_mask_embeddings,
                    nomask_embeddings=batch_no_mask_embeddings,
                )

                tokens = list(
                    map(
                        self.tokenizer.convert_ids_to_tokens,
                        tokenized_batch["input_ids"].numpy(),
                    )
                )

                for sent, sent_classification in zip(
                    tokens,
                    attention_values.numpy()[..., 0],
                ):
                    assert len(sent) == len(
                        sent_classification
                    ), f"Classification length mismatch {len(sent)} != {len(sent_classification)}"

                    tokens_classifications = list(
                        filterfalse(
                            lambda x: x[0] in self.tokenizer_special_tokens,
                            self.merge_tokens_w_classifications(
                                list(map(clean, sent)),
                                list(map(should_merge, sent)),
                                sent_classification,
                            ),
                        )
                    )

                    detections = self.high_probablity_token_groups(
                        tokens_classifications, threshold=config.get("threshold", 0.9)
                    )  # List[List[Tuple[str, float]]]

                    datasets.extend(detections)

            return "|".join(set(datasets))

        if config.get("inference_progress_bar", False):
            tqdm.pandas()
            df["model_prediction"] = df["text"].progress_apply(infer_sample)
        else:
            df["model_prediction"] = df["text"].apply(infer_sample)

        return df


def entry_point():
    bm.train = train
    bm.validate = validate
    bm.main()


if __name__ == "__main__":
    entry_point()

    # import src.data.kaggle_repository as kr
    # import src.evaluate.model as em

    # class MockRepo:
    #     def __init__(self, df):
    #         self.df = df
    #     def get_validation_data(self):
    #         return self.df
    #     def copy(self):
    #         return MockRepo(self.df.copy())

    # biomed roberta model config
    # config = dict(
    #     support_mask_embedding_path = "models/kaggle_model1/sub_biomed_roberta/embeddings/support_mask_embeddings.npy",
    #     support_no_mask_embedding_path = "models/kaggle_model1/sub_biomed_roberta/embeddings/support_nomask_embeddings.npy",
    #     n_support_samples = 100,
    #     model_tokenizer_name = "models/kaggle_model1/sub_biomed_roberta",
    #     weights_path = "models/kaggle_model1/sub_biomed_roberta/embeddings",
    #     batch_size = 128,
    #     seq_len = 320,
    #     is_roberta = True,
    # )

    # scibert model config
    # config = dict(
    #     support_mask_embedding_path = "models/kaggle_model1/sub_scibert/embeddings/support_mask_embeddings.npy",
    #     support_no_mask_embedding_path = "models/kaggle_model1/sub_scibert/embeddings/support_nomask_embeddings.npy",
    #     n_support_samples = 100,
    #     model_tokenizer_name = "models/kaggle_model1/sub_scibert",
    #     weights_path = "models/kaggle_model1/sub_scibert/embeddings",
    #     batch_size = 128,
    #     seq_len = 320,
    #     is_roberta = False,
    # )

    # model = KaggleModel1()

    # sample = kr.KaggleRepository().get_training_sample_by_id(
    #     "3af0a4ad-2fd3-430f-880b-c0c8c1b097e1"
    # )
    # # sample["text"] = sample["text"]

    # outs = KaggleModel1().inference(config, sample)

    # print(outs)

    # repo = MockRepo(next(kr.KaggleRepository().get_validation_data(batch_size=32)))

    # outs = em.evaluate_model(
    #     repo.copy(),
    #     KaggleModel1(),
    #     config,
    # )
