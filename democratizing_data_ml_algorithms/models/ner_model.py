# BSD 3-Clause License

# Copyright (c) 2023, AUTHORS
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Extract entities from text using an Named Entity Recognition (NER) model.

This model uses an NER based model from the HuggingFace transformers library.

It uses the following NER tags:
 - "O": 0
 - "B-DAT": 1
 - "I-DAT": 2

Example:

    >>> import pandas as pd
    >>> import democratizing_data_ml_algorithms.models.ner_model as ner
    >>> df = pd.DataFrame({"text": ["This is a sentence with an entity in it."]})
    >>> config = {
    >>>     "model_tokenizer_name": "path/to/model_and_tokenizer",
    >>>     "optimizer": "torch.optim.AdamW",
    >>> }
    >>> model = ner.NERModel_pytorch(config)
    >>> df = rm.inference(config, df)
"""

import logging
import os
import warnings
from functools import partial
from itertools import filterfalse
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm

from democratizing_data_ml_algorithms.models.text_segmentizer_protocol import (
    TextSegmentizer,
)

try:
    import datasets as ds
    import torch
    import transformers as tfs
    from datasets.utils.logging import disable_progress_bar
    from transformers import AutoModelForTokenClassification, AutoTokenizer
except ImportError:
    raise ImportError("Running NerModel requires extras 'ner_model' or 'all'")

disable_progress_bar()

import democratizing_data_ml_algorithms.models.base_model as bm
from democratizing_data_ml_algorithms.data.repository import Repository
from democratizing_data_ml_algorithms.models.spacy_default_segementizer import (
    DefaultSpacySegmentizer,
)

MODEL_OBJECTS = Tuple[
    tfs.modeling_utils.PreTrainedModel,
    tfs.tokenization_utils_base.PreTrainedTokenizerBase,
    tfs.data.data_collator.DataCollatorMixin,
]
MODEL_OBJECTS_WITH_OPTIMIZER = Tuple[
    tfs.modeling_utils.PreTrainedModel,
    tfs.tokenization_utils_base.PreTrainedTokenizerBase,
    tfs.data.data_collator.DataCollatorMixin,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler._LRScheduler,
]

logger = logging.getLogger("ner_model")

EXPECTED_KEYS = {
    "model_tokenizer_name",
}


def train(
    repository: Repository,
    config: Dict[str, Any],
    training_logger: bm.SupportsLogging,
) -> None:
    """Train the model.

    Args:
        repository (Repository): The repository containing the data.
        config (Dict[str, Any]): The config for the model.
        training_logger (Optional[bm.SupportsLogging], optional): The logger to use for training. Defaults to None.

    Returns:
        None
    """

    bm.validate_config(EXPECTED_KEYS, config)
    training_logger.log_parameters(bm.flatten_hparams_for_logging(config))
    model = NERModel_pytorch()
    model.train(repository, config, training_logger)


def inference(config: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    bm.validate_config(EXPECTED_KEYS, config)
    model = NERModel_pytorch()
    return model.inference(config, df)


# ==============================================================================
# Preparing the batch for training (https://huggingface.co/docs/transformers/tasks/token_classification#preprocess)
# Mapping all tokens to their corresponding word with the word_ids method.
# Assigning the label -100 to the special tokens [CLS] and [SEP] so they’re ignored by the PyTorch loss function.
# Only labeling the first token of a given word. Assign -100 to other subtokens from the same word.
# ==============================================================================
def convert_ner_tags_to_ids(
    lbl_to_id: Dict[str, int], ner_tags: List[List[str]]
) -> List[int]:
    """Converts a list of NER tags to a list of NER tag ids.

    Args:
        lbl_to_id (Dict[str, int]): A mapping from NER tag to NER tag id.
        ner_tags (List[List[str]]): A list of NER tags.

    Returns:
        List[int]: A list of NER tag ids.
    """
    tag_f = lambda ner_tags: [lbl_to_id[ner_tag] for ner_tag in ner_tags]
    return [tag_f(ner_tags) for ner_tags in ner_tags]


def convert_sample_ner_tags_to_ids(
    lbl_to_id: Dict[str, int], sample: Dict[str, Any]
) -> Dict[str, Any]:
    """Converts the NER tags in a sample to NER tag ids.

    Args:
        lbl_to_id (Dict[str, int]): A mapping from NER tag to NER tag id.
        sample (Dict[str, Any]): A sample should have a key "labels" with a
                                 value of a list of NER tags.


    Returns:
        Dict[str, Any]: A sample with NER tags converted to NER tag ids. Replacing
                        the key "labels" in place.
    """

    sample["labels"] = convert_ner_tags_to_ids(lbl_to_id, sample["labels"])
    return sample


def tokenize_and_align_labels(
    tokenizer_f: tfs.tokenization_utils_base.PreTrainedTokenizerBase,
    examples: Dict[str, Any],
) -> Dict[str, Any]:
    """Tokenize and align labels.

    Args:
        tokenizer_f (tfs.tokenization_utils_base.PreTrainedTokenizerBase): The tokenizer to use.
        examples (Dict[str, Any]): The examples to tokenize and align labels for.

    Returns:
        Dict[str, Any]: The tokenized and aligned labels.
    """
    tokenized_inputs = tokenizer_f(examples["text"])

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100] * len(word_ids)  # assume all tokens are special
        top_word_id = max(map(lambda x: x if x else -1, word_ids))
        for word_idx in range(top_word_id + 1):
            label_ids[word_ids.index(word_idx)] = label[word_idx]
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def prepare_batch(
    tokenizer: tfs.tokenization_utils_base.PreTrainedTokenizerBase,
    data_collator: tfs.data.data_collator.DataCollatorMixin,
    lbl_to_id: Dict[str, int],
    config: Dict[str, Any],
    batch: pd.DataFrame,
) -> Dict[str, torch.Tensor]:

    ner_to_id_f = partial(convert_sample_ner_tags_to_ids, lbl_to_id)
    tokenize_f = partial(
        tokenize_and_align_labels,
        partial(
            tokenizer,
            **config.get("tokenizer_call_kwargs", {}),
        ),
    )

    transformed_batch = (
        ds.Dataset.from_pandas(
            batch.drop(columns=["tags"]).rename(columns={"ner_tags": "labels"})
        )
        .map(ner_to_id_f, batched=True)
        .map(tokenize_f, batched=True, remove_columns=["text"])
    )

    return data_collator(list(transformed_batch))


def lbl_to_color(lbl):
    import matplotlib.colors as mcolors

    value = 1 - lbl[0]
    saturation = max(lbl[1], lbl[2]) - min(lbl[1], lbl[2])
    BLUE = 240 / 360
    RED = 360 / 360
    hue = BLUE * lbl[1] + RED * lbl[2]
    return mcolors.hsv_to_rgb([hue, saturation, value])


# based on
# https://stackoverflow.com/questions/36264305/matplotlib-multi-colored-title-text-in-practice
def color_text_figure(tokens, colors_true, colors_pred):  # pragma: no cover
    # print("tokens", tokens)
    # print("colors_true", colors_true)
    # print("colors_pred", colors_pred)
    import matplotlib.pyplot as plt

    f, ax = plt.subplots(figsize=(10, 1))
    ax.set_title(
        "Top: Predicted, Bottom: True, Red:B-DAT, Blue:I-DAT, Black:O, Grey/White:Uncertain"
    )
    r = f.canvas.get_renderer()
    ax.set_axis_off()
    space = 0.025
    w = 0.0

    for i, (token, color_true, color_pred) in enumerate(
        zip(
            tokens,
            list(map(lbl_to_color, colors_true)),
            list(map(lbl_to_color, colors_pred)),
        )
    ):
        t = ax.text(
            w, 0.25, token, color=color_true, ha="left", va="center", fontsize=18
        )
        ax.text(w, 0.75, token, color=color_pred, ha="left", va="center", fontsize=18)
        transf = ax.transData.inverted()
        bb = t.get_window_extent(renderer=f.canvas.renderer)
        bb = bb.transformed(transf)
        w = w + bb.xmax - bb.xmin + space
    return f


def merge_tokens_w_classifications(
    tokens: List[str], classifications: List[float]
) -> List[Tuple[str, float]]:
    merged = []
    for token, classification in zip(tokens, classifications):
        if token.startswith("##"):
            merged[-1] = (merged[-1][0] + token[2:], merged[-1][1])
        else:
            merged.append((token, classification))
    return merged


def is_special_token(token):
    return token.startswith("[") and token.endswith("]")


def high_probablity_token_groups(
    tokens_classifications: List[Tuple[str, float]],
    threshold: float = 0.9,
) -> List[List[Tuple[str, float]]]:

    datasets = []
    dataset = []
    for token, score in tokens_classifications:
        if score >= threshold:
            dataset.append((token, score))
        else:
            if len(dataset) > 0:
                datasets.append(dataset)
                dataset = []
    if len(dataset) > 0:
        datasets.append(dataset)

    return datasets


class NERModel_pytorch(bm.Model):
    """A PyTorch NER model to detect datasets in the text."""

    lbl_to_id = {"O": 0, "B-DAT": 1, "I-DAT": 2}
    id_to_lbl = {v: k for k, v in lbl_to_id.items()}

    def get_model_objects(
        self,
        config: Dict[str, Any],
        include_optimizer: bool = False,
    ) -> Union[MODEL_OBJECTS, MODEL_OBJECTS_WITH_OPTIMIZER]:

        tokenizer = AutoTokenizer.from_pretrained(
            config["model_tokenizer_name"],
            **config.get("tokenizer_kwargs", {}),
        )

        collator = tfs.data.data_collator.DataCollatorForTokenClassification(
            tokenizer,
            return_tensors="pt",
        )

        model = AutoModelForTokenClassification.from_pretrained(
            config["model_tokenizer_name"],
            num_labels=len(self.lbl_to_id),
            id2label=self.id_to_lbl,
            label2id=self.lbl_to_id,
            **config.get("model_kwargs", {}),
        )
        if torch.cuda.is_available():
            model = model.cuda()

        if include_optimizer:
            optimizer = eval(config["optimizer"])(
                model.parameters(), **config.get("optimizer_kwargs", {})
            )

            if "scheduler" in config:
                scheduler = eval(config["scheduler"])(
                    optimizer, **config.get("scheduler_kwargs", {})
                )
            else:
                scheduler = bm.MockLRScheduler()

            return model, tokenizer, collator, optimizer, scheduler

        else:
            return model, tokenizer, collator

    def resolve_segmentizer(self, config: Dict[str, Any]) -> TextSegmentizer:
        if "segmentizer" in config:
            segmentizer_class = config["segmentizer"]
            segmentizer_kwargs = config.get("segmentizer_kwargs", {})
            if type(segmentizer_class) is str:
                import democratizing_data_ml_algorithms

                segmentizer = eval(segmentizer_class)(segmentizer_kwargs)
            else:
                segmentizer = segmentizer_class(segmentizer_kwargs)
        else:
            warnings.warn("No segmentizer provided, using spacy sentence segmentation")
            segmentizer = DefaultSpacySegmentizer()

        return segmentizer

    def inference(self, config: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model, tokenizer, collator = self.get_model_objects(
            config, include_optimizer=False
        )

        segmentizer = self.resolve_segmentizer(config)

        model.eval()
        model.to(device)
        ng = torch.no_grad()
        ng.__enter__()

        def infer_sample(text: str) -> str:
            sents = segmentizer(text)  # List[List[str]]
            assert len(sents) > 0, "No segments found in text"

            datasets, contexts = [], []
            for _batch in spacy.util.minibatch(sents, config["batch_size"]):
                batch = tokenizer(
                    [b.split() for b in _batch],
                    return_tensors="pt",
                    padding=True,
                    **config.get("tokenizer_call_kwargs", {}),
                )
                batch.to(device)

                outputs = model(**batch)

                # [batch, seq_len, 3]
                classifications = (
                    torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
                )

                # [batch, seq_len]
                # we're going to merge the B and I labels into one label and
                # assume token continuity
                token_classification = classifications[:, :, 1:].sum(axis=-1)

                tokens = list(
                    map(
                        tokenizer.convert_ids_to_tokens,
                        batch["input_ids"].cpu().numpy(),
                    )
                )  # [batch, seq_len]

                for sent, sent_classification in zip(tokens, token_classification):
                    assert len(sent) == len(
                        sent_classification
                    ), "Classification length mismatch"
                    t_classifications = list(
                        filterfalse(
                            lambda x: is_special_token(x[0]),
                            merge_tokens_w_classifications(sent, sent_classification),
                        )
                    )

                    detections = high_probablity_token_groups(
                        t_classifications, threshold=config.get("threshold", 0.9)
                    )  # List[List[Tuple[str, float]]]

                    datasets.extend(detections)
                    contexts.extend([sent] * len(detections))

            matches = "|".join(
                list(map(lambda x: " ".join(map(lambda y: y[0], x)), datasets))
            )

            confidences = "|".join(
                list(map(lambda x: " ".join(map(lambda y: y[1], x)), datasets))
            )

            snippets = "|".join(contexts)

            return matches, snippets, confidences



        df[["model_prediction", "prediction_snippet", "prediction_confidence"]] = df.apply(
            lambda x: infer_sample(x["text"]),
            result_type="expand",
            axis=1,
        )

        # if config.get("inference_progress_bar", False):
        #     tqdm.pandas()
        #     df["model_prediction"] = df["text"].progress_apply(infer_sample)
        # else:
        #     df["model_prediction"] = df["text"].apply(infer_sample)

        ng.__exit__(None, None, None)

        return df

    def filter_by_idx(self, tokenizer, batch, outputs, idx):
        idx_mask = batch["input_ids"][idx] > 102

        mask_f = lambda x: torch.masked_select(x, idx_mask)

        selected_tokens = tokenizer.convert_ids_to_tokens(
            mask_f(batch["input_ids"][idx]).detach().cpu().numpy()
        )

        selected_labels = mask_f(batch["labels"][idx]).detach().cpu()
        selected_labels = torch.where(
            selected_labels == -100, torch.tensor(0), selected_labels
        )
        selected_labels = (
            torch.nn.functional.one_hot(selected_labels, num_classes=3)
            .detach()
            .cpu()
            .numpy()
        )

        selected_logits = outputs.logits[idx]
        mask_f = lambda x: torch.masked_select(x, idx_mask.unsqueeze(-1)).view(-1, 3)
        selected_preds = (
            torch.nn.functional.softmax(mask_f(outputs.logits[idx]))
            .detach()
            .cpu()
            .numpy()
        )
        return selected_tokens, selected_labels, selected_preds

    def train(
        self,
        repository: Repository,
        config: Dict[str, Any],
        training_logger: bm.SupportsLogging,
    ) -> None:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model, tokenizer, collator, optimizer, scheduler = self.get_model_objects(
            config, include_optimizer=True
        )

        # test_samples = repository.get_test_data(batch_size=config["batch_size"])
        step = config.get("start_step", 0)
        for epoch in range(config["epochs"]):
            model.train()
            for batch in tqdm(
                repository.get_training_data(
                    batch_size=config["batch_size"],
                    balance_labels=config.get("balance_labels", False),
                ),
                desc=f"Training Epoch {epoch}",
            ):
                batch = prepare_batch(
                    tokenizer, collator, NERModel_pytorch.lbl_to_id, config, batch
                )
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)

                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % config.get("steps_per_eval", 5) == 0:

                    if config["save_model"]:
                        save_path = os.path.join(
                            config["model_path"],
                            training_logger.get_key(),
                        )
                        model.save_pretrained(save_path)
                        tokenizer.save_pretrained(save_path)

                    per_sample_loss = (
                        torch.nn.functional.cross_entropy(
                            outputs.logits.view(-1, outputs.logits.size(-1)),
                            batch["labels"].view(-1),
                            reduce=False,
                            ignore_index=-100,
                        )
                        .view(batch["labels"].size())
                        .mean(dim=1)
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    # print(per_sample_loss.shape)
                    best, worst = np.argmin(per_sample_loss), np.argmax(per_sample_loss)

                    # continue here with logging

                    # idx_mask = batch["labels"][best] != -100
                    # best_tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][best,...].numpy())
                    # best_labels = torch.nn.functional.one_hot(batch["labels"][best]).numpy()
                    # best_preds = torch.nn.functional.softmax(outputs.logits[best,...]).numpy()

                    best_tokens, best_labels, best_preds = self.filter_by_idx(
                        tokenizer, batch, outputs, best
                    )
                    worst_tokens, worst_labels, worst_preds = self.filter_by_idx(
                        tokenizer, batch, outputs, worst
                    )

                    with training_logger.train():
                        training_logger.log_metric(
                            "loss", loss.detach().cpu().numpy(), step=step
                        )

                        training_logger.log_metric(
                            "learning_rate",
                            # scheduler.get_last_lr()[0],
                            optimizer.state_dict()["param_groups"][0]["lr"],
                            step=step,
                        )

                        training_logger.log_figure(
                            "best_sample_from_batch",
                            color_text_figure(best_tokens, best_labels, best_preds),
                            step=step,
                        )

                        training_logger.log_figure(
                            "worst_sample_from_batch",
                            color_text_figure(worst_tokens, worst_labels, worst_preds),
                            step=step,
                        )

                    model.eval()
                    total_loss, total_n = 0, 0
                    for i, batch in enumerate(
                        tqdm(
                            repository.get_validation_data(
                                batch_size=config["batch_size"]
                            ),
                            desc=f"Testing Epoch {epoch}",
                        )
                    ):
                        batch = prepare_batch(
                            tokenizer,
                            collator,
                            NERModel_pytorch.lbl_to_id,
                            config,
                            batch,
                        )
                        batch = {k: v.to(device) for k, v in batch.items()}
                        outputs = model(**batch)
                        loss = outputs.loss
                        total_loss += loss.item() * len(batch["input_ids"])
                        total_n += len(batch["input_ids"])

                        if i == config.get("eval_n_test_batches", 10):
                            break

                    per_sample_loss = (
                        torch.nn.functional.cross_entropy(
                            outputs.logits.view(-1, outputs.logits.size(-1)),
                            batch["labels"].view(-1),
                            reduce=False,
                            ignore_index=-100,
                        )
                        .view(batch["labels"].size())
                        .mean(dim=1)
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    best, worst = np.argmin(per_sample_loss), np.argmax(per_sample_loss)

                    best_tokens, best_labels, best_preds = self.filter_by_idx(
                        tokenizer, batch, outputs, best
                    )
                    worst_tokens, worst_labels, worst_preds = self.filter_by_idx(
                        tokenizer, batch, outputs, worst
                    )

                    with training_logger.test():
                        training_logger.log_metric(
                            "loss", total_loss / total_n, step=step
                        )

                        training_logger.log_figure(
                            "best_sample_from_batch",
                            color_text_figure(best_tokens, best_labels, best_preds),
                            step=step,
                        )

                        training_logger.log_figure(
                            "worst_sample_from_batch",
                            color_text_figure(worst_tokens, worst_labels, worst_preds),
                            step=step,
                        )

                # the balanced version of the dataset is quite large so offer
                # the option to get out of the epoch early
                if step >= config.get("steps_per_epoch", np.inf):
                    break


if __name__ == "__main__":
    bm.train = train
    bm.inference = inference
    bm.main()

    # import src.data.kaggle_repository as kr
    # import src.evaluate.model as em

    # class MockRepo:
    #     def __init__(self, df):
    #         self.df = df
    #     def get_validation_data(self):
    #         return self.df
    #     def copy(self):
    #         return MockRepo(self.df.copy())

    # config = {
    #     "accum_for": 1,
    #     "max_cores": 24,
    #     "max_seq_len": 512,
    #     "use_amp": True,
    #     "learning_rate": 1e-5,
    #     "model_path": "baseline",
    #     "batch_size": 16,
    #     "save_model": True,

    #     "epochs": 5,
    #     "model_tokenizer_name": "distilbert-base-cased",
    #     "tokenizer_call_kwargs": {
    #         "max_length":512,
    #         "truncation":True,
    #         "is_split_into_words": True
    #     },
    #     "model_kwargs": {
    #     },
    #     "optimizer": "torch.optim.Adam",
    #     "optimizer_kwargs": {
    #     }
    # }

    # print("getting")
    # repo = MockRepo(next(kr.KaggleRepository().get_validation_data(batch_size=32)))
    # print("data retieved")
    # outs = em.evaluate_model(
    #     repo.copy(),
    #     NERModel_pytorch(),
    #     config,
    # )

    # repository = resolve_repo("snippet-ner")
    # config = {
    #     "accum_for": 1,
    #     "max_cores": 24,
    #     "max_seq_len": 128,
    #     "use_amp": True,
    #     "learning_rate": 1e-5,
    #     "model_path": "baseline",
    #     "batch_size": 16,
    #     "save_model": True,

    #     "epochs": 5,
    #     "model_tokenizer_name": "distilbert-base-cased",
    #     "tokenizer_kwargs": {
    #         "do_lower_case": False
    #     },
    #     "model_kwargs": {
    #     },
    #     "optimizer": "torch.optim.Adam",
    #     "optimizer_kwargs": {
    #     }
    # }
    # from comet_ml import Experiment

    # training_logger = Experiment(
    #     workspace="democratizingdata",
    #     project_name="ner-model",
    #     auto_metric_logging=False,
    #     disabled=True,
    # )

    # model = NERModel_pytorch()
    # model.train(repository, config, training_logger)
