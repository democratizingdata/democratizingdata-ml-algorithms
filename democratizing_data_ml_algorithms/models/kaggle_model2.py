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

"""This model is an adaptation of the second-place model from the Kaggle competition.

# Model inference notebook:
# https://github.com/Coleridge-Initiative/rc-kaggle-models/blob/original_submissions/2nd%20Chun%20Ming%20Lee/2nd-place-coleridge-inference-code.ipynb
# Model training script:
# https://github.com/Coleridge-Initiative/rc-kaggle-models/blob/original_submissions/2nd%20Chun%20Ming%20Lee/label_classifier.py
# Original labels location:
# https://github.com/Coleridge-Initiative/rc-kaggle-models/blob/original_submissions/2nd%20Chun%20Ming%20Lee/roberta-annotate-abbr.csv

At a high-level the model extracts entities from the text using an entity
extraction algorithm and then classifies them uses a classifier.

Example:

    >>> import pandas as pd
    >>> import democratizing_data_ml_algorithms.models.kaggle_model2 as km2
    >>> df = pd.DataFrame({"text": ["This is a sentence with an entity in it."]})
    >>> config = {
    >>>     "pretrained_model": "path/to/model_and_tokenizer",
    >>> }
    >>> model = km2.KaggleModel2(config)
    >>> df = rm.inference(config, df)

"""
import logging
import os
from itertools import islice
from random import shuffle
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from tqdm import trange

try:
    import torch
    from scipy.special import softmax
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer  # type: ignore # TODO: figure out if we want to stub ALL of transformers...
except ImportError:
    raise ImportError("Running KaggleModel2 requires extras 'kaggle_model2' or 'all'")


from democratizing_data_ml_algorithms.data.repository import Repository
import democratizing_data_ml_algorithms.models.base_model as bm
import democratizing_data_ml_algorithms.evaluate.model as em

logger = logging.getLogger("KaggleModel2")

EXPECTED_KEYS = {
    "pretrained_model",
}


def train(
    repository: Repository,
    config: Dict[str, Any],
    training_logger: Optional[bm.SupportsLogging] = None,
) -> None:
    """Trains the model and saves the results to config['model_path']

    Args:
        repository (Repository): Repository object
        config (Dict[str, Any]): Configuration dictionary
        training_logger: (SupportsLogging, optional): Logging object for model
                                                      training

    Returns:
        None
    """
    bm.validate_config(EXPECTED_KEYS, config)
    training_logger.log_parameters(bm.flatten_hparams_for_logging(config))
    model = KaggleModel2()
    model.train(repository, config, training_logger)



def inference(config: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    pass


# https://stackoverflow.com/a/62913856/2691018
def batcher(iterable:Iterable, batch_size:int):
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


class KaggleModel2(bm.Model):
    def inference(self, config: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:

        df = (
            config["extractor"]
            .inference(config["extractor_config"], df)
            .rename(columns={"model_prediction": "entities"})
        )

        # Load the model
        pretrained_config = AutoConfig.from_pretrained(
            config["pretrained_model"], num_labels=2
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            config["pretrained_model"], config=pretrained_config
        )

        tokenizer = AutoTokenizer.from_pretrained(config["pretrained_model"])
        model.eval()

        def infer_sample(text: str) -> str:
            entities = text.split("|")

            filtered_entities = []
            for batch in batcher(entities, config["batch_size"]):

                batch_features = tokenizer(
                    batch,
                    truncation=True,
                    max_length=64,
                    padding="max_length",
                    add_special_tokens=True,
                    return_tensors="pt",
                )
                model_outputs = model(**batch_features, return_dict=True)
                classifications = (
                    torch.softmax(model_outputs.logits, -1).detach().cpu().numpy()
                )

                filtered_entities.extend(
                    list(
                        map(
                            lambda ent_cls: ent_cls[0],
                            filter(
                                lambda ent_cls: ent_cls[1][1] > config["min_prob"],
                                zip(batch, classifications),
                            ),
                        )
                    )
                )
            return "|".join(filtered_entities)

        df["model_prediction"] = df["entities"].apply(infer_sample)

        return df

    def train(self, repository: Repository, config: Dict[str, Any], training_logger: bm.SupportsLogging) -> None:  # type: ignore[override]
        pretrained_config = AutoConfig.from_pretrained(
            config["pretrained_model"], num_labels=2
        )

        if torch.cuda.is_available():
            model = AutoModelForSequenceClassification.from_pretrained(
                config["pretrained_model"], config=pretrained_config
            ).cuda()
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                config["pretrained_model"], config=pretrained_config
            )

        tokenizer = AutoTokenizer.from_pretrained(config["pretrained_model"])


        if "learning_rate" in config:
            opt = eval(config["optimizer"])(model.parameters(), lr=config["learning_rate"])
        else:
            opt = eval(config["optimizer"])(model.parameters())

        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            opt, lr_lambda=lambda e: 0.95
        )
        # get all samples
        train_samples = repository.get_training_data(
            config.get("balance_labels", False)
        )
        test_samples = repository.get_test_data()

        config["step"] = 0
        for epoch in trange(config["num_epochs"]):
            config["step"] = self._train_epoch(
                config,
                model,
                tokenizer,
                train_samples,
                opt,
                training_logger,
                epoch,
                test_samples,
            )

            scheduler.step()
            if config["save_model"]:
                save_path = os.path.join(
                    config["model_path"],
                    training_logger.get_key(),
                )
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)

    def _train_epoch(
        self,
        config,
        model,
        tokenizer,
        samples,
        opt,
        training_logger,
        curr_epoch,
        test_samples,
    ) -> None:
        all_strings, all_labels = samples["entity"].values, samples["label"].values
        train_indices = list(range(len(all_labels)))
        shuffle(train_indices)
        train_strings = all_strings[train_indices]
        train_labels = all_labels[train_indices]

        iter = 0
        accum_for = config.get("accum_for", 1)
        running_total_loss = 0  # Display running average of loss across epoch
        with trange(
            0,
            len(train_indices),
            config["batch_size"],
            desc="Epoch {}".format(curr_epoch),
        ) as t:
            for batch_idx_start in t:
                model.train()
                config["step"] += 1
                iter += 1
                batch_idx_end = min(
                    batch_idx_start + config["batch_size"], len(train_indices)
                )

                current_batch = list(train_strings[batch_idx_start:batch_idx_end])
                batch_features = tokenizer(
                    current_batch,
                    truncation=True,
                    max_length=64,
                    padding="max_length",
                    add_special_tokens=True,
                    return_tensors="pt",
                )
                batch_labels = torch.tensor(train_labels[batch_idx_start:batch_idx_end])

                if torch.cuda.is_available():
                    # the .long() call is from the original code, but I am not
                    # sure why it is needed. I am not an avid pytorch user.
                    batch_labels = batch_labels.long().cuda()
                    batch_features = {k: v.cuda() for k, v in batch_features.items()}

                model_outputs = model(
                    **batch_features, labels=batch_labels, return_dict=True
                )
                loss = model_outputs["loss"]
                loss = loss / config["accum_for"]  # Normalize if we're doing GA

                loss.backward()

                if torch.cuda.is_available():
                    running_total_loss += loss.detach().cpu().numpy()
                else:
                    running_total_loss += loss.detach().numpy()

                t.set_postfix(loss=running_total_loss / iter)

                if iter % accum_for == 0:
                    opt.step()
                    opt.zero_grad()

                if iter % 10 == 0:
                    with training_logger.train():
                        training_logger.log_metric(
                            "loss",
                            loss.detach().cpu().numpy(),
                            step=config["step"],
                        )
                        y_pred = softmax(
                            model_outputs["logits"].detach().cpu().numpy(), axis=1
                        )
                        y_true = batch_labels.detach().cpu().numpy()

                        matches = np.argmax(y_pred, axis=1) == y_true
                        training_logger.log_metric(
                            "positive_accuracy",
                            matches[y_true].mean(),
                            step=config["step"],
                        )

                        training_logger.log_metric(
                            "negative_accuracy",
                            matches[~y_true].mean(),
                            step=config["step"],
                        )

                    test_strings, test_labels = (
                        test_samples["entity"].values,
                        test_samples["label"].values,
                    )

                    model.eval()
                    iter = 0
                    running_total_loss = 0
                    test_preds = []

                    with trange(
                        0,
                        len(test_strings),
                        config["batch_size"],
                        desc="Epoch {}".format(curr_epoch),
                    ) as t, torch.no_grad():
                        for batch_idx_start in t:
                            iter += 1
                            batch_idx_end = min(
                                batch_idx_start + config["batch_size"],
                                len(test_strings),
                            )

                            current_batch = list(
                                test_strings[batch_idx_start:batch_idx_end]
                            )
                            batch_features = tokenizer(
                                current_batch,
                                truncation=True,
                                max_length=64,
                                padding="max_length",
                                add_special_tokens=True,
                                return_tensors="pt",
                            )
                            batch_labels = torch.tensor(
                                test_labels[batch_idx_start:batch_idx_end]
                            )

                            if torch.cuda.is_available():
                                # the .long() call is from the original code, but I am not
                                # sure why it is needed. I am not an avid pytorch user.
                                batch_labels = batch_labels.long().cuda()
                                batch_features = {
                                    k: v.cuda() for k, v in batch_features.items()
                                }

                            model_outputs = model(
                                **batch_features, labels=batch_labels, return_dict=True
                            )
                            loss = model_outputs["loss"]
                            loss = (
                                loss / config["accum_for"]
                            )  # Normalize if we're doing GA

                            test_preds.append(
                                softmax(
                                    model_outputs["logits"].detach().cpu().numpy(),
                                    axis=1,
                                )
                            )

                            if torch.cuda.is_available():
                                running_total_loss += loss.detach().cpu().numpy()
                            else:
                                running_total_loss += loss.detach().numpy()

                            t.set_postfix(loss=running_total_loss / iter)

                    test_preds = np.concatenate(test_preds, axis=0)
                    test_preds_labels = np.argmax(test_preds, axis=1)

                    with training_logger.test():
                        training_logger.log_metric(
                            "loss",
                            running_total_loss / iter,
                            step=config["step"],
                        )

                        matches = test_preds_labels == test_labels
                        training_logger.log_metric(
                            "positive_accuracy",
                            matches[test_labels].mean(),
                            step=config["step"],
                        )

                        training_logger.log_metric(
                            "negative_accuracy",
                            matches[~test_labels].mean(),
                            step=config["step"],
                        )

                        training_logger.log_confusion_matrix(
                            y_true=test_labels,
                            y_predicted=test_preds_labels,
                            labels=["negative", "positive"],
                            index_to_example_function=lambda idx: test_strings[idx]
                            + " "
                            + str(test_preds[idx, test_preds_labels[idx]]),
                            step=config["step"],
                            title="Test Set Confusion Matrix",
                        )

        return config["step"]

    def _test_epoch(
        self, config, model, tokenizer, samples, training_logger, curr_epoch
    ) -> None:
        test_strings, test_labels = samples["entity"].values, samples["label"].values

        model.eval()
        iter = 0
        running_total_loss = 0
        test_preds = []

        with trange(
            0,
            len(test_strings),
            config["batch_size"],
            desc="Epoch {}".format(curr_epoch),
        ) as t, torch.no_grad():
            for batch_idx_start in t:
                iter += 1
                batch_idx_end = min(
                    batch_idx_start + config["batch_size"], len(test_strings)
                )

                current_batch = list(test_strings[batch_idx_start:batch_idx_end])
                batch_features = tokenizer(
                    current_batch,
                    truncation=True,
                    max_length=64,
                    padding="max_length",
                    add_special_tokens=True,
                    return_tensors="pt",
                )
                batch_labels = torch.tensor(test_labels[batch_idx_start:batch_idx_end])

                if torch.cuda.is_available():
                    # the .long() call is from the original code, but I am not
                    # sure why it is needed. I am not an avid pytorch user.
                    batch_labels = batch_labels.long().cuda()
                    batch_features = {k: v.cuda() for k, v in batch_features.items()}

                model_outputs = model(
                    **batch_features, labels=batch_labels, return_dict=True
                )
                loss = model_outputs["loss"]
                loss = loss / config["accum_for"]  # Normalize if we're doing GA

                test_preds.append(
                    softmax(model_outputs["logits"].detach().cpu().numpy(), axis=1)
                )

                if torch.cuda.is_available():
                    running_total_loss += loss.detach().cpu().numpy()
                else:
                    running_total_loss += loss.detach().numpy()

                t.set_postfix(loss=running_total_loss / iter)

        test_preds = np.concatenate(test_preds, axis=0)
        test_preds_labels = np.argmax(test_preds, axis=1)

        with training_logger.test():
            training_logger.log_metric(
                "loss",
                running_total_loss / iter,
                step=config["step"],
            )

            matches = test_preds_labels == test_labels
            training_logger.log_metric(
                "positive_accuracy",
                matches[test_labels].mean(),
                step=config["step"],
            )

            training_logger.log_metric(
                "negative_accuracy",
                matches[~test_labels].mean(),
                step=config["step"],
            )

            training_logger.log_confusion_matrix(
                y_true=test_labels,
                y_predicted=test_preds_labels,
                labels=["negative", "positive"],
                index_to_example_function=lambda idx: test_strings[idx]
                + " "
                + str(test_preds[idx, test_preds_labels[idx]]),
                step=config["step"],
                title="Test Set Confusion Matrix",
            )


def entry_point():
    bm.train = train
    bm.inference = inference
    bm.main()


if __name__ == "__main__":
    entry_point()
