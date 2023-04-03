# Model inference notebook:
# https://github.com/Coleridge-Initiative/rc-kaggle-models/blob/original_submissions/2nd%20Chun%20Ming%20Lee/2nd-place-coleridge-inference-code.ipynb
# Model training script:
# https://github.com/Coleridge-Initiative/rc-kaggle-models/blob/original_submissions/2nd%20Chun%20Ming%20Lee/label_classifier.py
# Original labels location:
# https://github.com/Coleridge-Initiative/rc-kaggle-models/blob/original_submissions/2nd%20Chun%20Ming%20Lee/roberta-annotate-abbr.csv

# This model is a baseline model that uses the original labels from the Kaggle
# competition. It uses pytorch and the transformers library.
import json
import logging
import os
from itertools import islice
from random import shuffle
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

# from apex import amp
# from apex.optimizers import FusedAdam
from tqdm import trange
from scipy.special import expit, softmax
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer  # type: ignore # TODO: figure out if we want to stub ALL of transformers...

from democratizing_data_ml_algorithms.data.repository import Repository
import democratizing_data_ml_algorithms.models.base_model as bm
import democratizing_data_ml_algorithms.evaluate.model as em
import democratizing_data_ml_algorithms.models.schwartz_hearst_model as shm

logger = logging.getLogger("KaggleModel2")


def validate_config(config: Dict[str, Any]) -> None:

    expected_keys = [
        "accum_for",
        "max_cores",
        "max_seq_len",
        "use_amp",
        "pretrained_model",
        "learning_rate",
        "num_epochs",
        "optimizer",
        "model_path",
        "batch_size",
        "save_model",
    ]

    for key in expected_keys:
        assert key in config, f"Missing key {key} in config"


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
    validate_config(config)
    training_logger.log_parameters(config)
    model = KaggleModel2()
    model.train(repository, config, training_logger)


def validate(repository: Repository, config: Dict[str, Any]) -> None:
    """Validates the model and saves the results to config['model_path']

    Args:
        repository (Repository): Repository object
        config (Dict[str, Any]): Configuration dictionary

    Returns:
        None
    """
    validate_config(config)

    model = KaggleModel2()
    model_evaluation = em.evaluate_model(repository, model, config)

    print(model_evaluation)

    logger.info(f"Saving evaluation to {config['eval_path']}")
    with open(config["eval_path"], "w") as f:
        json.dump(model_evaluation.to_json(), f)


# https://stackoverflow.com/a/62913856/2691018
def batcher(iterable, batch_size):
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

        # if torch.cuda.is_available():
        #     model = AutoModelForSequenceClassification.from_pretrained(
        #         config["pretrained_model"], config=pretrained_config
        #     ).cuda()
        # else:
        #     model = AutoModelForSequenceClassification.from_pretrained(
        #         config["pretrained_model"], config=pretrained_config
        #     )

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

        opt = eval(config["optimizer"])(model.parameters(), lr=config["learning_rate"])
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

            # self._test_epoch(config, model, tokenizer, test_samples, training_logger, epoch)
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
        accum_for = config["accum_for"]
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

                # if config["use_amp"]:
                #     with amp.scale_loss(loss, opt) as scaled_loss:
                #         scaled_loss.backward()
                # else:
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


if __name__ == "__main__":
    bm.train = train
    bm.validate = validate
    bm.main()
