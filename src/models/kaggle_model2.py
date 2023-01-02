# Model inference notebook:
# https://github.com/Coleridge-Initiative/rc-kaggle-models/blob/original_submissions/2nd%20Chun%20Ming%20Lee/2nd-place-coleridge-inference-code.ipynb
# Model training script:
# https://github.com/Coleridge-Initiative/rc-kaggle-models/blob/original_submissions/2nd%20Chun%20Ming%20Lee/label_classifier.py
# Original labels location:
# https://github.com/Coleridge-Initiative/rc-kaggle-models/blob/original_submissions/2nd%20Chun%20Ming%20Lee/roberta-annotate-abbr.csv

# This model is a baseline model that uses the original labels from the Kaggle
# competition. It uses pytorch and the transformers library.
import dataclasses as dc
import logging
from random import shuffle
from typing import Any, Dict

import pandas as pd
import torch

# from apex import amp
# from apex.optimizers import FusedAdam
from tqdm import trange
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer  # type: ignore # TODO: figure out if we want to stub ALL of transformers...

from src.data.repository import Repository
from src.models.base_model import Hyperparameters, Model

logger = logging.getLogger("KaggleModel2")


@dc.dataclass
class Model2Hyperparameters(Hyperparameters):
    accum_for: int = 1
    max_cores: int = 24
    max_seq_len: int = 128
    use_amp: bool = True


class KaggleModel2(Model):
    def train(self, repository: Repository, config: Model2Hyperparameters) -> None:  # type: ignore[override]
        pretrained_config = AutoConfig.from_pretrained(
            config.pretrained_model, num_labels=2
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            config.pretrained_model, config=pretrained_config
        )

        tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)

        opt = config.optimizer(model.parameters(), lr=config.learning_rate)

        # get all samples
        train_samples = next(repository.get_training_data_dataframe(1000000))
        test_samples = next(repository.get_test_data_dataframe(1000000))

        for epoch in trange(config.num_epochs):
            self._train_epoch(model, tokenizer, train_samples, opt, self.logger, epoch)

            self._test_epoch(model, tokenizer, test_samples, self.logger, epoch)
            if config.save_model:
                model.save_pretrained(config.model_path)

    def _train_epoch(self, model, tokenizer, samples, opt, logger, curr_epoch) -> None:

        all_strings, all_labels = samples["long"].values, samples["is_dataset"].values
        train_indices = list(range(len(all_labels)))
        shuffle(train_indices)
        train_strings = all_strings[train_indices]
        train_labels = all_labels[train_indices]

        model.train()
        iter = 0
        accum_for = config.accum_for
        running_total_loss = 0  # Display running average of loss across epoch
        with trange(
            0,
            len(train_indices),
            config.batch_size,
            desc="Epoch {}".format(curr_epoch),
        ) as t:
            for batch_idx_start in t:
                iter += 1
                batch_idx_end = min(
                    batch_idx_start + config.batch_size, len(train_indices)
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
                loss = loss / config.accum_for  # Normalize if we're doing GA

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

    def _test_epoch(self, model, tokenizer, samples, logger, curr_epoch) -> None:
        test_strings, test_labels = samples["long"].values, samples["is_dataset"].values

        model.eval()
        iter = 0
        running_total_loss = 0

        with trange(
            0,
            len(test_strings),
            config.batch_size,
            desc="Epoch {}".format(curr_epoch),
        ) as t, torch.no_grad():
            for batch_idx_start in t:
                iter += 1
                batch_idx_end = min(
                    batch_idx_start + config.batch_size, len(test_strings)
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
                loss = loss / config.accum_for  # Normalize if we're doing GA

                if torch.cuda.is_available():
                    running_total_loss += loss.detach().cpu().numpy()
                else:
                    running_total_loss += loss.detach().numpy()

                t.set_postfix(loss=running_total_loss / iter)

                if iter % config["accum_for"] == 0:
                    logger.log_metric()

    def inference_string(self, config: Dict[str, Any], text: str) -> str:
        raise NotImplementedError()

    def inference_dataframe(
        self, config: Dict[str, Any], df: pd.DataFrame
    ) -> pd.DataFrame:
        raise NotImplementedError()


if __name__ == "__main__":
    config = Model2Hyperparameters(
        use_amp=True,
        pretrained_model="roberta-base",
        save_model=True,
        model_path="models/kaggle_model2/baseline",
        max_cores=24,
        max_seq_len=128,
        num_epochs=5,
        batch_size=32,
        accum_for=1,
        learning_rate=1e-5,
        optimizer=torch.optim.Adam,
    )

    from src.data.entity_repository import EntityRepository

    logging.basicConfig(level=logging.INFO)
    model = KaggleModel2()
    model.train(EntityRepository(), config)
