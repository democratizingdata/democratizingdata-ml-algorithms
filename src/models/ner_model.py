
from itertools import islice
import logging
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
import torch
import transformers as tfs
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer


from src.data.repository import Repository
import src.models.base_model as bm

MODEL_OBJECTS = Tuple[tfs.modeling_utils.PreTrainedModel, tfs.tokenization_utils_base.PreTrainedTokenizer]
MODEL_OBJECTS_WITH_OPTIMIZER = Tuple[tfs.modeling_utils.PreTrainedModel, tfs.tokenization_utils_base.PreTrainedTokenizer, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]

logger = logging.getLogger("ner_model")

def validate_config(config: Dict[str, Any]) -> None:

    expected_keys = {
        "epochs",
        "model_tokenizer_name"
        "tokenizer_kwargs",
        "model_kwargs",
        "optimizer",
        "optimizer_kwargs",
    }

    missing_keys = expected_keys - set(config.keys())
    assert not missing_keys, f"Missing keys: {missing_keys}"

def train(
    repository: Repository,
    config: Dict[str, Any],
    training_logger: Optional[bm.SupportsLogging] = None,
) -> None:

        validate_config(config)
        training_logger.log_parameters(config)


def vaidate(repository: Repository, config: Dict[str, Any]) -> None:

    validate_config(config)


# ==============================================================================
# Preparing the batch for training (https://huggingface.co/docs/transformers/tasks/token_classification#preprocess)
# Mapping all tokens to their corresponding word with the word_ids method.
# Assigning the label -100 to the special tokens [CLS] and [SEP] so they’re ignored by the PyTorch loss function.
# Only labeling the first token of a given word. Assign -100 to other subtokens from the same word.
# ==============================================================================
def prepare_batch(tokenizer: tfs.tokenization_utils_base.PreTrainedTokenizer, batch: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pass



# https://stackoverflow.com/a/62913856/2691018
def batcher(iterable, batch_size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


class NERModel_pytorch(bm.Model):

    def get_model_objects(
        self,
        config: Dict[str, Any],
        include_optimizer: bool = False,
    ) -> Union[MODEL_OBJECTS, MODEL_OBJECTS_WITH_OPTIMIZER]:

        tokenizer = AutoTokenizer.from_pretrained(
            config["model_tokenizer_name"],
            **config["tokenizer_kwargs"],
        )

        pretrained_config = AutoConfig.from_pretrained(
            config["model_tokenizer_name"],
            **config["model_kwargs"],
        )

        model = AutoModelForTokenClassification.from_pretrained(
            config["pretrained_model"], config=pretrained_config
        )
        if torch.cuda.is_available(): model = model.cuda()

        if include_optimizer:
            optimizer = eval(config["optimizer"])(
                model.parameters(),
                **config["optimizer_kwargs"]
            )

            if "scheduler" in config:
                scheduler = eval(config["scheduler"])(optimizer, **config["scheduler_kwargs"])
            else:
                scheduler = bm.MockLRScheduler()

            return model, tokenizer, optimizer, scheduler

        else:
            return model, tokenizer


    def inference(self, config: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
        return super().inference(config, df)




    def train(self, repository: Repository, config: Dict[str, Any], training_logger: bm.SupportsLogging) -> None:

        model, tokenizer, optimizer, scheduler = self.get_model_objects(config, include_optimizer=True)

        train_samples = repository.get_training_data(
            balance_labels = config.get("balance_labels", False),
        )

        test_samples = repository.get_test_data()

        step = 0
        for epoch in range(config["epochs"]):
            model.train()
            for batch in batcher(train_samples, config["batch_size"]):
                batch = prepare_batch(batch, tokenizer)
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                training_logger.log_metrics({"loss": loss.item()}, step=step)

            training_logger.log_metrics(self._evaluate(model, repository, tokenizer), step=step)


