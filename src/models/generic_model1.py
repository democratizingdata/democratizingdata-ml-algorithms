# This is a token classification model trained with two loss functions:
# 1. Cross entropy loss for the dataset tokens
# 2. Some other loss for comparing embeddings of the context tokens
#
# See https://github.com/Coleridge-Initiative/rc-kaggle-models/blob/main/1st%20ZALO%20FTW/MODEL_SUMMARY.pdf
# for more details.



from itertools import islice
import logging
from typing import Any, Dict, Optional

import pandas as pd
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from src.data.repository import Repository
import src.models.base_model as bm

logger = logging.getLogger("token_classification_model")

def validate_config(config: Dict[str, Any]) -> None:

    expected_keys = {
        "model_tokenizer_name",
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

        training_logger.log_parameters(
            {
                "model_tokenizer_name": config["model_tokenizer_name"],
                "optimizer": config["optimizer"],
            }
            | config["tokenizer_kwargs"]
            | config["model_kwargs"]
            | config["optimizer_kwargs"]
        )



def validate(repository: Repository, config: Dict[str, Any]) -> None:

    validate_config(config)


# https://stackoverflow.com/a/62913856/2691018
def batcher(iterable, batch_size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


class GenericModel1(bm.Model):

    def inference(self, config: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
        return super().inference(config, df)


    def train(self, repository: Repository, config: Dict[str, Any], training_logger: bm.SupportsLogging) -> None:

        tokenizer = AutoTokenizer.from_pretrained(
            config["model_tokenizer_name"],
            **config.get("tokenizer_kwargs", {})
        )

        needed_for_embbedding_comparison = dict(
            output_attentions=True,
            output_hidden_states=True,
        )
        model_config = AutoConfig.from_pretrained(
            config["model_tokenizer_name"],
            needed_for_embbedding_comparison | config.get("model_kwargs", {})
        )

        model = AutoModelForTokenClassification.from_pretrained(model_config)

        optimzer = eval(config["optimizer"])(model.parameters(), config.get("optimizer_kwargs", {}))

        training_iter = repository.get_training_data(
            batch_size=config["batch_size"],
            balance_labels=config.get("balance_labels", False),
        )

        for _ in range(config["epochs"]):
            for batch in training_iter:
                optimzer.zero_grad()
                model_outputs = model(**batch)

                #calculate loss for the dataset tokens



                optimzer.step()




if __name__ == "__main__":
    bm.train = train
    bm.validate = validate
    bm.main()