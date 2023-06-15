# Model 3 uses simple string matching, this uses the same keywords as the third
# place submission, but rather than string searching it uses regex.

import json
import logging
from typing import Any, Dict, Optional

import democratizing_data_ml_algorithms.evaluate.model as em
from democratizing_data_ml_algorithms.data.repository import Repository
from democratizing_data_ml_algorithms.models.base_model import SupportsLogging
from democratizing_data_ml_algorithms.models.regex_model import RegexModel

logger = logging.getLogger("RegexModel")


def validate_config(config: Dict[str, Any]) -> None:

    expected_keys = ["eval_path", "keywords"]

    for key in expected_keys:
        assert key in config, f"Missing key {key} in config"


def train(
    repository: Repository,
    config: Dict[str, Any],
    training_logger: Optional[SupportsLogging] = None,
) -> None:
    raise NotImplementedError("RegexModel does not support training")


def validate(repository: Repository, config: Dict[str, Any] = dict()) -> None:
    with open(config["model_path"], "r") as f:
        config["keywords"] = [l.strip() for l in f.readlines()]

    model = Kaggle3RegexInference(config)
    model_evaluation = em.evaluate_model(repository, model, config)

    print(model_evaluation)

    logger.info(f"Saving evaluation to {config['eval_path']}")
    with open(config["eval_path"], "w") as f:
        json.dump(model_evaluation.to_json(), f)


class Kaggle3RegexInference(RegexModel):
    def __init__(self, config: Dict[str, Any]):
        with open(config["model_path"], "r") as f:
            config["keywords"] = [l.strip() for l in f.readlines()]

        config["regex_pattern"] = ""
        super().__init__(config)
