import json
from typing import Any, Dict, Optional, Protocol

import matplotlib.pyplot as plt
import pandas as pd
from democratizing_data_ml_algorithms.data.entity_repository import EntityRepository

from democratizing_data_ml_algorithms.data.repository import Repository
from democratizing_data_ml_algorithms.data.repository_resolver import resolve_repo


import click


VALID_REPOS = ["kaggle", "entity", "snippet"]
REPO_HELP_TEXT = (
    f"REPO indicates repository to use, valid options are: {','.join(VALID_REPOS)}"
)
CONFIG_HELP_TEXT = "--config indicates the json config file to use. If not specified, an empty dictionary will be passed."
NOT_IMPLEMENTED = "You need to implement a function called {0}. It should have the signature: `{0}(repository: Repository, config: Dict[str, Any]) -> None:`"


class MockLRScheduler:
    def step(self) -> None:
        ...


class SupportsLogging(Protocol):
    def log_metric(self, key: str, value: float) -> None:
        ...

    def log_parameter(self, key: str, value: Any) -> None:
        ...

    def log_parameters(self, key: str, value: Any) -> None:
        ...

    def log_figure(self, key: str, value: plt.Figure) -> None:
        ...

    def get_key(self) -> str:
        ...


def flatten_hparams_for_logging(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """Flattens a dictionary of hyperparameters for logging to SupportsLogging"""
    flattened = {}
    for key, value in dictionary.items():
        if isinstance(value, dict):
            flattened.update(
                flatten_hparams_for_logging({f"{key}:{k}": v for k, v in value.items()})
            )
        else:
            flattened[key] = value
    return flattened


class Model(Protocol):
    def train(
        self,
        repository: Repository,
        config: Dict[str, Any],
        exp_logger: SupportsLogging,
    ) -> None:
        ...

    def inference(self, config: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
        ...


def train(
    repository: Repository,
    config: Dict[str, Any],
    training_logger: Optional[SupportsLogging] = None,
) -> None:
    raise NotImplementedError(NOT_IMPLEMENTED.format("train"))


def validate(repository: Repository, config: Dict[str, Any]) -> None:
    raise NotImplementedError(NOT_IMPLEMENTED.format("validate"))


def resolve_training_logger(
    comet_workspace: str, comet_project: str
) -> SupportsLogging:

    if comet_project:
        from comet_ml import Experiment

        # if you are issues with authenticating, you need to set the the comet
        # api key as an environment variable.
        # export COMET_API_KEY=your_api_key
        experiment = Experiment(
            workspace=comet_workspace,
            project_name=comet_project,
            auto_metric_logging=False,
            disabled=False,
        )

        return experiment

    else:
        return None


def main() -> None:
    @click.group()
    def cli() -> None:
        pass

    @cli.command(name="train", help=REPO_HELP_TEXT)
    @click.argument("repo", default="kaggle")
    @click.option("--config", default="", help=CONFIG_HELP_TEXT)
    @click.option(
        "--comet_workspace", default="democratizingdata", help="Comet workspace name"
    )
    @click.option("--comet_project", default="", help="Comet project name")
    def _train(
        repo: str, config: Dict[str, Any], comet_workspace: str, comet_project: str
    ) -> None:
        repository = resolve_repo(repo)
        training_logger = resolve_training_logger(comet_workspace, comet_project)
        with open(config) as f:
            config_dict = json.load(f)
        train(repository, config_dict, training_logger)

    @cli.command(name="validate", help=REPO_HELP_TEXT)
    @click.argument("repo", default="kaggle")
    @click.option("--config", default="", help=CONFIG_HELP_TEXT)
    def _validate(repo: str, config: Dict[str, Any]) -> None:
        repository = resolve_repo(repo)
        with open(config) as f:
            config_dict = json.load(f)
        validate(repository, config_dict)

    cli()


if __name__ == "__main__":
    main()
