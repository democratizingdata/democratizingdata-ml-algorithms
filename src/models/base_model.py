import json
from typing import Any, Dict, Optional, Protocol

import matplotlib.pyplot as plt
import pandas as pd
from src.data.entity_repository import EntityRepository

from src.data.repository import Repository

import click
from src.data.kaggle_repository import KaggleRepository

VALID_REPOS = ["kaggle"]
REPO_HELP_TEXT = f"REPO indicates repository to use, valid options are: {','.join(VALID_REPOS)}"
CONFIG_HELP_TEXT = "--config indicates the json config file to use. If not specified, an empty dictionary will be passed."
NOT_IMPLEMENTED = "You need to implement a function called {0}. It should have the signature: `{0}(repository: Repository, config: Dict[str, Any]) -> None:`"

class SupportsLogging(Protocol):
    def log_metric(self, key: str, value: float) -> None:
        ...

    def log_param(self, key: str, value: Any) -> None:
        ...

    def log_figure(self, key: str, value: plt.Figure) -> None:
        ...

class Model(Protocol):
    def train(self, repository: Repository, config: Dict[str, Any], exp_logger:SupportsLogging) -> None:
        ...

    def inference(
        self, config: Dict[str, Any], df: pd.DataFrame
    ) -> pd.DataFrame:
        ...

def train(repository: Repository, config: Dict[str, Any], training_logger: Optional[SupportsLogging]=None) -> None:
    raise NotImplementedError(NOT_IMPLEMENTED.format("train"))

def validate(repository: Repository, config: Dict[str, Any]) -> None:
    raise NotImplementedError(NOT_IMPLEMENTED.format("validate"))

def resolve_repo(repo_name: str) -> Repository:
    if repo_name == "kaggle":
        return KaggleRepository()
    elif repo_name == "entity":
        return EntityRepository()
    else:
        raise ValueError(f"Unknown repository: {repo_name}")

def main() -> None:
    @click.group()
    def cli() -> None:
        pass

    @cli.command(name="train", help=REPO_HELP_TEXT)
    @click.argument("repo", default="kaggle")
    @click.option("--config", default="", help=CONFIG_HELP_TEXT)
    @click.option("--comet_project", default="", help="Comet project name")
    def _train(repo: str, config: Dict[str, Any]) -> None:
        repository = resolve_repo(repo)
        with open(config) as f:
            config_dict = json.load(f)
        train(repository, config_dict)

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