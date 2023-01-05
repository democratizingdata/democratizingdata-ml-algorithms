import json
from typing import Any, Dict

import click
import pandas as pd

from src.data.repository import Repository
from src.data.kaggle_repository import KaggleRepository

VALID_REPOS = ["kaggle"]
REPO_HELP_TEXT = f"REPO indicates repository to use, valid options are: {','.join(VALID_REPOS)}"
CONFIG_HELP_TEXT = "--config indicates the json config file to use. If not specified, an empty dictionary will be passed."
NOT_IMPLEMENTED = "You need to implement a function called {0}. It should have the signature: `{0}(repository: Repository, config: Dict[str, Any]) -> None:`"


def train(repository: Repository, config: Dict[str, Any]) -> None:
    raise NotImplementedError(NOT_IMPLEMENTED.format("train"))


def validate(repository: Repository, config: Dict[str, Any]) -> pd.DataFrame:
    raise NotImplementedError(NOT_IMPLEMENTED.format("validate"))


def resolve_repo(repo_name: str) -> Repository:
    if repo_name == "kaggle":
        return KaggleRepository()
    elif repo_name == "s3":
        return None
    else:
        raise ValueError(f"Unknown repository: {repo_name}")


def main() -> None:
    @click.group()
    def cli() -> None:
        pass

    @cli.command(name="train", help=REPO_HELP_TEXT)
    @click.argument("repo", default="kaggle")
    @click.option("--config", default="", help=CONFIG_HELP_TEXT)
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
        validate(repository, config)

    cli()


if __name__ == "__main__":
    main()
