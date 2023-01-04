from typing import Any, Dict

import click
import pandas as pd

from src.data.repository import Repository

VALID_REPOS = ["kaggle"]
REPO_HELP_TEXT = f"REPO indicates repository to use, valid options are: {','.join(VALID_REPOS)}"
CONFIG_HELP_TEXT = "--config indicates the json config file to use. If not specified, an empty dictionary will be passed."
NOT_IMPLEMENTED = "You need to implement a function called {0}. It should have the signature: `{0}(repository: Repository, config: Dict[str, Any]) -> None:`"



def train(repository: Repository, config: Dict[str, Any]) -> None:
    raise NotImplementedError(NOT_IMPLEMENTED.format("train"))


def infer(repository: Repository, config: Dict[str, Any]) -> pd.DataFrame:
    raise NotImplementedError(NOT_IMPLEMENTED.format("infer"))


def resolve_repo(repo_name: str) -> Repository:
    if repo_name == "kaggle":
        return None
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
        print("made it here", repository, config, repo)
        train(repository, config)

    @cli.command(name="infer", help=REPO_HELP_TEXT)
    @click.argument("repo", default="kaggle")
    @click.option("--config", default="", help=CONFIG_HELP_TEXT)
    def _infer(repo: str, config: Dict[str, Any]) -> None:
        repository = resolve_repo(repo)
        infer(repository, config)

    cli()


if __name__ == "__main__":
    main()
