import dataclasses as dc
import json
from typing import Any, Dict

import pandas as pd

from src.data.repository import Repository


import click
from src.data.kaggle_repository import KaggleRepository

VALID_REPOS = ["kaggle"]
REPO_HELP_TEXT = f"REPO indicates repository to use, valid options are: {','.join(VALID_REPOS)}"
CONFIG_HELP_TEXT = "--config indicates the json config file to use. If not specified, an empty dictionary will be passed."

class Model:
    def train(self, repository: Repository, config: Dict[str, Any]) -> None:
        raise NotImplementedError()

    def inference(
        self, config: Dict[str, Any], df: pd.DataFrame
    ) -> pd.DataFrame:
        raise NotImplementedError()
