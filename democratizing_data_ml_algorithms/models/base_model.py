import json
import warnings
from typing import Any, Dict, List, Optional, Protocol, Set, Union

import matplotlib.pyplot as plt
import pandas as pd

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
    """Mock learning rate scheduler for use in protocol/tests"""

    def __init__(self) -> None:
        self.warned = False

    def step(self) -> None:
        if not self.warned:
            self.warned = True
            warnings.warn("Using the mock learning rate scheduler doesn't do anything.")


class SupportsLogging(Protocol):
    """Protocol for logging metrics, parameters, and figures to a logger."""

    def log_metric(self, key: str, value: float) -> None:
        """Logs a metric to the logger.

        Args:
            key (str): Key to log the metric under
            value (float): Value of the metric

        Returns:
            None
        """
        ...

    def log_parameter(self, key: str, value: Any) -> None:
        """Logs a parameter to the logger.

        Args:
            key (str): Key to log the parameter under
            value (Any): Value of the parameter

        Returns:
            None
        """
        ...

    def log_parameters(self, parameters: Dict[str, Any]) -> None:
        """Logs a dictionary of parameters to the logger.

        Args:
            key (str): Key to log the parameters under
            value (Dict[str, Any]): Dictionary of parameters

        Returns:
            None
        """
        ...

    def log_figure(self, key: str, value: plt.Figure) -> None:
        """Logs a figure to the logger.

        Args:
            key (str): Key to log the figure under
            value (plt.Figure): Figure to log

        Returns:
            None
        """
        ...

    def get_key(self) -> str:
        """Returns the unique id associated with this logger/experiment.

        Returns:
            str: Unique id associated with this logger/experiment
        """
        ...


def validate_config(
    expected_keys: Union[Set[str], List[str]], config: Dict[str, Any]
) -> None:
    """Validates a configuration dictionary.

    Args:
        config (Dict[str, Any]): Dictionary of configuration parameters

    Raises:
        AssertionError: If the configuration is invalid
    """

    missing_keys = set(expected_keys) - set(config.keys())
    assert not missing_keys, f"Missing keys: {missing_keys}"


def convert_to_T(T: type, vals: List[str]) -> List["T"]:
    """Converts a list of values to a list of type T.

    Args:
        T (type): Type to convert to
        vals (List[str]): List of values to convert

    Returns:
        List[float]: List of converted values
    """
    return [T(x) for x in vals]


def flatten_hparams_for_logging(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """Flattens a dictionary of hyperparameters for logging to SupportsLogging.

    Args:
        dictionary (Dict[str, Any]): Dictionary of hyperparameters to flatten

    Returns:
        Dict[str, Any]: Flattened dictionary of hyperparameters
    """
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
    """Protocol for a model that can be trained and used for inference."""

    def train(
        self,
        repository: Repository,
        config: Dict[str, Any],
        exp_logger: SupportsLogging,
    ) -> None:
        """Trains a model.

        Args:
            repository (Repository): Repository to use for training
            config (Dict[str, Any]): Dictionary of training configuration
            exp_logger (SupportsLogging): Logger to use for logging metrics,
                                          parameters, and figures

        Returns:
            None
        """

        ...

    def inference(self, config: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
        """Performs inference on a dataframe.

        Args:
            config (Dict[str, Any]): Dictionary of inference configuration
            df (pd.DataFrame): Dataframe to perform inference on

        Returns:
            pd.DataFrame: Dataframe with inference results
        """
        ...


def train(
    repository: Repository,
    config: Dict[str, Any],
    training_logger: Optional[SupportsLogging] = None,
) -> None:
    """Top level function for training a model.

    Args:
        repository (Repository): Repository to use for training
        config (Dict[str, Any]): Dictionary of training configuration
        training_logger (Optional[SupportsLogging], optional): Logger to use for logging metrics,

    Returns:
        None
    """
    raise NotImplementedError(NOT_IMPLEMENTED.format("train"))


def inference(config: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """Top level function for inference.

    Args:
        config (Dict[str, Any]): Dictionary of inference configuration
        df (pd.DataFrame): Dataframe to perform inference on

    Returns:
        pd.DataFrame: Dataframe with inference results
    """
    raise NotImplementedError(NOT_IMPLEMENTED.format("inference"))


def resolve_training_logger(
    comet_workspace: str, comet_project: str
) -> SupportsLogging:
    """Resolves a training logger.

    Args:
        comet_workspace (str): Comet workspace name
        comet_project (str): Comet project name

    Returns:
        SupportsLogging: Training logger
    """

    if comet_project:
        from comet_ml import Experiment

        # if you are having  issues with authenticating, you need to set the the
        # comet api key as an environment variable. export
        # COMET_API_KEY=your_api_key
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
    """Main entrypoint for the cli"""

    @click.group()
    def cli() -> None:
        pass

    @cli.command(name="train")
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

    @cli.command(name="validate")
    @click.argument("repo", default="kaggle", help=REPO_HELP_TEXT)
    @click.option("--config", default="", help=CONFIG_HELP_TEXT)
    def _validate(repo: str, config: Dict[str, Any]) -> None:
        repository = resolve_repo(repo)
        with open(config) as f:
            config_dict = json.load(f)
        validate(repository, config_dict)

    @cli.command(name="inference")
    @click.argument("repo", help=REPO_HELP_TEXT)
    @click.option("--config", help=CONFIG_HELP_TEXT)
    def _inference(repo: str, config: Dict[str, Any]) -> None:
        repository = resolve_repo(repo)
        with open(config) as f:
            config_dict = json.load(f)
        inference(repository, config_dict)

    cli()


if __name__ == "__main__":
    main()
