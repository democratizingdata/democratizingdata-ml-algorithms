import dataclasses as dc
from functools import partial
from itertools import chain
from time import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from thefuzz import fuzz, process
from tqdm import tqdm

from src.data.repository import Repository
from src.data.kaggle_repository import KaggleRepository
from src.models.base_model import Model


@dc.dataclass
class ModelEvaluation:
    output_statistics: pd.DataFrame
    run_time: float
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        try:
            return self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            return 0

    @property
    def recall(self) -> float:
        try:
            return self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            return 0

    def to_json(self) -> Dict[str, Any]:
        json_encoding = dict(
            output_statistics=self.output_statistics.to_json(),
            run_time=self.run_time,
            tp=self.tp,
            fp=self.fp,
            fn=self.fn,
        )
        return json_encoding

    @staticmethod
    def from_json(json_encoding: Dict[str, Any]) -> "ModelEvaluation":
        return ModelEvaluation(
            output_statistics=pd.read_json(json_encoding["output_statistics"]),
            run_time=json_encoding["run_time"],
            tp=json_encoding["tp"],
            fp=json_encoding["fp"],
            fn=json_encoding["fn"],
        )

    def __repr__(self) -> str:
        return f"""
        Model Evaluation:

        - Run time: {self.run_time} seconds, avg: {self.run_time / len(self.output_statistics)} seconds per sample
        - True Postive Count: {self.tp}, avg: {self.tp / len(self.output_statistics)} per sample
        - Precision: {self.precision}
        - Recall: {self.recall}
        """


def retrieve_tpfpfn(
    candidate_list: List[str],
    target_list: List[str],
    scorer: Optional[Callable[[str, str], int]] = None,
    processor: Optional[Callable[[str], str]] = None,
    min_score: int = 90,
    top_n: int = 5,
) -> Tuple[List[str], List[str], List[str]]:
    """Calculate true positives, false positives, and false negatives."""

    true_positives = []
    false_negatives = list(target_list)
    false_positives = list(candidate_list)
    for target in target_list:
        extracted = process.extract(
            target, candidate_list, processor=processor, scorer=scorer, limit=top_n
        )
        for candidate, _ in filter(lambda x: x[1] > min_score, extracted):
            if candidate in false_positives:
                false_positives.remove(candidate)
            if target not in true_positives:
                true_positives.append(target)
            if target in false_negatives:
                false_negatives.remove(target)

    return true_positives, false_positives, false_negatives


def calculate_statistics(
    row: pd.DataFrame,
    scorer: Callable[[str, str], int] = fuzz.partial_ratio,
    processor: Callable[[str], str] = lambda s: s.lower(),
    score_cutoff: int = 90,
) -> Dict[str, List[str]]:
    """Calculate statistics for a row of the validation set.
    This function is meant to be used with pandas.DataFrame.apply()

    Args:
        row (pd.DataFrame): Row of the validation set.
    Returns:
        Dict[str, List[str]]: Dictionary with the containing the keys:
                              - "labels" -> list of the dataset labels
                              - "stats" -> containing "TP", "FP" or "FN" for
                                each label in "labels".
    """
    predictions = list(
        set(filter(lambda x: len(x) > 0, row["model_prediction"].strip().split("|")))
    )
    labels = row["label"].strip().split("|")

    true_positives, false_positives, false_negatives = retrieve_tpfpfn(
        predictions, labels, scorer, processor, score_cutoff
    )

    make_list = lambda s, list_s: [s] * len(list_s)

    output_statistics = {
        "labels": true_positives + false_negatives + false_positives,
        "stats": (
            make_list("TP", true_positives)
            + make_list("FN", false_negatives)
            + make_list("FP", false_positives)
        ),
    }

    return output_statistics


def evaluate_model(
    repository: Repository,
    model: Model,
    config: Dict[str, Any],
    scorer: Callable[[str, str], int] = fuzz.partial_ratio,
    processor: Callable[[str], str] = lambda s: s.lower(),
    min_score: int = 90,
) -> ModelEvaluation:

    validation_dataframe = repository.get_validation_data()

    start = time()
    output = model.inference(config, validation_dataframe)
    total = time() - start

    calc_f = partial(
        calculate_statistics,
        score_cutoff=min_score,
        scorer=scorer,
        processor=processor,
    )

    validation_dataframe["statistics"] = output.apply(calc_f, axis=1)

    all_labels = list(
        chain(
            *list(map(lambda x: x["labels"], validation_dataframe["statistics"].values))
        )
    )
    global_stats = list(
        chain(
            *list(map(lambda x: x["stats"], validation_dataframe["statistics"].values))
        )
    )

    return ModelEvaluation(
        output_statistics=validation_dataframe.loc[:, ["id", "label", "statistics"]],
        run_time=total,
        tp=global_stats.count("TP"),
        fp=global_stats.count("FP"),
        fn=global_stats.count("FN"),
    )


def evaluate_kaggle_private(
    model: Model,
    config: Dict[str, Any],
    scorer: Callable[[str, str], int] = fuzz.partial_ratio,
    processor: Callable[[str], str] = lambda s: s.lower(),
    min_score: int = 90,
) -> ModelEvaluation:
    """Evaluate a model on the validation set.

    Args:
        model (BaseModel): Model to evaluate.
        config (Dict[str, str]): Model Configuration.
        batch_size (int): Batch size.

    Returns:
        ModelEvaluation
    """

    validation_dataframe = KaggleRepository().get_validation_data()

    start = time()
    output = model.inference(config, validation_dataframe)
    total = time() - start

    calc_f = partial(
        calculate_statistics,
        score_cutoff=min_score,
        scorer=scorer,
        processor=processor,
    )

    validation_dataframe["statistics"] = output.apply(calc_f, axis=1)

    all_labels = list(
        chain(*list(map(lambda x: x["labels"], output["statistics"].values)))
    )
    global_stats = list(
        chain(*list(map(lambda x: x["stats"], output["statistics"].values)))
    )

    return ModelEvaluation(
        output_statistics=validation_dataframe.loc[:, ["id", "label", "statistics"]],
        run_time=total,
        tp=global_stats.count("TP"),
        fp=global_stats.count("FP"),
        fn=global_stats.count("FN"),
    )


if __name__ == "__main__":
    from src.models.schwartz_hearst_model import SchwartzHearstModel

    tqdm.pandas()
    evaluation = evaluate_kaggle_private(SchwartzHearstModel(), dict(), 1)

    print(evaluation)
