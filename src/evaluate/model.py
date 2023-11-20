import dataclasses as dc
from functools import partial
from itertools import chain, starmap
from time import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from thefuzz import fuzz, process
from tqdm import tqdm

from src.data.repository import Repository
from src.data.kaggle_repository import KaggleRepository
from src.models.base_model import Model

@dc.dataclass
class LabelsStats:
    labels: List[str]
    statistics: List[str]

    def to_json(self) -> Dict[str, Any]:
        return dict(
            labels=self.labels,
            statistics=self.statistics,
        )

    @staticmethod
    def from_json(json_encoding: Dict[str, Any]) -> "LabelsStats":
        return LabelsStats(
            labels=json_encoding["labels"],
            statistics=json_encoding["statistics"],
        )

    def merge(self, other:"LabelsStats") -> "LabelsStats":

        def idx_or_none(lst, item):
            try:
                return lst.index(item)
            except ValueError:
                return None

        def merge_single_statistic(
            self_idx:Union[int, None],
            other_idx:Union[int, None]) -> str:

            self_stat = self.statistics[self_idx] if self_idx is not None else None
            other_stat = other.statistics[other_idx] if other_idx is not None else None



        all_labels = sorted(list(set(self.labels + other.labels)))
        all_statistics = list(starmap(
            merge_single_statistic,
            map(
                lambda lbl: (idx_or_none(self.labels, lbl), idx_or_none(other.labels, lbl)),
                all_labels
            )
        ))




def merge_single_stat(this:str, that:str, lbl:str) -> str:
    if this == "TP" or that == "TP":
        return "TP"
    elif this != that:
        raise ValueError("Inconsistent stats for label: {}. got {} - {}".format(lbl, this, that))
    else:
        return this


def merge_stats(row:pd.DataFrame) -> Dict[str, Any]:
    _id = row["id"]
    self_labels = row["statistics_self"]["labels"]
    other_labels = row["statistics_other"]["labels"]
    self_stats = row["statistics_self"]["stats"]
    other_stats = row["statistics_other"]["stats"]

    merged_lbls = []
    merged_stats = []
    for self_lbl in self_labels:
        self_stat = self_stats[self_labels.index(self_lbl)]
        merged_lbls.append(self_lbl)

        if self_lbl in other_labels:
            merged_stats.append(merge_single_stat(
                self_stat,
                other_stats[other_labels.index(self_lbl)],
                self_lbl,
            ))
        else:
            merged_stats.append(self_stat)

    # ----
    for other_lbl in set(other_labels) - set(self_labels):
        other_stat = other_stats[other_labels.index(other_lbl)]
#         print(other_lbl, other_stat)
        merged_lbls.append(other_lbl)
        merged_stats.append(other_stat)

    return {"id":_id, "statistics": dict(labels=merged_lbls, stats=merged_stats)}


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
        
    @property
    def f1(self) -> float:
        return 2 / (self.recall**-1 + self.precision**-1)

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
        - f1: {self.f1}
        """

    def __or__(self, other:"ModelEvaluation") -> "ModelEvaluation":
        """Combine two model evaluations.

        Args:
            other (ModelEvaluation): Other model evaluation.

        Returns:
            ModelEvaluation: Combined model evaluation.
        """
        if not isinstance(other, ModelEvaluation):
            raise TypeError("Can only combine ModelEvaluation objects.")

        assert np.setdiff1d(
            self.output_statistics.loc[:, ["id"]].values,
            other.output_statistics.loc[:, ["id"]].values
        ).size == 0, "Can only combine ModelEvaluation objects with same documents"

        merged_df = self.output_statistics.loc[:, ["id", "statistics"]].merge(
            other.output_statistics.loc[:, ["id", "statistics"]],
            on="id",
            suffixes=("_self", "_other"),
        ).apply(merge_stats, axis=1, result_type="expand")

        merged_stats = list(
            chain(
                *list(map(lambda x: x["stats"], merged_df["statistics"].values))
            )
        )

        return ModelEvaluation(
            output_statistics=merged_df,
            run_time=self.run_time + other.run_time,
            tp=merged_stats.count("TP"),
            fp=merged_stats.count("FP"),
            fn=merged_stats.count("FN"),
        )


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
    predictions = list(set(filter(
        lambda x: len(x) > 0,
        row["model_prediction"].strip().split("|")
    )))
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
