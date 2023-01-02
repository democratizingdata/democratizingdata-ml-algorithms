import dataclasses as dc
from functools import partial
from itertools import chain
from time import time
from typing import Dict, List

import pandas as pd
from thefuzz import process
from tqdm import tqdm

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
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn)

    def __repr__(self) -> str:
        return f"""
        Model Evaluation:

        - Run time: {self.run_time} seconds, avg: {self.run_time / len(self.output_statistics)} seconds per sample
        - True Postive Count: {self.tp}, avg: {self.tp / len(self.output_statistics)} per sample
        - Precision: {self.precision}
        - Recall: {self.recall}
        """


def calculate_statistics(
    row: pd.DataFrame, score_cutoff: int = 50
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
    predictions = list(set(row["model_prediction"].split("|")))
    labels = row["labels"].split("|")

    label_values = ["FN" for i in range(len(labels))]
    false_positives = []

    if len(predictions[0]) > 0:

        for i, prediction in enumerate(predictions):
            lbl_match = process.extractOne(
                prediction, labels, score_cutoff=score_cutoff
            )
            if lbl_match is not None:
                label_values[labels.index(lbl_match[0])] = "TP"
            else:
                false_positives.append(prediction)

    output_statistics = {
        "labels": labels + false_positives,
        "stats": label_values + ["FP" for _ in range(len(false_positives))],
    }

    return output_statistics


def evaluate_kaggle_private(
    model: Model, config: Dict[str, str], batch_size: int, min_score: int = 50
) -> ModelEvaluation:
    """Evaluate a model on the validation set.

    Args:
        model (BaseModel): Model to evaluate.
        config (Dict[str, str]): Model Configuration.
        batch_size (int): Batch size.

    Returns:
        ModelEvaluation
    """

    batch_size = -1
    validation_dataframe = KaggleRepository().get_validation_dataframe(batch_size)

    start = time()
    output = model.inference_dataframe(config, validation_dataframe)
    total = time() - start

    calc_f = partial(calculate_statistics, score_cutoff=min_score)

    tqdm.pandas()
    validation_dataframe["statistics"] = output.progress_apply(calc_f, axis=1)

    all_labels = list(
        chain(*list(map(lambda x: x["labels"], output["statistics"].values)))
    )
    global_stats = list(
        chain(*list(map(lambda x: x["stats"], output["statistics"].values)))
    )

    return ModelEvaluation(
        output_statistics=validation_dataframe.loc[:, ["ids", "labels", "statistics"]],
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
