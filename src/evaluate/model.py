from typing import Dict, List

import pandas as pd
from thefuzz import process

from src.models.base_model import Model


def calculate_statistics(row: pd.DataFrame) -> Dict[str, List[str]]:
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

    # Assume all labels are false negatives, then upate the ones that are
    # true positives.
    label_values = ["FN" for i in range(len(labels))]
    false_positives = []

    # str.split() will return an string  with a single empty string so we need
    # to check the first entry.
    if len(predictions[0]) > 0:
        for prediction in predictions:
            lbl_match = process.extractOne(prediction, labels, score_cutoff=50)
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
    model: Model, config: Dict[str, str], batch_size: int
) -> pd.DataFrame:
    """Evaluate a model on the validation set.

    Args:
        model (BaseModel): Model to evaluate.
        config (Dict[str, str]): Model Configuration.
        batch_size (int): Batch size.

    Returns:
        None
    """

    raise NotImplementedError()
