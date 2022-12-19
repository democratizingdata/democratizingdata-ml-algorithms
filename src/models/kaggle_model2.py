# Model inference notebook:
# https://github.com/Coleridge-Initiative/rc-kaggle-models/blob/original_submissions/2nd%20Chun%20Ming%20Lee/2nd-place-coleridge-inference-code.ipynb
# Model training script:
# https://github.com/Coleridge-Initiative/rc-kaggle-models/blob/original_submissions/2nd%20Chun%20Ming%20Lee/label_classifier.py
# Original labels location:
# https://github.com/Coleridge-Initiative/rc-kaggle-models/blob/original_submissions/2nd%20Chun%20Ming%20Lee/roberta-annotate-abbr.csv

# This model is a baseline model that uses the original labels from the Kaggle 
# competition. It uses pytorch and the transformers library.

from random import shuffle
from typing import Any, Dict

import pandas as pd
import torch
# from apex import amp
# from apex.optimizers import FusedAdam
from tqdm import trange
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer)

from src.data.repository import Repository
from src.models.base_model import Model


class KaggleModel2(Model):
    def train(self, repository: Repository, config: Dict[str, Any]) -> None:
        pretrained_config = AutoConfig.from_pretrained(
            config["pretrained_model"], 
            num_labels=2
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            config["pretrained_model"], 
            config=pretrained_config
        )

        opt = config["optimizer"]

        # get all samples
        all_strings, all_labels = next(repository.get_training_data_raw(1000000))
        train_indices = list(range(len(all_labels)))
        shuffle(train_indices)
        train_strings = all_strings[train_indices]
        train_labels = all_labels[train_indices]
        
        model.train()
        iter = 0
        running_total_loss = 0
        



    def inference_string(self, config: Dict[str, Any], text: str) -> str:
        raise NotImplementedError()

    def inference_dataframe(
        self, config: Dict[str, Any], df: pd.DataFrame
    ) -> pd.DataFrame:
        raise NotImplementedError()


if __name__=="__main__":
    config = dict(
        use_amp = True,
        pretrained_model = "roberta-base",
        save_model = True,
        model_path = "../models/kaggle_model2/baseline",
        max_cores = 24,
        max_seq_len = 128,
        num_epochs = 5,
        batch_size = 32,
        accum_for = 1,
        lr = 1e-5,
    
    )
    config["optimizer"] = torch.optim.Adam(lr=config["lr"])

    from src.data.entity_repository import EntityRepository    
    model = KaggleModel2()
    model.train(EntityRepository(), config)
