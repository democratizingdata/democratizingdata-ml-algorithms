import dataclasses as dc
import glob
from itertools import filterfalse
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import spacy
import tensorflow as tf
import transformers as tfs

from src.data.repository import Repository
import src.models.base_model as bm

from src.models.kaggle_model1_support.model_1_QueryDataLoader import QueryDataLoader
from src.models.kaggle_model1_support.model_1_SupportQueryDataLoader import SupportQueryDataLoader
from src.models.kaggle_model1_support.model_1_MetricLearningModel_static import MetricLearningModel

def train(
    repository: Repository,
    config: Dict[str, Any],
    training_logger: Optional[bm.SupportsLogging] = None,
) -> None:
    pass


def validate(repository: Repository, config: Dict[str, Any]) -> None:
    pass



class KaggleModel1(bm.Model):
    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_sm")

    def train(
        self,
        repository: Repository,
        config: Dict[str, Any],
        exp_logger: bm.SupportsLogging,
    ) -> None:
        raise NotImplementedError("This class only runs pretrained models from the competition.")


    def sentencize_text(self, text: str) -> List[str]:
        max_tokens = 10_000

        tokens = text.split()
        if len(tokens) > max_tokens:
            texts = [
                " ".join(tokens[i : i + max_tokens])
                for i in range(0, len(tokens), max_tokens)
            ]
            tokens = tokens[:max_tokens]
        else:
            texts = [text]

        process_generator = self.nlp.pipe(
            texts,
            disable=["lemmatizer", "ner", "textcat"],
        )

        sents = []
        for doc in process_generator:
            sents.extend([sent.text for sent in doc.sents])

        return sents


    def get_support_mask_embed(self, path:str, n_samples:int) -> np.ndarray:

        support_tokens = np.load(path) #[n, embed_dim]

        # print("getting", n_samples, "from", path, support_tokens.shape)

        # sample_idxs = np.random.choice(
        #     np.arange(support_tokens.shape[0]),
        #     n_samples
        # )

        # support_tokens = np.mean(
        #     support_tokens[sample_idxs, :], # [n, embed_dim]
        #     axis=0,
        #     keepdims=True,
        # )[np.newaxis, ...].astype(np.float32) # [1, 1, embed_dim]


        return support_tokens


    def inference(self, config: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:

        model_config = tfs.AutoConfig.from_pretrained(config["model_tokenizer_name"])
        model_config.output_attentions = True
        model_config.output_hidden_states = True

        ll_model = tfs.TFAutoModel.from_config(config=model_config)
        print(ll_model)
        model = MetricLearningModel(config=model_config, name="metric_learning_model")
        model.main_model = ll_model
        model.K = 3

        tokenizer = tfs.AutoTokenizer.from_pretrained(config["model_tokenizer_name"])
        print(type(tokenizer))
        mask_embedding = self.get_support_mask_embed(
            config["support_mask_embedding_path"],
            config["n_support_samples"],
        ) # [n,     embed_dim]

        no_mask_embedding = self.get_support_mask_embed(
            config["support_no_mask_embedding_path"],
            config["n_support_samples"],
        ) # [n,     embed_dim]

        query_dl = QueryDataLoader(df, batch_size=config["batch_size"])
        test_dataloader = SupportQueryDataLoader(
            df,
            tokenizer=tokenizer,
            batch_size=config["batch_size"],
            is_train=False,
            training_steps=len(query_dl),
            query_dataloader=query_dl,
            return_query_ids=True,
        )
        query_batch = test_dataloader.__getitem__(0)
        print(query_batch["input_ids"].shape, query_batch["attention_mask"].shape)
        _ = model(
            [
                query_batch["input_ids"][:1, ...],
                query_batch["attention_mask"][:1, ...],
            ],
            training=True,
            sequence_labels=None,
            mask_embeddings=mask_embedding[:1, :],
            nomask_embeddings=no_mask_embedding[:1, :],
        )

        weights_path = glob.glob(os.path.join(config["weights_path"], "*.h5"))[0]
        model.load_weights(weights_path, by_name=True)
        # model = tf.function(model, experimental_relax_shapes=True)


        def infer_sample(text: str) -> str:
            sents = self.sentencize_text(text)

            for batch in spacy.util.minibatch(sents, config["batch_size"]):
                tokenized_batch = tokenizer(
                    batch,
                    return_tensors="tf",
                    max_length=512,
                    truncation=True,
                    padding=True,
                )

                batch_mask_embeddings = mask_embedding[
                    np.random.choice(
                        np.arange(mask_embedding.shape[0]),
                        config["n_support_samples"],
                    ),
                    ...
                ].mean(axis=0, keepdims=True)[np.newaxis, ...] # [1, 1, embed_dim]

                batch_no_mask_embeddings = no_mask_embedding[
                    np.random.choice(
                        np.arange(no_mask_embedding.shape[0]),
                        config["n_support_samples"],
                    ),
                    ...
                ].mean(axis=0, keepdims=True)[np.newaxis, ...] # [1, 1, embed_dim]

                (
                    query_embeddings,
                    query_mask_embeddings,
                    query_nomask_embeddings,
                    attention_values
                ) = model(
                    [
                        tokenized_batch["input_ids"],
                        tokenized_batch["attention_mask"],
                    ],
                    training=False,
                    sequence_labels=None,
                    mask_embeddings=batch_mask_embeddings,
                    nomask_embeddings=batch_no_mask_embeddings,
                )

                tokens = list(map(
                    tokenizer.convert_ids_to_tokens,
                    tokenized_batch["input_ids"].numpy()
                ))

                for sent, sent_classification in zip(
                    tokens,
                    attention_values.numpy(),
                ):
                    assert len(sent) == len(sent_classification), f"Classification length mismatch {len(sent)} != {len(sent_classification)}"
                    # tokens_classifications = list(filterfalse(
                    #     lambda x: is_special_token(x[0]),
                    #     merge_tokens_w_classifications(
                    #         list(map(clean, sent)),
                    #         list(map(should_merge, sent)),
                    #         sent_classification,
                    #     )
                    # ))

                    # detections = high_probablity_token_groups(
                    #     tokens_classifications,
                    #     threshold=config.get("threshold", 0.9)
                    # ) # List[List[Tuple[str, float]]]

                    # datasets.extend(detections)
                print("query_embeddings", query_embeddings.shape)
                print("query_mask_embeddings", query_mask_embeddings.shape)

        df["model_prediction"] = df["text"].apply(infer_sample)


if __name__ == "__main__":
    # bm.train = train
    # bm.validate = validate
    # bm.main()
    import src.data.kaggle_repository as kr
    import src.evaluate.model as em

    class MockRepo:
        def __init__(self, df):
            self.df = df
        def get_validation_data(self):
            return self.df
        def copy(self):
            return MockRepo(self.df.copy())

    config = dict(
        support_mask_embedding_path = "models/kaggle_model1/sub_biomed_roberta/embeddings/support_mask_embeddings.npy",
        support_no_mask_embedding_path = "models/kaggle_model1/sub_biomed_roberta/embeddings/support_nomask_embeddings.npy",
        n_support_samples = 100,
        model_tokenizer_name = "models/kaggle_model1/sub_biomed_roberta",
        weights_path = "models/kaggle_model1/sub_biomed_roberta/embeddings",
        batch_size = 128,
    )

    model = KaggleModel1()

    repo = MockRepo(next(kr.KaggleRepository().get_validation_data(batch_size=32)))

    outs = em.evaluate_model(
        repo.copy(),
        KaggleModel1(),
        config,
    )
