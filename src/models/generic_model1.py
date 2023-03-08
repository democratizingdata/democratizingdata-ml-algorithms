# This is a token classification model trained with two loss functions:
# 1. Cross entropy loss for the dataset tokens
# 2. Arcface for comparing embeddings of the context tokens
#
# See https://github.com/Coleridge-Initiative/rc-kaggle-models/blob/main/1st%20ZALO%20FTW/MODEL_SUMMARY.pdf
# for more details.


from functools import partial
from itertools import filterfalse, islice, starmap
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import datasets as ds
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
import transformers as tfs
from pytorch_metric_learning import losses as metric_losses
import spacy
from tqdm import tqdm

from datasets.utils.logging import disable_progress_bar

disable_progress_bar()


from src.data.repository import Repository
import src.models.base_model as bm

logger = logging.getLogger("token_classification_model")


def validate_config(config: Dict[str, Any]) -> None:

    expected_keys = {
        "model_tokenizer_name",
        "tokenizer_kwargs",
        "model_kwargs",
        "optimizer",
        "optimizer_kwargs",
    }

    missing_keys = expected_keys - set(config.keys())
    assert not missing_keys, f"Missing keys: {missing_keys}"

    if "save_model" in config:
        assert (
            "model_path" in config
        ), "if save_model is true, you need to provide a path"


def train(
    repository: Repository,
    config: Dict[str, Any],
    training_logger: Optional[bm.SupportsLogging] = None,
) -> None:
    validate_config(config)

    training_logger.log_parameters(bm.flatten_hparams_for_logging(config))

    model = GenericModel1()
    model.train(repository, config, training_logger)


def validate(repository: Repository, config: Dict[str, Any]) -> None:
    validate_config(config)


def convert_to_T(T: type, vals: List[str]) -> List[float]:
    return [T(x) for x in vals]


def tokenize_and_align_labels(
    tokenizer_f: Callable[[Dict[str, Any]], Dict[str, Any]], examples: Dict[str, Any]
) -> Dict[str, Any]:
    tokenized_inputs = tokenizer_f(examples["text"])

    labels = []
    for i, label in enumerate(examples["mask_token_indicator"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100] * len(word_ids)  # assume all tokens are special
        top_word_id = max(map(lambda x: x if x else -1, word_ids))
        for word_idx in range(top_word_id + 1):
            label_ids[word_ids.index(word_idx)] = label[word_idx]
        labels.append(label_ids)

    tokenized_inputs["mask_token_indicator"] = labels
    return tokenized_inputs


def apply_mask_sample(
    tokens: List[str], mask_token_indicator: List[float]
) -> List[str]:

    tokens = list(map(lambda t, m: "[MASK]" if m else t, tokens, mask_token_indicator))
    return tokens


def apply_mask_batched(dataset: Dict[str, Any]) -> Dict[str, Any]:
    # inintially every token is masked, however, we want to group them
    # so that a single token represents an entire dataset
    ungrouped_masks = list(
        starmap(
            apply_mask_sample,
            zip(dataset["text"], dataset["mask"]),
        )
    )

    text_mask = list(
        zip(
            *list(
                starmap(
                    group_mask_sample,
                    zip(ungrouped_masks, dataset["mask"]),
                )
            )
        )
    )

    dataset["text"], dataset["mask"] = list(text_mask[0]), list(text_mask[1])

    return dataset


def group_mask_sample(
    tokens: List[str], mask_token_indicator: List[float]
) -> List[str]:

    # group the masks
    grouped_text_masks = [tokens[0]]
    grouped_mask_token_indicator = [mask_token_indicator[0]]

    for index in range(1, len(tokens)):
        if not (
            mask_token_indicator[index] == 1 and mask_token_indicator[index - 1] == 1
        ):
            grouped_text_masks.append(tokens[index])
            grouped_mask_token_indicator.append(mask_token_indicator[index])

    return grouped_text_masks, grouped_mask_token_indicator


def convert_dataset(
    tokenizer_f: Callable, collator: Callable, dataset: ds.Dataset
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

    convert_f = partial(convert_to_T, int)

    dataset = (
        dataset.map(
            lambda dset: {"mask_token_indicator": list(map(convert_f, dset["mask"]))},
            batched=True,
        )
        .map(
            partial(tokenize_and_align_labels, tokenizer_f),
            batched=True,
        )
        .remove_columns(
            ["text", "mask"]
            # we rename mask_token_indicator to labels, because that is what the
            # so that the data collator will pad it.
        )
        .rename_column("mask_token_indicator", "labels")
        .rename_column("label", "seq_labels")
    )

    # the collator doesn't know what to do with the seq_labels, so we
    # remove it, and then add it back in after the collator is done.
    # The collator also changes our type from Dataset to dict, so we
    # we are now working with a dictionary of tensors.
    tmp_seq_labels = dataset["seq_labels"]
    dataset = collator(list(dataset.remove_columns(["seq_labels"])))

    dataset_inputs = dict(
        input_ids=dataset["input_ids"],
        attention_mask=dataset["attention_mask"],
    )

    mask_token_indicator = torch.where(dataset["labels"] == 1, 1, 0)
    special_token_indicator = torch.where(dataset["labels"] == -100, 1, 0)

    dataset_labels = dict(
        mask_token_indicator=mask_token_indicator,
        special_token_indicator=special_token_indicator,
        seq_labels=torch.Tensor(tmp_seq_labels),
    )
    # At this point, the dataset is a dictionary of tensors, where the
    # tensors are all the same length.
    return dataset_inputs, dataset_labels


def prepare_batch(
    tokenizer: tfs.tokenization_utils_base.PreTrainedTokenizerBase,
    data_collator: tfs.data.data_collator.DataCollatorMixin,
    n_query: int,
    tokenizer_call_kwargs: Dict[str, Any],
    batch: pd.DataFrame,
) -> Tuple[
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
]:

    # we need to split the batch into support and query sets, well use the
    # train/test split functionality from datasets.Dataset to select random
    # subsets of the batch as the query/support sets. where query will be "train"
    if n_query > 1:
        dataset = ds.Dataset.from_pandas(
            batch.drop(columns=["pos_tags"])
        ).class_encode_column("label")

        # it can be the case thatll of the samples come from one class. That will
        # cause an error when we try to split the dataset.
        try:
            dataset = dataset.train_test_split(
                train_size=n_query, stratify_by_column="label"
            )
        except ValueError as e:
            label_count_issues = [
                "minimum number of groups for any class",
                "minimum class count",
                "should be greater or equal to the number of classes",
            ]
            if not any(map(lambda x: x in str(e).lower(), label_count_issues)):
                raise e

            dataset = dataset.train_test_split(train_size=n_query)
    else:
        dataset = ds.Dataset.from_pandas(
            batch.drop(columns=["pos_tags"])
        ).train_test_split(train_size=n_query)

    ds_query = dataset["train"]
    ds_support = dataset["test"].map(apply_mask_batched, batched=True)

    batch_query, labels_query = convert_dataset(
        tokenizer_f=partial(tokenizer, **tokenizer_call_kwargs),
        collator=data_collator,
        dataset=ds_query,
    )

    batch_support, labels_support = convert_dataset(
        tokenizer_f=partial(tokenizer, **tokenizer_call_kwargs),
        collator=data_collator,
        dataset=ds_support,
    )

    return batch_query, batch_support, labels_query, labels_support


def filter_prep_tokens(
    tokens: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[List[str], List[float], List[float]]:
    # we need to filter out special tokens [CLS] and [SEP]

    idxs = list(
        filter(
            lambda i: tokens[i] not in ["[CLS]", "[SEP]", "[PAD]"], range(len(tokens))
        )
    )

    return np.array(tokens)[idxs], y_true[idxs].astype(np.float32), y_pred[idxs]


# based on
# https://stackoverflow.com/questions/36264305/matplotlib-multi-colored-title-text-in-practice
def color_text_figure_binary(tokens, cmap, y_true, y_pred, threshold=0.5):
    f, ax = plt.subplots(figsize=(10, 1))
    r = f.canvas.get_renderer()
    ax.set_axis_off()
    space = 0.025
    w = 0.0
    ax.set_title("Top Predicted, Bottom True, Green = 1, Black = 0")
    for i, (token, y, yh) in enumerate(zip(tokens, y_true, y_pred)):
        t = ax.text(w, 0.25, token, color=cmap(y), ha="left", va="center", fontsize=18)
        ax.text(w, 0.75, token, color=cmap(yh), ha="left", va="center", fontsize=18)

        if y >= threshold:
            ax.text(
                w,
                0.0,
                "{:.2f}".format(y),
                color="black",
                ha="left",
                va="center",
                fontsize=10,
            )
        if yh >= threshold:
            ax.text(
                w,
                1.0,
                "{:.2f}".format(yh),
                color="black",
                ha="left",
                va="center",
                fontsize=10,
            )

        transf = ax.transData.inverted()
        bb = t.get_window_extent(renderer=f.canvas.renderer)
        bb = bb.transformed(transf)
        w = w + bb.xmax - bb.xmin + space
    return f


def masked_mean(
    arr: torch.Tensor,
    keep_mask: torch.Tensor,
    axis: int,
    keepdim: bool = True,
    be_safe: bool = True,
) -> torch.Tensor:
    """"""
    masked_arr = arr * keep_mask
    n = keep_mask.sum(axis=axis, keepdim=keepdim)

    if be_safe:
        n = torch.where(n == 0, torch.ones_like(n, dtype=n.dtype), n)

    return masked_arr.sum(axis=axis, keepdim=keepdim) / n

def merge_tokens_w_classifications(
    tokens:List[str],
    token_should_be_merged:List[bool],
    classifications:List[float]
) -> List[Tuple[str, float]]:
    merged = []
    for token, do_merge, classification in zip(tokens, token_should_be_merged, classifications):
        if do_merge:
            merged[-1] = (merged[-1][0] + token, merged[-1][1])
        else:
            merged.append((token, classification))
    return merged


def is_special_token(token):
    return (
        token.startswith("[") and token.endswith("]")
        or token.startswith("<") and token.endswith(">")
    )



def high_probablity_token_groups(
    tokens_classifications: List[Tuple[str, float]],
    threshold:float=0.9,
) -> List[List[Tuple[str, float]]]:

    datasets = []
    dataset = []
    for token, score in tokens_classifications:
        if score >= threshold:
            dataset.append((token, score))
        else:
            if len(dataset) > 0:
                datasets.append(dataset)
                dataset = []
    if len(dataset) > 0:
        datasets.append(dataset)

    return datasets

class GenericModel1(bm.Model):
    blk_grn = mcolors.LinearSegmentedColormap.from_list(
        "my_colormap",
        ["black", "green"],
    )

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def get_model_objects(
        self, config: Dict[str, Any], include_optimizer: bool
    ) -> None:
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = tfs.AutoTokenizer.from_pretrained(
            config["model_tokenizer_name"], **config.get("tokenizer_kwargs", {})
        )

        collator = tfs.data.data_collator.DataCollatorForTokenClassification(
            tokenizer,
            return_tensors="pt",
            label_pad_token_id=0,
        )

        model = tfs.AutoModel.from_pretrained(
            config["model_tokenizer_name"], **config.get("model_kwargs", {})
        )

        print("Model linear layer size:", model.config.hidden_size)
        linear = torch.nn.Linear(
            model.config.hidden_size,
            model.config.hidden_size,
            bias=True,
        )

        # this might be brittle. We are going to assume that if the model name
        # is a path, then we are loading a model that has already been trained
        # and therefore has the dense layer saved. If it is not a path, then
        # then it is a model that is being pulled from huggingface and we'll
        # keep the random weights
        if os.path.exists(config["model_tokenizer_name"]):
            print(
                "Loading pretrained linear layer from:",
                os.path.join(
                    config["model_tokenizer_name"],
                    "linear.bin"
                )
            )
            linear.load_state_dict(
                torch.load(os.path.join(
                    config["model_tokenizer_name"],
                    "linear.bin"
                )),
                strict=True,
            )

        # model.to(device)
        # linear.to(device)

        if include_optimizer:
            optimizer = eval(config["optimizer"])(
                list(model.parameters()) + list(linear.parameters()),
                **config.get("optimizer_kwargs", {})
            )

            if "scheduler" in config:
                scheduler = eval(config["scheduler"])(
                    optimizer, **config.get("scheduler_kwargs", {})
                )
            else:
                scheduler = bm.MockLRScheduler()

            metric_loss = metric_losses.ArcFaceLoss(
                num_classes=2,
                embedding_size=model.config.hidden_size,
                **config.get("metric_loss_kwargs", {}),
            )

            metric_optimizer = eval(config["metric_optimizer"])(
                metric_loss.parameters(), **config.get("metric_optimizer_kwargs", {})
            )

            if "metric_scheduler" in config:
                metric_scheduler = eval(config["metric_scheduler"])(
                    metric_optimizer, **config.get("metric_scheduler_kwargs", {})
                )
            else:
                metric_scheduler = bm.MockLRScheduler()

            return (
                model,
                linear,
                tokenizer,
                collator,
                optimizer,
                scheduler,
                metric_loss,
                metric_optimizer,
                metric_scheduler,
            )
        else:
            return model, linear, tokenizer, collator

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

        sample_idxs = np.random.choice(
            np.arange(support_tokens.shape[0]),
            n_samples
        )

        support_tokens = np.mean(
            support_tokens[sample_idxs, :], # [n, embed_dim]
            axis=0,
            keepdims=True,
        )[np.newaxis, ...].astype(np.float32) # [1, 1, embed_dim]

        print("getting:", path, support_tokens.shape)

        return np.mean(
            support_tokens,
            axis=0,
            keepdims=True
        )[np.newaxis, ...].astype(np.float32)

    def inference(self, config: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                                             # [batch, token, embed_dim]
        mask_embedding = self.get_support_mask_embed(
            config["support_mask_embedding_path"],
            config["n_support_samples"],
        ) # [1,     1,     embed_dim]

        no_mask_embedding = self.get_support_mask_embed(
            config["support_no_mask_embedding_path"],
            config["n_support_samples"],
        ) # [1,     1,     embed_dim]

        if config.get("is_roberta", False):
            print("Merging tokens based on Roberta tokenizer")
            should_merge = lambda t: not t.startswith("Ġ") and not t.startswith("<")
            clean = lambda t: t.replace("Ġ", "")
        else:
            print("Merging tokens based on BERT tokenizer")
            should_merge = lambda t: t.startswith("##")
            clean = lambda t: t.replace("##", "")


        model, linear, tokenizer, _ = self.get_model_objects(config, include_optimizer=False)

        model.to(device)
        linear.to(device)
        model.eval()
        linear.eval()
        ng = torch.no_grad()
        ng.__enter__()

        masked_embedding = torch.from_numpy(mask_embedding).to(device)
        no_mask_embedding = torch.from_numpy(no_mask_embedding).to(device)
        def infer_sample(text:str) -> str:
            sents = self.sentencize_text(text) # List[List[str]]
            assert len(sents) > 0, "No sentences found in text"

            datasets = []
            for batch in spacy.util.minibatch(sents, config["batch_size"]):
                batch = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    **config.get("tokenizer_call_kwargs", {}),
                )

                batch = batch.to(device)
                outputs = model(**batch)
                output_embedding = linear(outputs.last_hidden_state) # [batch, token, embed_dim]
                cos_sim = torch.nn.functional.cosine_similarity(
                    output_embedding, masked_embedding, dim=-1
                ) # [batch, token]
                token_classification = torch.nn.functional.sigmoid(cos_sim * 10) # [batch, token]

                token_non_classification = 1 - torch.nn.functional.sigmoid(
                    torch.nn.functional.cosine_similarity(
                        output_embedding, no_mask_embedding, dim=-1
                    ) *  10
                ) # [batch, token]

                # Not sure why they do this  ===================================
                # This isn't documented in the paper, but it's in the code
                token_classification = token_classification.cpu().numpy() # [batch, token]
                token_non_classification = token_non_classification.cpu().numpy() # [batch, token]
                merged_classifications = 0.5 * (token_classification + token_non_classification)
                # merged_classifications = token_classification

                merged_classifications = merged_classifications# * batch.attention_mask.cpu().numpy()
                # Not sure why they do this  ===================================


                tokens = list(map(
                    tokenizer.convert_ids_to_tokens,
                    batch["input_ids"].cpu().numpy()
                )) # [batch, token]

                for sent, sent_classification in zip(
                    tokens,
                    merged_classifications
                ):
                    assert len(sent) == len(sent_classification), "Classification length mismatch"
                    tokens_classifications = list(filterfalse(
                        lambda x: is_special_token(x[0]),
                        merge_tokens_w_classifications(
                            list(map(clean, sent)),
                            list(map(should_merge, sent)),
                            sent_classification,
                        )
                    ))

                    detections = high_probablity_token_groups(
                        tokens_classifications,
                        threshold=config.get("threshold", 0.9)
                    ) # List[List[Tuple[str, float]]]

                    datasets.extend(detections)

            return "|".join(list(map(
                lambda x: " ".join(map(lambda y: y[0], x)),
                datasets
            )))


        if config.get("inference_progress_bar", False):
            tqdm.pandas()
            df["model_prediction"] = df["text"].progress_apply(infer_sample)
        else:
            df["model_prediction"] = df["text"].apply(infer_sample)

        ng.__exit__(None, None, None)

        return df

    def train(
        self,
        repository: Repository,
        config: Dict[str, Any],
        training_logger: bm.SupportsLogging,
    ) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        (
            model,
            linear,
            tokenizer,
            collator,
            optimizer,
            scheduler,
            metric_loss,
            metric_optimizer,
            metric_scheduler,
        ) = self.get_model_objects(config, include_optimizer=True)

        model.to(device)
        linear.to(device)

        training_iter = repository.get_training_data(
            batch_size=config["batch_size"],
            balance_labels=config.get("balance_labels", True),
        )

        step = config.get("start_step", 0)
        for epoch in range(config["epochs"]):
            for batch in tqdm(training_iter, desc=f"Training Epoch {epoch}"):
                model.train()
                linear.train()
                metric_loss.train()

                # bs = support batch size
                # bq = query batch size
                # seq_len = sequence length
                # emb_dim = embedding dimension

                (
                    batch_query,
                    batch_support,
                    query_labels,
                    support_labels,
                ) = prepare_batch(
                    tokenizer,
                    collator,
                    config.get("n_query", 1),
                    config.get("tokenizer_call_kwargs", {}),
                    batch,
                )

                send_to_device = lambda d: {k: v.to(device) for k, v in d.items()}
                batch_query = send_to_device(batch_query)
                batch_support = send_to_device(batch_support)
                query_labels = send_to_device(query_labels)
                support_labels = send_to_device(support_labels)

                # These are the snippet level labels
                query_seq_labels = query_labels["seq_labels"]  # [bq, 1]
                support_seq_labels = support_labels["seq_labels"]  # [bs, 1]

                # These indicate which of the support tokens are mask tokens
                support_mask_token_indicator = support_labels[
                    "mask_token_indicator"
                ]  # [bs, seq_len]
                # These indicate which of the query tokens are label tokens
                query_label_token_indicator = query_labels[
                    "mask_token_indicator"
                ]  # [bq, seq_len]

                optimizer.zero_grad()
                metric_optimizer.zero_grad()

                # Variables are named to match the document from the first place
                # solution document, linked above, Section 2.3.3.3. Embeddings extraction

                # First process the support set ================================
                output_support = model(**batch_support)
                support_hidden_states = (
                    output_support.last_hidden_state
                )  # [bs, seq_len, emb_dim]
                # might be support_hidden_states = output_support.hidden_states[-1]
                support_linear_embedding = linear(support_hidden_states)  # [bs, seq_len, emb_dim]

                # the first token is the CLS token which is the support embedding
                support_embedding = support_hidden_states[:, 0, :]  # [bs, emb_dim]

                # ==============================================================

                mask_embedding = masked_mean(
                    support_linear_embedding,
                    support_mask_token_indicator.unsqueeze(-1),
                    axis=1,  # average over sequence length
                    keepdim=False,
                ).mean(
                    axis=0, keepdim=True
                )  # average over batch size, [1, emb_dim]

                non_mask_embedding = masked_mean(
                    support_linear_embedding,
                    (1 - support_mask_token_indicator).unsqueeze(-1),
                    axis=1,  # average over sequence length
                    keepdim=False,
                ).mean(
                    axis=0, keepdim=True
                )  # average over batch size, [1, emb_dim]

                # Second process the query set =================================

                output_query = model(**batch_query)
                query_hidden_states = (
                    output_query.last_hidden_state
                )  # [bq, seq_len, emb_dim]
                # might be query_hidden_states = output_query.hidden_states[-1]
                query_linear_embedding = linear(query_hidden_states)  # [bq, seq_len, emb_dim]

                label_embedding = masked_mean(
                    query_linear_embedding,
                    query_label_token_indicator.unsqueeze(-1),
                    axis=1,  # average over sequence length
                    keepdim=False,
                )  # [bq, emb_dim]

                non_label_embedding = masked_mean(
                    query_linear_embedding,
                    (1 - query_label_token_indicator).unsqueeze(-1)
                    * batch_query["attention_mask"].unsqueeze(-1),
                    axis=1,  # average over sequence length
                    keepdim=False,
                )  # [bq, emb_dim]

                # ==============================================================

                # Third compute the loss =======================================
                # From the document, Section 2.3.3.3. Applying criterions
                # We apply ArcFace loss to two pairs of embeddings:
                #   ● SUPPORT MASK and QUERY LABEL (if available) are considered class 0.
                #   ● SUPPORT non-MASK and QUERY non-LABEL are considered class 1.
                # We also apply ArcFace loss for these embeddings:
                #   ● SUPPORT embeddings are considered class 0
                #   ● Positive QUERY embeddings are considered class 0
                #   ● Negative QUERY embeddings are considered class 1
                #
                # Different from the document, we are going to switch the classes
                # so that the positive class is 1 and the negative class is 0.

                # We want to group the masked tokens from the support set with
                # the positive tokens from the query set. Both of these
                # embeddings are from the same class, they represent dataset
                # tokens.
                class_1_embedding = torch.cat(
                    [mask_embedding, label_embedding], dim=0
                ).mean(
                    dim=0, keepdim=True
                )  # [1, emb_dim]

                # We want to group the non-masked tokens from the support set
                # with the negative tokens from the query set. Both of these
                # embeddings represent non-dataset tokens.
                class_0_embedding = torch.cat(
                    [non_mask_embedding, non_label_embedding], dim=0
                ).mean(
                    dim=0, keepdim=True
                )  # [1, emb_dim]

                class_1_cls_support_embedding = masked_mean(
                    support_embedding,  # [bs, emb_dim]
                    (support_seq_labels == 1).float().unsqueeze(-1),  # [bs, 1]
                    axis=0,  # average over batch size
                )  # [1, emb_dim]

                class_0_cls_support_embedding = masked_mean(
                    support_embedding,  # [bs, emb_dim]
                    (support_seq_labels == 0).float().unsqueeze(-1),  # [bs, 1]
                    axis=0,  # average over batch size
                )  # [1, emb_dim]

                class_1_cls_query_embedding = masked_mean(
                    label_embedding,  # [bq, emb_dim]
                    (query_seq_labels == 1).float().unsqueeze(-1),  # [bq, 1]
                    axis=0,  # average over batch size
                )  # [1, emb_dim]

                class_0_cls_query_embedding = masked_mean(
                    non_label_embedding,  # [bq, emb_dim]
                    (query_seq_labels == 0).float().unsqueeze(-1),  # [bq, 1]
                    axis=0,  # average over batch size
                )  # [1, emb_dim]

                class_1_cls_embedding = torch.cat(
                    [class_1_cls_support_embedding, class_1_cls_query_embedding], dim=0
                ).mean(
                    dim=0, keepdim=True
                )  # [1, emb_dim]

                class_0_cls_embedding = torch.cat(
                    [class_0_cls_support_embedding, class_0_cls_query_embedding], dim=0
                ).mean(
                    dim=0, keepdim=True
                )  # [1, emb_dim]

                class_1_samples = torch.cat(
                    [class_1_embedding, class_1_cls_embedding], dim=0
                )  # [2, emb_dim]

                class_0_samples = torch.cat(
                    [class_0_embedding, class_0_cls_embedding], dim=0
                )  # [2, emb_dim]

                # labels need to be long type
                metric_labels = torch.cat(
                    [
                        torch.ones(len(class_1_samples), dtype=torch.long),
                        torch.zeros(len(class_0_samples), dtype=torch.long),
                    ],
                    dim=0,
                )  # [4]

                # This is our metric based loss
                loss_m = metric_loss(
                    torch.cat([class_1_samples, class_0_samples], dim=0),
                    metric_labels,
                )

                # Token classification loss -- from the document:
                # Now we go to the primary objective, which is token
                # classification.
                # 1. We compute the cosine similarity between the extracted MASK
                #    embedding and each token embedding from the query’s full
                #    sequence embedding.
                # 2. The result will then be multiplied by 10 and then
                #    normalized by sigmoid function, the rescale is to ensure
                #    the output range to be (almost) in the range of (0,1).
                # 3. The ground truth is a binary vector with every token that
                #    belongs to a dataset name denoted as 1 and the rest as 0.
                # 4. The BCE loss is then applied for the output and the
                #    groundtruth, and a training step is completed.

                # Step 1
                cos_sim_mask_query = torch.nn.functional.cosine_similarity(
                    # we need to add a 'token' dimension to the mask embedding
                    # so that it can be broadcasted with the linear_embedding
                    mask_embedding.view(len(mask_embedding), 1, -1),
                    query_linear_embedding,
                    dim=-1,
                )  # [bq, seq_len]

                # Step 2
                cos_sim_mask_query = cos_sim_mask_query * 10  # [bq, seq_len]

                # Step 3 -- we're going to skip this step and compute the loss
                #           with respect to the logits, not the sigmoid because
                #           it's more stable
                # cos_sim_mask_query = torch.sigmoid(cos_sim_mask_query) # [bq, seq_len]

                # Step 4
                # this will be 0 where the tokenizer set -100
                # we don't want to penalize the loss for special tokens indicated
                # by -100
                mask_special = 1 - query_labels["special_token_indicator"][0, :]

                masked_model_outputs = (
                    query_labels["mask_token_indicator"] * mask_special
                )

                loss_t_batch = torch.nn.functional.binary_cross_entropy_with_logits(
                    cos_sim_mask_query,
                    masked_model_outputs.float(),
                    weight=batch_query["attention_mask"],
                    reduction="none",
                )  # [bq, seq_len]

                loss_t_per_sample = loss_t_batch.sum(dim=-1) / batch_query[
                    "attention_mask"
                ].sum(
                    axis=-1
                )  # [bq]

                loss_t = loss_t_per_sample.mean()

                # you have to pass retain_graph=true if you want to call backward
                # multiple times on the same graph
                loss_t.backward(retain_graph=True)
                loss_m.backward()
                metric_optimizer.step()
                optimizer.step()
                scheduler.step()
                metric_scheduler.step()
                step += 1

                if step % config.get("steps_per_eval", 50) == 0:
                    persample_detached = loss_t_per_sample.detach().cpu().numpy()

                    if config.get("save_model", False):
                        save_path = os.path.join(
                            config["model_path"],
                            training_logger.get_key(),
                        )
                        model.save_pretrained(save_path)
                        tokenizer.save_pretrained(save_path)
                        torch.save(
                            linear.state_dict(),
                            os.path.join(save_path, "linear.pt")
                        )

                    with training_logger.train():
                        if persample_detached.shape[0] > 1:
                            best_sample = np.argmin(persample_detached)
                            worst_sample = np.argmax(persample_detached)

                            (
                                best_tokens,
                                best_y_true,
                                best_y_pred,
                            ) = filter_prep_tokens(
                                tokenizer.convert_ids_to_tokens(
                                    batch_query["input_ids"][best_sample]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                ),
                                query_labels["mask_token_indicator"][best_sample]
                                .detach()
                                .cpu()
                                .numpy(),
                                torch.nn.functional.sigmoid(
                                    cos_sim_mask_query[best_sample]
                                )
                                .detach()
                                .cpu()
                                .numpy(),
                            )

                            best_f = color_text_figure_binary(
                                best_tokens,
                                GenericModel1.blk_grn,
                                best_y_true,
                                best_y_pred,
                                threshold=0.5,
                            )
                            training_logger.log_figure("Best Query", best_f, step=step)

                            (
                                worst_tokens,
                                worst_y_true,
                                worst_y_pred,
                            ) = filter_prep_tokens(
                                tokenizer.convert_ids_to_tokens(
                                    batch_query["input_ids"][worst_sample]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                ),
                                query_labels["mask_token_indicator"][worst_sample]
                                .detach()
                                .cpu()
                                .numpy(),
                                torch.nn.functional.sigmoid(
                                    cos_sim_mask_query[worst_sample]
                                )
                                .detach()
                                .cpu()
                                .numpy(),
                            )

                            worst_f = color_text_figure_binary(
                                worst_tokens,
                                GenericModel1.blk_grn,
                                worst_y_true,
                                worst_y_pred,
                                threshold=0.5,
                            )
                            training_logger.log_figure(
                                "Worst Query", worst_f, step=step
                            )

                        else:

                            (tokens, y_true, y_pred) = filter_prep_tokens(
                                tokenizer.convert_ids_to_tokens(
                                    batch_query["input_ids"][0].detach().cpu().numpy()
                                ),
                                query_labels["mask_token_indicator"][0]
                                .detach()
                                .cpu()
                                .numpy(),
                                torch.nn.functional.sigmoid(cos_sim_mask_query[0])
                                .detach()
                                .cpu()
                                .numpy(),
                            )

                            f = color_text_figure_binary(
                                tokens,
                                GenericModel1.blk_grn,
                                y_true,
                                y_pred,
                                threshold=0.5,
                            )
                            training_logger.log_figure("Query", f, step=step)

                        training_logger.log_metric(
                            "loss",
                            persample_detached.mean(),
                            step=step,
                        )

                        training_logger.log_metric(
                            "metric_loss",
                            loss_m.detach().cpu().numpy(),
                            step=step,
                        )

                    # test evaluation ==========================================
                    model.eval()
                    metric_loss.eval()
                    linear.eval()

                    total_loss, total_n = 0, 0
                    total_metric_loss, total_metric_loss_n = 0, 0
                    ng = torch.no_grad()
                    ng.__enter__()
                    for i, test_batch in enumerate(
                        tqdm(
                            repository.get_validation_data(
                                batch_size=config["batch_size"],
                            ),
                            desc=f"Testing Epoch {epoch}",
                        )
                    ):
                        (
                            batch_query,
                            batch_support,
                            query_labels,
                            support_labels,
                        ) = prepare_batch(
                            tokenizer,
                            collator,
                            # config.get("n_query", 1),
                            2,
                            config.get("tokenizer_call_kwargs", {}),
                            test_batch,
                        )

                        batch_query = send_to_device(batch_query)
                        batch_support = send_to_device(batch_support)
                        query_labels = send_to_device(query_labels)
                        support_labels = send_to_device(support_labels)

                        query_seq_labels = query_labels["seq_labels"]  # [bq, 1]
                        support_seq_labels = support_labels["seq_labels"]  # [bs, 1]

                        support_mask_token_indicator = support_labels[
                            "mask_token_indicator"
                        ]  # [bs, seq_len]
                        query_label_token_indicator = query_labels[
                            "mask_token_indicator"
                        ]  # [bq, seq_len]

                        output_support = model(**batch_support)
                        support_hidden_states = (
                            output_support.last_hidden_state
                        )  # [bs, seq_len, emb_dim]
                        support_linear_embedding = linear(support_hidden_states) # [bs, seq_len, emb_dim]

                        support_embedding = support_hidden_states[
                            :, 0, :
                        ]  # [bs, emb_dim]

                        mask_embedding = masked_mean(
                            support_linear_embedding,
                            support_mask_token_indicator.unsqueeze(-1),
                            axis=1,  # average over sequence length
                            keepdim=False,
                        ).mean(
                            axis=0, keepdim=True
                        )  # average over batch size

                        non_mask_embedding = masked_mean(
                            support_linear_embedding,
                            (1 - support_mask_token_indicator).unsqueeze(-1),
                            axis=1,  # average over sequence length
                            keepdim=False,
                        ).mean(
                            axis=0, keepdim=True
                        )  # average over batch size

                        output_query = model(**batch_query)
                        query_hidden_states = (
                            output_query.last_hidden_state
                        )  # [bq, seq_len, emb_dim]
                        query_linear_embedding = linear(query_hidden_states) # [bq, seq_len, emb_dim]

                        label_embedding = masked_mean(
                            query_linear_embedding,
                            query_label_token_indicator.unsqueeze(-1),
                            axis=1,  # average over sequence length
                            keepdim=False,
                        )  # [bq, emb_dim]

                        non_label_embedding = masked_mean(
                            query_linear_embedding,
                            (1 - query_label_token_indicator).unsqueeze(-1)
                            * batch_query["attention_mask"].unsqueeze(-1),
                            axis=1,  # average over sequence length
                            keepdim=False,
                        )  # [bq, emb_dim]

                        class_1_embedding = torch.cat(
                            [mask_embedding, label_embedding], dim=0
                        ).mean(
                            dim=0, keepdim=True
                        )  # [1, emb_dim]

                        class_0_embedding = torch.cat(
                            [non_mask_embedding, non_label_embedding], dim=0
                        ).mean(
                            dim=0, keepdim=True
                        )  # [1, emb_dim]

                        class_1_cls_support_embedding = masked_mean(
                            support_embedding,  # [bs, emb_dim]
                            (support_seq_labels == 1).float().unsqueeze(-1),  # [bs, 1]
                            axis=0,  # average over batch size
                        )  # [1, emb_dim]

                        class_0_cls_support_embedding = masked_mean(
                            support_embedding,  # [bs, emb_dim]
                            (support_seq_labels == 0).float().unsqueeze(-1),  # [bs, 1]
                            axis=0,  # average over batch size
                        )  # [1, emb_dim]

                        class_1_cls_query_embedding = masked_mean(
                            label_embedding,  # [bq, emb_dim]
                            (query_seq_labels == 1).float().unsqueeze(-1),  # [bq, 1]
                            axis=0,  # average over batch size
                        )  # [1, emb_dim]

                        class_0_cls_query_embedding = masked_mean(
                            non_label_embedding,  # [bq, emb_dim]
                            (query_seq_labels == 0).float().unsqueeze(-1),  # [bq, 1]
                            axis=0,  # average over batch size
                        )  # [1, emb_dim]

                        class_1_cls_embedding = torch.cat(
                            [
                                class_1_cls_support_embedding,
                                class_1_cls_query_embedding,
                            ],
                            dim=0,
                        ).mean(
                            dim=0, keepdim=True
                        )  # [1, emb_dim]

                        class_0_cls_embedding = torch.cat(
                            [
                                class_0_cls_support_embedding,
                                class_0_cls_query_embedding,
                            ],
                            dim=0,
                        ).mean(
                            dim=0, keepdim=True
                        )  # [1, emb_dim]

                        class_1_samples = torch.cat(
                            [class_1_embedding, class_1_cls_embedding], dim=0
                        )  # [2, emb_dim]

                        class_0_samples = torch.cat(
                            [class_0_embedding, class_0_cls_embedding], dim=0
                        )  # [2, emb_dim]

                        metric_labels = torch.cat(
                            [
                                torch.ones(len(class_1_samples), dtype=torch.long),
                                torch.zeros(len(class_0_samples), dtype=torch.long),
                            ],
                            dim=0,
                        )  # [4]

                        loss_m = metric_loss(
                            torch.cat([class_1_samples, class_0_samples], dim=0),
                            metric_labels,
                        )

                        cos_sim_mask_query = torch.nn.functional.cosine_similarity(
                            mask_embedding.view(len(mask_embedding), 1, -1),
                            query_linear_embedding,
                            dim=-1,
                        )  # [bq, seq_len]

                        cos_sim_mask_query = cos_sim_mask_query * 10  # [bq, seq_len]

                        mask_special = (
                            1
                            - (
                                query_labels["mask_token_indicator"][0, :] == -100
                            ).float()
                        )
                        masked_model_outputs = (
                            query_labels["mask_token_indicator"] * mask_special
                        )
                        loss_t_batch = (
                            torch.nn.functional.binary_cross_entropy_with_logits(
                                cos_sim_mask_query,
                                masked_model_outputs,
                                weight=batch_query["attention_mask"],
                                reduction="none",
                            )
                        )  # [bq, seq_len]

                        loss_t_per_sample = loss_t_batch.sum(dim=-1) / batch_query[
                            "attention_mask"
                        ].sum(
                            axis=-1
                        )  # [bq]

                        loss_t = loss_t_per_sample.mean()

                        total_loss += loss_t.detach().cpu().numpy()
                        total_metric_loss += loss_m.detach().cpu().numpy()
                        total_n += len(batch_query["input_ids"])
                        total_metric_loss_n += len(class_1_samples) + len(
                            class_0_samples
                        )
                        if i == config.get("test_eval_batches", 10):
                            persample_detached = (
                                loss_t_per_sample.detach().cpu().numpy()
                            )
                            with training_logger.test():
                                training_logger.log_metric(
                                    "loss",
                                    total_loss / total_n,
                                    step=step,
                                )

                                training_logger.log_metric(
                                    "metric_loss",
                                    total_metric_loss / total_metric_loss_n,
                                    step=step,
                                )

                                if persample_detached.shape[0] > 1:
                                    best_sample = np.argmin(persample_detached)
                                    worst_sample = np.argmax(persample_detached)

                                    (
                                        best_tokens,
                                        best_y_true,
                                        best_y_pred,
                                    ) = filter_prep_tokens(
                                        tokenizer.convert_ids_to_tokens(
                                            batch_query["input_ids"][best_sample]
                                            .detach()
                                            .cpu()
                                            .numpy()
                                        ),
                                        query_labels["mask_token_indicator"][
                                            best_sample
                                        ]
                                        .detach()
                                        .cpu()
                                        .numpy(),
                                        torch.nn.functional.sigmoid(
                                            cos_sim_mask_query[best_sample]
                                        )
                                        .detach()
                                        .cpu()
                                        .numpy(),
                                    )

                                    best_f = color_text_figure_binary(
                                        best_tokens,
                                        GenericModel1.blk_grn,
                                        best_y_true,
                                        best_y_pred,
                                        threshold=0.5,
                                    )
                                    training_logger.log_figure(
                                        "Best Query", best_f, step=step
                                    )

                                    (
                                        worst_tokens,
                                        worst_y_true,
                                        worst_y_pred,
                                    ) = filter_prep_tokens(
                                        tokenizer.convert_ids_to_tokens(
                                            batch_query["input_ids"][worst_sample]
                                            .detach()
                                            .cpu()
                                            .numpy()
                                        ),
                                        query_labels["mask_token_indicator"][
                                            worst_sample
                                        ]
                                        .detach()
                                        .cpu()
                                        .numpy(),
                                        torch.nn.functional.sigmoid(
                                            cos_sim_mask_query[worst_sample]
                                        )
                                        .detach()
                                        .cpu()
                                        .numpy(),
                                    )

                                    worst_f = color_text_figure_binary(
                                        worst_tokens,
                                        GenericModel1.blk_grn,
                                        worst_y_true,
                                        worst_y_pred,
                                        threshold=0.5,
                                    )
                                    training_logger.log_figure(
                                        "Worst Query", worst_f, step=step
                                    )

                                else:

                                    (tokens, y_true, y_pred) = filter_prep_tokens(
                                        tokenizer.convert_ids_to_tokens(
                                            batch_query["input_ids"][0]
                                            .detach()
                                            .cpu()
                                            .numpy()
                                        ),
                                        query_labels["mask_token_indicator"][0]
                                        .detach()
                                        .cpu()
                                        .numpy(),
                                        torch.nn.functional.sigmoid(
                                            cos_sim_mask_query[0]
                                        )
                                        .detach()
                                        .cpu()
                                        .numpy(),
                                    )

                                    f = color_text_figure_binary(
                                        tokens,
                                        GenericModel1.blk_grn,
                                        y_true,
                                        y_pred,
                                        threshold=0.5,
                                    )
                                    training_logger.log_figure("Query", f, step=step)

                            break

                    ng.__exit__(None, None, None)

                # the balanced version of the dataset is quite large so offer
                # the option to get out of the epoch early
                if step >= config.get("steps_per_epoch", np.inf):
                    break


if __name__ == "__main__":
    bm.train = train
    bm.validate = validate
    bm.main()

    # import src.data.kaggle_repository as kr
    # import src.evaluate.model as em

    # class MockRepo:
    #     def __init__(self, df):
    #         self.df = df
    #     def get_validation_data(self):
    #         return self.df
    #     def copy(self):
    #         return MockRepo(self.df.copy())


    # model_name = "biomed_roberta"
    # config = dict(
    #     support_no_mask_embedding_path = f"models/generic_model1/sub_{model_name}/embeddings/support_nomask_embeddings.npy",
    #     support_mask_embedding_path = f"models/generic_model1/sub_{model_name}/embeddings/support_mask_embeddings.npy",
    #     batch_size = 2,
    #     threshold = 0.7,
    #     inference_progress_bar = True,
    #     n_support_samples = 16 * 100,
    #     model_tokenizer_name = f"models/generic_model1/sub_{model_name}",
    #     model_kwargs=dict(from_tf=True, output_hidden_states=True),
    #     tokenizer_kwargs=dict(add_prefix_space=True),
    #     tokenizer_call_kwargs=dict(max_length=512, truncation=True, is_split_into_words=True),
    #     is_roberta=True,
    # )

    # print("getting")
    # repo = MockRepo(next(kr.KaggleRepository().get_validation_data(batch_size=32)))
    # print("data retieved")
    # outs = em.evaluate_model(
    #     repo.copy(),
    #     GenericModel1(),
    #     config,
    # )

    # from src.data.repository_resolver import resolve_repo

    # repository = resolve_repo("snippet-masked_lm")
    # config = dict(
    #     model_tokenizer_name="allenai/scibert_scivocab_cased",
    #     tokenizer_kwargs={},
    #     tokenizer_call_kwargs=dict(max_length=512, truncation=True, is_split_into_words=True),
    #     model_kwargs={},
    #     optimizer="torch.optim.AdamW",
    #     optimizer_kwargs=dict(lr=1e-5),
    #     metric_optimizer="torch.optim.SGD",
    #     metric_optimizer_kwargs=dict(lr=1e-2),
    #     batch_size=8,
    #     epochs=1,
    #     steps_per_eval=10,
    #     balance_labels=True,
    #     n_query=2,
    #     save_model=True,
    # )

    # from comet_ml import Experiment

    # training_logger = Experiment(
    #     workspace="democratizingdata",
    #     project_name="generic-model1",
    #     auto_metric_logging=False,
    #     disabled=False,
    # )

    # model = GenericModel1()
    # model.train(repository, config, training_logger)
