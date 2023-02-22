# This is a token classification model trained with two loss functions:
# 1. Cross entropy loss for the dataset tokens
# 2. Some other loss for comparing embeddings of the context tokens
#
# See https://github.com/Coleridge-Initiative/rc-kaggle-models/blob/main/1st%20ZALO%20FTW/MODEL_SUMMARY.pdf
# for more details.



from functools import partial
from itertools import islice, starmap
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import datasets as ds
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
import transformers as tfs
from pytorch_metric_learning import losses as metric_losses
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

def train(
    repository: Repository,
    config: Dict[str, Any],
    training_logger: Optional[bm.SupportsLogging] = None,
) -> None:
        validate_config(config)

        training_logger.log_parameters(
            {
                "model_tokenizer_name": config["model_tokenizer_name"],
                "optimizer": config["optimizer"],
            }
            | config["tokenizer_kwargs"]
            | config["model_kwargs"]
            | config["optimizer_kwargs"]
        )



def validate(repository: Repository, config: Dict[str, Any]) -> None:
    validate_config(config)


def convert_to_T(T:type, vals:List[str]) -> List[float]:
    return [T(x) for x in vals]


def tokenize_and_align_labels(
    tokenizer_f:Callable[[Dict[str, Any]], Dict[str, Any]],
    examples:Dict[str, Any]
) -> Dict[str, Any]:
    tokenized_inputs = tokenizer_f(examples["text"])

    labels = []
    for i, label in enumerate(examples["mask_token_indicator"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100] * len(word_ids) # assume all tokens are special
        top_word_id = max(map(lambda x: x if x else -1, word_ids))
        for word_idx in range(top_word_id + 1):
            label_ids[word_ids.index(word_idx)] = label[word_idx]
        labels.append(label_ids)

    tokenized_inputs["mask_token_indicator"] = labels
    return tokenized_inputs


def apply_mask_sample(
    tokens:List[str],
    mask_token_indicator:List[float]
) -> List[str]:

    tokens = list(map(
        lambda t, m: "[MASK]" if m else t,
        tokens,
        mask_token_indicator
    ))
    return tokens


def apply_mask_batched(dataset:Dict[str, Any]) -> Dict[str, Any]:
    # inintially every token is masked, however, we want to group them
    # so that a single token represents an entire dataset
    ungrouped_masks = list(starmap(
        apply_mask_sample,
        zip(dataset["text"], dataset["mask"]),
    ))

    text_mask = list(zip(*list(starmap(
        group_mask_sample,
        zip(ungrouped_masks, dataset["mask"]),
    ))))

    dataset["text"], dataset["mask"] = list(text_mask[0]), list(text_mask[1])

    return dataset


def group_mask_sample(
    tokens:List[str],
    mask_token_indicator:List[float]
) -> List[str]:

    # group the masks
    grouped_text_masks = [tokens[0]]
    grouped_mask_token_indicator = [mask_token_indicator[0]]

    for index in range(1, len(tokens)):
        if not (mask_token_indicator[index] == 1 and mask_token_indicator[index-1] == 1):
            grouped_text_masks.append(tokens[index])
            grouped_mask_token_indicator.append(mask_token_indicator[index])

    return grouped_text_masks, grouped_mask_token_indicator


def convert_dataset(
    tokenizer_f:Callable,
    collator:Callable,
    dataset:ds.Dataset
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

    convert_f = partial(convert_to_T, int)

    dataset = dataset.map(
        lambda dset: { "mask_token_indicator" : list(map(convert_f, dset["mask"]))},
        batched=True,
    ).map(
        partial(tokenize_and_align_labels, tokenizer_f),
        batched=True,
    ).remove_columns(
        ["text", "mask"]
    # we rename mask_token_indicator to labels, because that is what the
    # so that the data collator will pad it.
    ).rename_column(
        "mask_token_indicator", "labels"
    ).rename_column(
        "label", "seq_labels"
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

    dataset_labels = dict(
        mask_token_indicator = dataset["labels"],
        seq_labels = torch.Tensor(tmp_seq_labels),
    )
    # At this point, the dataset is a dictionary of tensors, where the
    # tensors are all the same length.
    return dataset_inputs, dataset_labels


def prepare_batch(
        tokenizer: tfs.tokenization_utils_base.PreTrainedTokenizerBase,
        data_collator: tfs.data.data_collator.DataCollatorMixin,
        n_query: int,
        batch: pd.DataFrame,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

    # we need to split the batch into support and query sets, well use the
    # train/test split functionality from datasets.Dataset to select random
    # subsets of the batch as the query/support sets. where query will be "train"
    dataset = ds.Dataset.from_pandas(
        batch.drop(columns=["pos_tags"])
    ).train_test_split(train_size=n_query)

    ds_query= dataset["train"]
    ds_support = dataset["test"].map(apply_mask_batched, batched=True)

    batch_query, labels_query = convert_dataset(
        tokenizer_f = partial(tokenizer, is_split_into_words=True, truncation=True),
        collator = data_collator,
        dataset = ds_query,
    )

    batch_support, labels_support = convert_dataset(
        tokenizer_f = partial(tokenizer, is_split_into_words=True, truncation=True),
        collator = data_collator,
        dataset = ds_support,
    )

    return batch_query, batch_support, labels_query, labels_support


# based on
# https://stackoverflow.com/questions/36264305/matplotlib-multi-colored-title-text-in-practice
def color_text_figure_binary(tokens, cmap, y_true, y_pred, threshold=0.5):
    f, ax = plt.subplots(figsize=(10, 1))
    r = f.canvas.get_renderer()
    ax.set_axis_off()
    space = 0.025
    w = 0.0
    for i, (token, y, yh) in enumerate(zip(tokens, y_true, y_pred)):
        t = ax.text(w, 0.25, token, color=cmap(y), ha="left", va="center", fontsize=18)
        ax.text(w, 0.75, token, color=cmap(yh), ha="left", va="center", fontsize=18)

        if y >= threshold:
            ax.text(w, 0.0, "{:.2f}".format(y), color="black", ha="left", va="center", fontsize=10)
        if yh >= threshold:
            ax.text(w, 1.0, "{:.2f}".format(yh), color="black", ha="left", va="center", fontsize=10)

        transf = ax.transData.inverted()
        bb = t.get_window_extent(renderer=f.canvas.renderer)
        bb = bb.transformed(transf)
        w = w + bb.xmax-bb.xmin + space
    return f


class GenericModel1(bm.Model):
    blk_grn = mcolors.LinearSegmentedColormap.from_list(
        'my_colormap',
        ['black','green'],
    )


    def get_model_objects(
        self,
        config:Dict[str, Any],
        include_optimizer:bool
    ) -> None:

        tokenizer = tfs.AutoTokenizer.from_pretrained(
            config["model_tokenizer_name"],
            **config.get("tokenizer_kwargs", {})
        )

        collator = tfs.data.data_collator.DataCollatorForTokenClassification(
            tokenizer,
            return_tensors="pt",
            label_pad_token_id=0,
        )

        model = tfs.AutoModel.from_pretrained(
            config["model_tokenizer_name"],
            **config.get("model_kwargs", {})
        )

        if torch.cuda.is_available(): model = model.cuda()

        if include_optimizer:
            optimizer = eval(config["optimizer"])(
                model.parameters(),
                **config.get("optimizer_kwargs", {})
            )

            if "scheduler" in config:
                scheduler = eval(config["scheduler"])(
                    optimizer,
                    **config.get("scheduler_kwargs", {})
                )
            else:
                scheduler = bm.MockLRScheduler()

            metric_loss = metric_losses.ArcFaceLoss(
                num_classes=2,
                embedding_size=model.config.hidden_size,
                **config.get("metric_loss_kwargs", {})
            )

            metric_optimizer = eval(config["metric_optimizer"])(
                metric_loss.parameters(),
                **config.get("metric_optimizer_kwargs", {})
            )

            if "metric_scheduler" in config:
                metric_scheduler = eval(config["metric_scheduler"])(
                    metric_optimizer,
                    **config.get("metric_scheduler_kwargs", {})
                )
            else:
                metric_scheduler = bm.MockLRScheduler()


            return (
                model,
                tokenizer,
                collator,
                optimizer,
                scheduler,
                metric_loss,
                metric_optimizer,
                metric_scheduler
            )
        else:
            return model, tokenizer, collator


    def inference(self, config: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
        return super().inference(config, df)


    def train(self, repository: Repository, config: Dict[str, Any], training_logger: bm.SupportsLogging) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        (
            model,
            tokenizer,
            collator,
            optimizer,
            scheduler,
            metric_loss,
            metric_optimizer,
            metric_scheduler
        ) = self.get_model_objects(config, include_optimizer=True)

        training_iter = repository.get_training_data(
            batch_size=config["batch_size"],
            balance_labels = config.get("balance_labels", False)
        )

        step  = config.get("start_step", 0)
        for epoch in range(config["epochs"]):
            for batch in tqdm(training_iter, desc=f"Training Epoch {epoch}"):
                model.train()
                metric_loss.train()

                # bs = support batch size
                # bq = query batch size
                # seq_len = sequence length
                # emb_dim = embedding dimension

                (
                    batch_query,
                    batch_support,
                    query_labels,
                    support_labels
                ) = prepare_batch(tokenizer, collator, config.get("n_query", 1), batch)

                send_to_device = lambda d: {k:v.to(device) for k,v in d.items()}
                batch_query = send_to_device(batch_query)
                batch_support = send_to_device(batch_support)
                query_labels = send_to_device(query_labels)
                support_labels = send_to_device(support_labels)

                # These are the snippet level labels
                query_seq_labels = query_labels["seq_labels"] # [bq, 1]
                support_seq_labels = support_labels["seq_labels"] # [bs, 1]

                # These indicate which of the support tokens are mask tokens
                support_mask_token_indicator = support_labels["mask_token_indicator"] # [bs, seq_len]
                # These indicate which of the query tokens are label tokens
                query_label_token_indicator = query_labels["mask_token_indicator"] # [bq, seq_len]

                optimizer.zero_grad()
                metric_optimizer.zero_grad()

                # Variables are named to match the document from the first place
                # solution document, linked above, Section 2.3.3.3. Embeddings extraction


                # First process the support set ================================
                output_support = model(**batch_support)
                support_hidden_states = output_support.last_hidden_state # [bs, seq_len, emb_dim]
                # might be support_hidden_states = output_support.hidden_states[-1]


                # the first token is the CLS token which is the support embedding
                support_embedding = support_hidden_states[:, 0, :] # [bs, emb_dim]

                # NEED TO ALSO ZERO OUT BY ATTENTION MASK

                # zero-out non mask tokens
                #                         [bs, seq_len, emb_dim] * [bs, seq_len, 1]
                support_mask_tokens_seq = support_hidden_states * support_mask_token_indicator.unsqueeze(-1) # [bs, seq_len, emb_dim]
                support_mask_token_sum = support_mask_tokens_seq.sum(dim=1) # [bs, emb_dim]
                support_mask_token_n = support_mask_token_indicator.sum(dim=1, keepdim=True) # [bs, 1]
                mask_embedding = (support_mask_token_sum / support_mask_token_n).mean(axis=0, keepdim=True) # [1, emb_dim]

                # flip the mask and zero-out mask tokens
                support_non_mask_tokens_seq = support_hidden_states * (1 - support_mask_token_indicator).unsqueeze(-1) # [bs, seq_len, emb_dim]
                                            # [bs, seq_len, emb_dim]      * [bs, seq_len, 1]
                support_non_mask_tokens_seq = support_non_mask_tokens_seq * batch_support["attention_mask"].unsqueeze(-1) # [bs, seq_len, emb_dim]
                support_non_mask_token_sum = support_non_mask_tokens_seq.sum(dim=1) # [bs, emb_dim]
                support_non_mask_token_n = (1 - support_mask_token_indicator).sum(dim=1, keepdim=True) # [bs, 1]
                non_mask_embedding = (support_non_mask_token_sum / support_non_mask_token_n).mean(axis=0, keepdim=True) # [1, emb_dim]
                # ==============================================================

                # Second process the query set =================================
                output_query = model(**batch_query)
                query_hidden_states = output_query.last_hidden_state # [bq, seq_len, emb_dim]
                # might be query_hidden_states = output_query.hidden_states[-1]

                query_mask_tokens_seq = query_hidden_states * query_label_token_indicator.unsqueeze(-1) # [bq, seq_len, emb_dime]
                query_mask_token_sum = query_mask_tokens_seq.sum(dim=1) # [bq, emb_dim]
                query_mask_token_n = query_label_token_indicator.sum(dim=1, keepdim=True) # [bq, 1]
                label_embedding = query_mask_token_sum / query_mask_token_n # [bq, emb_dim]

                # flip the mask and zero-out mask tokens
                query_non_mask_tokens_seq = query_hidden_states * (1 - query_label_token_indicator).unsqueeze(-1) # [bq, seq_len, emb_dime]
                                            # [bq, seq_len, emb_dim]      * [bq, seq_len, 1]
                query_non_mask_tokens_seq = query_non_mask_tokens_seq * batch_query["attention_mask"].unsqueeze(-1) # [bq, seq_len, emb_dim]
                query_non_mask_token_sum = query_non_mask_tokens_seq.sum(dim=1) # [bq, emb_dim]
                query_non_mask_token_n = (1 - query_label_token_indicator).sum(dim=1, keepdim=True) # [bq, 1]
                non_label_embedding = query_non_mask_token_sum / query_non_mask_token_n # [bq, emb_dim]
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
                class_1_embedding = torch.concat([mask_embedding, label_embedding], dim=0).mean(dim=0, keepdim=True) # [1, emb_dim]


                # We want to group the non-masked tokens from the support set
                # with the negative tokens from the query set. Both of these
                # embeddings represent non-dataset tokens.
                class_0_embedding = torch.concat([non_mask_embedding, non_label_embedding], dim=0).mean(dim=0, keepdim=True) # [1, emb_dim]

                #TODO: continue here with the boolean mask flattening things
                # We want to group the [cls] tokens into their respective classes
                def masked_average(
                    mask:torch.Tensor, # [bs]
                    embeddings:torch.Tensor, # [bs, emb_dim]
                ) -> torch.Tensor: # [emb_dim]
                    mask_float = mask.float() # [bs]
                    masked_vals = mask_float.unsqueeze(-1) * embeddings # [bs, emb_dim]
                    n = mask_float.sum() # [1]
                    return masked_vals.sum(dim=0, keepdim=True) / n # [1, emb_dim]

                class_1_cls_support_embedding = masked_average(support_seq_labels == 1, support_embedding) # [1, emb_dim]
                class_0_cls_support_embedding = masked_average(support_seq_labels == 0, support_embedding) # [1, emb_dim]

                class_1_cls_query_embedding = masked_average(query_seq_labels == 1, label_embedding) # [1, emb_dim]
                class_0_cls_query_embedding = masked_average(query_seq_labels == 0, non_label_embedding) # [1, emb_dim]

                class_1_cls_embedding = torch.concat(
                    [class_1_cls_support_embedding, class_1_cls_query_embedding],
                    dim=0
                ).mean(dim=0, keepdim=True) # [1, emb_dim]
                class_0_cls_embedding = torch.concat(
                    [class_0_cls_support_embedding, class_0_cls_query_embedding],
                    dim=0
                ).mean(dim=0, keepdim=True) # [1, emb_dim]

                class_1_samples = torch.concat(
                    [class_1_embedding, class_1_cls_embedding],
                    dim=0
                ) # [2, emb_dim]

                class_0_samples = torch.concat(
                    [class_0_embedding, class_0_cls_embedding],
                    dim=0
                ) # [2, emb_dim]

                # labels need to be long type
                metric_labels = torch.concat(
                    [
                        torch.ones(len(class_1_samples), dtype=torch.long),
                        torch.zeros(len(class_0_samples), dtype=torch.long)
                    ],
                    dim=0
                ) # [4]

                # This is our metric based loss
                loss_m = metric_loss(
                    torch.concat([class_1_samples, class_0_samples], dim=0),
                    metric_labels
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
                    # so that it can be broadcasted with the query_hidden_states
                    mask_embedding.view(len(mask_embedding), 1, -1),
                    query_hidden_states,
                    dim=-1
                ) # [bq, seq_len]

                # Step 2
                cos_sim_mask_query = cos_sim_mask_query * 10 # [bq, seq_len]

                # Step 3 -- we're going to skip this step and compute the loss
                #           with respect to the logits, not the sigmoid because
                #           it's more stable
                # cos_sim_mask_query = torch.sigmoid(cos_sim_mask_query) # [bq, seq_len]


                # Step 4
                # this will be 0 where the tokenizer set -100
                # we don't want to penalize the loss for special tokens indicated
                # by -100
                mask_special = ((query_labels["mask_token_indicator"][0,:] == -100).float() - 1).abs()
                masked_model_outputs = query_labels["mask_token_indicator"] * mask_special
                loss_t_batch = torch.nn.functional.binary_cross_entropy_with_logits(
                    cos_sim_mask_query,
                    masked_model_outputs,
                    weight=batch_query["attention_mask"],
                    reduction="none"
                ) # [bq, seq_len]

                loss_t_per_sample = loss_t_batch.sum(dim=-1) / batch_query["attention_mask"].sum(axis=-1) # [bq]

                loss_t = loss_t_per_sample.mean()

                # you have to pass retain_graph=true if you want to call backward
                # multiple times on the same graph
                loss_t.backward(retain_graph=True)
                loss_m.backward()
                metric_optimizer.step()
                optimizer.step()
                scheduler.step()
                metric_scheduler.step()

                if step % config.get("steps_per_eval", 50) == 0:

                    persample_detached = loss_t_per_sample.detach().cpu().numpy()

                    with training_logger.train():
                        if persample_detached.shape[0] > 1:
                            best_sample = np.argmin(persample_detached)
                            worst_sample = np.argmax(persample_detached)

                            best_f = color_text_figure_binary(
                                tokenizer.convert_ids_to_tokens(
                                    batch_query["input_ids"][best_sample].detach().cpu().numpy()
                                ),
                                GenericModel1.blk_grn,
                                query_labels["mask_token_indicator"][best_sample].detach().cpu().numpy(),
                                masked_model_outputs[best_sample].detach().cpu().numpy(),
                                threshold=0.5,
                            )
                            training_logger.log_figure("Best Query", best_f, step=step)

                            worst_f = color_text_figure_binary(
                                tokenizer.convert_ids_to_tokens(
                                    batch_query["input_ids"][worst_sample].detach().cpu().numpy()
                                ),
                                GenericModel1.blk_grn,
                                query_labels["mask_token_indicator"][worst_sample].detach().cpu().numpy(),
                                masked_model_outputs[worst_sample].detach().cpu().numpy(),
                                threshold=0.5,
                            )
                            training_logger.log_figure("Worst Query", worst_f, step=step)


                        else:
                            f = color_text_figure_binary(
                                tokenizer.convert_ids_to_tokens(
                                    batch_query["input_ids"][0].detach().cpu().numpy()
                                ),
                                GenericModel1.blk_grn,
                                query_labels["mask_token_indicator"][0].detach().cpu().numpy(),
                                masked_model_outputs[0].detach().cpu().numpy(),
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

                    total_loss, total_n = 0, 0

                    for i, batch in enumerate(tqdm(repository.get_test_data(
                        batch_size=config["batch_size"],
                        balanced=True,
                    ), desc=f"Testing Epoch {epoch}")):
                        pass

                        # inference time is different than training time because
                        # we need support sequences for inference



                        if i == config.get("test_eval_batches", 10):
                            break







if __name__ == "__main__":
    # bm.train = train
    # bm.validate = validate
    # bm.main()

    from src.data.repository_resolver import resolve_repo

    repository = resolve_repo("snippet-masked_lm")
    config = dict(
        model_tokenizer_name =  "allenai/scibert_scivocab_cased",
        tokenizer_kwargs = {},
        model_kwargs = {},
        optimizer =  "torch.optim.AdamW",
        optimizer_kwargs = dict(lr=1e-5),
        metric_optimizer = "torch.optim.SGD",
        metric_optimizer_kwargs = dict(lr=1e-2),
        batch_size = 8,
        epochs = 1
    )

    from comet_ml import Experiment

    training_logger = Experiment(
        workspace="democratizingdata",
        project_name="generic-model1",
        auto_metric_logging=False,
        disabled=True,
    )

    model = GenericModel1()
    model.train(repository, config, training_logger)