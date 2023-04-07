# notebook: https://www.kaggle.com/code/dathudeptrai/biomed-roberta-scibert-base/notebook
# model summary: https://github.com/Coleridge-Initiative/rc-kaggle-models/blob/main/1st%20ZALO%20FTW/MODEL_SUMMARY.pdf

import gc
import glob
from itertools import count, chain, starmap
from collections import Counter
import json
import os
import random
import re
import sys
from typing import Dict, List

from time import time  # GL

import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import (
    AutoConfig,
    BertTokenizerFast,
    DistilBertTokenizerFast,
    RobertaTokenizerFast,
    TFAutoModel,
)
from tqdm import tqdm

Document = List[Dict[str, str]]

# https://stackoverflow.com/q/57539273
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/disable_interactive_logging
# tf.keras.utils.disable_interactive_logging()
tf.get_logger().setLevel("ERROR")
physical_devices = tf.config.list_physical_devices("GPU")
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

from src.models.kaggle_model1_support.model_1_QueryDataLoader import QueryDataLoader
from src.models.kaggle_model1_support.model_1_MetricLearningModel_static import (
    MetricLearningModel,
)
from src.models.kaggle_model1_support.model_1_SupportQueryDataLoader import (
    SupportQueryDataLoader,
)

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
stop_words = set(stopwords.words("english"))

tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)

# Some of the code is too chatty if you wrap it in this, then it redirect the
# print statements to a black hole
# https://stackoverflow.com/a/54955536
from contextlib import contextmanager


@contextmanager
def stdout_redirector():
    class MyStream:
        def write(self, msg):
            pass

        def flush(self):
            pass

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    #     sys.stdout = MyStream()
    #     sys.stderr = MyStream()

    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


class Model1:
    def __init__(self, win_size, sequence_legnth) -> None:
        self.win_size = win_size
        self.sequence_length = sequence_legnth
        self.id = count()

    def preprocess(self, json_text: List[Dict[str, str]]) -> str:
        test_df = dict()

        sents = []
        full_text = list(map(lambda s: s["text"].replace("\n", " ").split(), json_text))
        for ft in full_text:

            st_end = generate_s_e_window_sliding(
                len(ft), self.win_size, int(0.75 * self.win_size)
            )

            for start, end in st_end:
                sents.append(" ".join(ft[start:end]).strip())

        test_df["text"] = sents

        test_df["id"] = ["-1"] * len(sents)
        test_df["label"] = ["unknow"] * len(sents)  # (sic)
        test_df["unique_id"] = ["-1"] * len(sents)

        full_text = " ".join(list(chain.from_iterable(filter(lambda x: x, full_text))))

        test_df["full_text"] = [full_text] * len(sents)
        test_df["group"] = [-1] * len(sents)
        test_df["title"] = [""] * len(sents)

        return pd.DataFrame(test_df)

    def batch_preprocess_single(self, text: Document):
        test_df = dict()

        sents = []
        full_text = list(map(lambda s: s["text"].replace("\n", " ").split(), text))
        for ft in full_text:

            st_end = generate_s_e_window_sliding(
                len(ft), self.win_size, int(0.75 * self.win_size)
            )

            for start, end in st_end:
                sents.append(" ".join(ft[start:end]).strip())

        test_df["text"] = sents
        ids = [text[0]["id"]] * len(sents)
        test_df["id"] = ids
        test_df["label"] = ["unknow"] * len(sents)  # (sic)

        full_text = " ".join(list(chain.from_iterable(filter(lambda x: x, full_text))))

        test_df["full_text"] = [full_text] * len(sents)
        test_df["group"] = [-1] * len(sents)
        test_df["title"] = [""] * len(sents)

        return pd.DataFrame(test_df)

    def batch_preprocess(self, text: List[Document]):
        results = list(map(self.batch_preprocess_single, tqdm(text)))

        ids = []
        texts = []
        labels = []
        unique_ids = []
        full_texts = []
        for result in tqdm(results):
            ids.extend(result["id"])
            texts.extend(result["text"])
            labels.extend(result["label"])
            full_texts.extend(result["full_text"])

        test_df = pd.DataFrame()
        test_df["id"] = ids
        test_df["text"] = texts
        test_df["label"] = labels
        test_df["group"] = [-1] * len(ids)
        test_df["title"] = [""] * len(ids)
        test_df["full_text"] = full_texts

        return test_df

    def batch_predict(
        self, inputs: pd.DataFrame, batch_size: int = 128
    ) -> pd.DataFrame:
        # this includes all of the predicted values for all the values in `inputs`
        accepted_predictions = np.unique(
            get_filtered_models_predictions(inputs, batch_size)
        )

        # match the accepted predictions back to their respective documents by
        # searching for each candidate in each document
        unique_ids = np.unique(inputs["id"].values)
        unique_texts = [
            inputs.loc[inputs["id"] == i, ["full_text"]].iloc[0, 0] for i in unique_ids
        ]

        predictions = [
            find_all_pred_in_text(ut, accepted_predictions) for ut in unique_texts
        ]

        return list(zip(unique_ids, predictions))

    def predict(self, text: Document) -> List[str]:

        accepted_predictions = get_filtered_models_predictions(text)

        predictions = find_all_pred_in_text(
            text.loc[0, "full_text"], np.unique(accepted_predictions)
        )

        predictions = np.unique(list(map(lambda x: clean_text(x.strip()), predictions)))

        return predictions


def clean_text(txt, lower=True):
    if lower:
        return re.sub("[^A-Za-z0-9]+", " ", str(txt).lower())
    else:
        return re.sub("[^A-Za-z0-9]+", " ", str(txt))


def clean_text_v2(txt):
    return re.sub("[^A-Za-z0-9\(\)\[\]]+", " ", str(txt).lower())


def generate_s_e_window_sliding(sample_len, win_size, step_size):
    start = 0
    end = win_size
    s_e = []
    s_e.append([start, end])
    while end < sample_len:
        start += step_size
        end = start + win_size
        s_e.append([start, end])

    s_e[-1][0] -= s_e[-1][1] - sample_len
    s_e[-1][0] = max(s_e[-1][0], 0)
    s_e[-1][1] = sample_len
    return s_e


def remove_acronym(preds):
    for i in range(len(preds)):
        pred_i = preds[i]
        pred_i = pred_i.replace("( ", "(")
        pred_i = pred_i.replace(" )", ")")
        pred_i = pred_i.replace("[ ", "[")
        pred_i = pred_i.replace(" ]", "]")
        try:
            new_pred_i = []
            for pi in pred_i.split("|"):
                if pi != "":
                    words = pi.split()
                    if "(" in words[-1] or "[" in words[-1]:
                        new_pred_i.append(" ".join(words[:-1]))
                    else:
                        new_pred_i.append(pi)
            new_pred_i = "|".join(new_pred_i)
            preds[i] = new_pred_i
        except:
            pass
    return preds


def remove_overlap(preds, preds_low_confidence):
    for i in range(len(preds_low_confidence)):
        if preds[i] == "" or preds_low_confidence[i] == "":
            continue
        pred_i = preds[i].split("|")
        pred_low_conf_i = preds_low_confidence[i].split("|")
        new_p_low = []
        for p_low in pred_low_conf_i:
            overlap = False
            for p in pred_i:
                if p in p_low:
                    overlap = True
                    break
            if overlap is False:
                new_p_low.append(p_low)
        if len(new_p_low) == 0:
            preds_low_confidence[i] = ""
        else:
            preds_low_confidence[i] = "|".join(new_p_low)
    return preds_low_confidence


# https://www.kaggle.com/code/dathudeptrai/biomed-roberta-scibert-base?scriptVersionId=66513188&cellId=34
def get_filtered_models_predictions(text: pd.DataFrame, batch_size=128):

    accepted_preds = []

    test_df = text

    PARAMS = [
        (
            "pretrainedbiomedrobertabase",
            "coleridgeinitiativebiomedrobertabasev2",
            [0.5, 0.7],
            -0.1,
        ),
        (
            "scibertbasecased",
            "coleridgeinitiativescibertbasecasedv10",
            [0.5, 0.7],
            -0.7,
        ),
    ]

    for i, param in enumerate(PARAMS):
        (
            ids,
            text_ids,
            inputs,
            cosines,
            preds,
            preds_low_confidence,
            tokenizer,
        ) = end2end(
            param[0],
            param[1],
            test_df,
            ner_threshold=param[2],
            batch_size=batch_size,
        )

        preds = remove_acronym(preds)
        preds_low_confidence = remove_acronym(preds_low_confidence)
        preds_low_confidence = remove_overlap(preds, preds_low_confidence)
        accepted_preds.extend(
            get_accepted_preds(
                preds, preds_low_confidence, cosines, param[3], tokenizer
            )
        )

    return accepted_preds


def end2end(
    pretrained_path, checkpoint_path, test_df, ner_threshold=[0.5, 0.7], batch_size=128
):
    config = AutoConfig.from_pretrained(f"model1/{pretrained_path}/")
    config.output_attentions = True
    config.output_hidden_states = True

    main_model = TFAutoModel.from_config(config=config)
    model = MetricLearningModel(config=config, name="metric_learning_model")
    model.main_model = main_model
    model.K = 3

    start = time()

    # load pre-extract embedding
    # checkpoint_path = f"/kaggle/input/{checkpoint_path}"
    checkpoint_path = f"model1/{pretrained_path}/embeddings"
    all_support_embeddings = np.load(
        os.path.join(checkpoint_path, "support_embeddings.npy")
    )
    all_support_mask_embeddings = np.load(
        os.path.join(checkpoint_path, "support_mask_embeddings.npy")
    )
    all_support_nomask_embeddings = np.load(
        os.path.join(checkpoint_path, "support_nomask_embeddings.npy")
    )

    # create tokenizer and dataloader
    if "distil" in pretrained_path:
        tokenizer = DistilBertTokenizerFast.from_pretrained(
            f"model1/{pretrained_path}/"
        )
    elif "roberta" in pretrained_path:
        tokenizer = RobertaTokenizerFast.from_pretrained(f"model1/{pretrained_path}/")
    elif "scibert" in pretrained_path:
        tokenizer = BertTokenizerFast.from_pretrained(
            f"model1/{pretrained_path}/", do_lower_case=False
        )

    query_dataloader = QueryDataLoader(test_df, batch_size=batch_size)
    test_dataloader = SupportQueryDataLoader(
        test_df,
        tokenizer=tokenizer,
        batch_size=batch_size,
        is_train=False,
        training_steps=len(query_dataloader),
        query_dataloader=query_dataloader,
        return_query_ids=True,
    )

    # build model with real input and load_weights
    query_batch = test_dataloader.__getitem__(0)
    (
        query_embeddings,
        query_mask_embeddings,
        query_nomask_embeddings,
        attention_values,
    ) = model(
        [
            query_batch["input_ids"][:1, ...],
            query_batch["attention_mask"][:1, ...],
        ],
        training=True,
        sequence_labels=None,
        mask_embeddings=all_support_mask_embeddings[:1, ...],
        nomask_embeddings=all_support_nomask_embeddings[:1, ...],
    )  # [B, F]

    # model.summary()
    weights_path = glob.glob(os.path.join(checkpoint_path, "*.h5"))[0]
    model.load_weights(weights_path, by_name=True)

    # apply tf.function
    model = tf.function(model, experimental_relax_shapes=True)

    #     print("!!!!!!!!!!!!!!!!!!!! ",time()-start)  # GL
    #     start=time()

    # run inference
    ids, text_ids, inputs, cosines, preds, preds_low_confidence = run_inference(
        test_dataloader,
        model,
        all_support_embeddings,
        all_support_mask_embeddings,
        all_support_nomask_embeddings,
        ner_threshold=ner_threshold,
    )
    #     print("!!!!!!!!!!!!!!!!!!!! Model1:inference:",time()-start)  # GL
    #     start=time()

    # release model
    del_everything(model)

    return (
        ids,
        text_ids,
        inputs,
        cosines,
        preds,
        preds_low_confidence,
        test_dataloader.tokenizer,
    )


def del_everything(model):
    tf.keras.backend.clear_session()
    del model
    gc.collect()
    sess = tf.compat.v1.keras.backend.get_session()
    del sess
    graph = tf.compat.v1.get_default_graph()
    del graph


def compute_cosine_similarity(x1, x2):
    x1_norm = tf.nn.l2_normalize(x1, axis=1)
    x2_norm = tf.nn.l2_normalize(x2, axis=1)
    cosine_similarity = tf.matmul(x1_norm, x2_norm, transpose_b=True)  # [B1, B2]
    return tf.clip_by_value(cosine_similarity, -1.0, 1.0)


def run_inference(
    test_dataloader,
    model,
    all_support_embeddings,
    all_support_mask_embeddings,
    all_support_nomask_embeddings,
    ner_threshold=[0.5, 0.7],
):
    preds = []
    preds_low_confidence = []
    cosines = []
    ids = []
    text_ids = []
    inputs = []
    N_TTA = 100

    tokenizer = test_dataloader.tokenizer

    for query_batch in test_dataloader:
        all_cosines = []
        support_embeddings = all_support_embeddings[
            np.random.choice(
                range(all_support_embeddings.shape[0]),
                size=query_batch["input_ids"].shape[0] * N_TTA,
            )
        ]
        support_mask_embeddings = all_support_mask_embeddings[
            np.random.choice(
                range(all_support_mask_embeddings.shape[0]),
                size=query_batch["input_ids"].shape[0] * N_TTA,
            )
        ]
        support_nomask_embeddings = all_support_nomask_embeddings[
            np.random.choice(
                range(all_support_nomask_embeddings.shape[0]),
                size=query_batch["input_ids"].shape[0] * N_TTA,
            )
        ]
        support_mask_embeddings = np.mean(
            np.reshape(support_mask_embeddings, (-1, N_TTA, 768)), axis=1
        )
        support_nomask_embeddings = np.mean(
            np.reshape(support_nomask_embeddings, (-1, N_TTA, 768)), axis=1
        )
        (
            query_embeddings,
            query_mask_embeddings,
            query_nomask_embeddings,
            attention_values,
        ) = model(
            [
                query_batch["input_ids"],
                query_batch["attention_mask"],
            ],
            training=False,
            sequence_labels=None,
            mask_embeddings=support_mask_embeddings,
            nomask_embeddings=support_nomask_embeddings,
        )  # [B, F]

        cosine = compute_cosine_similarity(query_embeddings, support_embeddings).numpy()
        cosine = np.mean(cosine, axis=1)
        all_cosines.extend(cosine)
        ids.extend(query_batch["ids"])

        for k in range(len(all_cosines)):
            for TH in ner_threshold:
                binary_pred_at_th = attention_values.numpy()[k, :, 0] >= TH
                if np.sum(binary_pred_at_th) > 0:
                    binary_pred_at_th = binary_pred_at_th.astype(np.int32)
                    start_end = find_all_start_end(binary_pred_at_th)
                    pred_candidates = []
                    for s_e in start_end:
                        if (s_e[1] - s_e[0]) >= 4:
                            pred_tokens = list(range(s_e[0], s_e[1]))
                            pred = tokenizer.decode(
                                query_batch["input_ids"][k, ...][pred_tokens]
                            )
                            pred_candidates.append(pred)
                    pred = "|".join(pred_candidates)
                else:
                    pred = ""
                if TH == 0.7:
                    preds.append(pred)
                else:
                    preds_low_confidence.append(pred)
            cosines.append(all_cosines[k])

    return ids, text_ids, inputs, cosines, preds, preds_low_confidence


def find_all_start_end(attention_values):
    start_offset = {}
    current_idx = 0
    is_start = False
    start_end = []
    while current_idx < len(attention_values):
        if attention_values[current_idx] == 1 and is_start is False:
            start_offset[current_idx] = 0
            is_start = True
            start_idx = current_idx
        elif attention_values[current_idx] == 1 and is_start is True:
            start_offset[start_idx] += 1
        elif attention_values[current_idx] == 0 and is_start is True:
            is_start = False
        current_idx += 1
    for k, v in start_offset.items():
        start_end.append([k, k + v + 1])
    return start_end


def check_valid_low_confidence_pred(pred):
    clean_pred = clean_text(pred, True)
    keywords = [
        "study",
        "survey",
        "studies",
        "database",
        "dataset",
        "data system",
        "system data",
        "data set",
        "data base",
        "program",
    ]
    if pred != "":
        words = pred.strip().split()
        clean_words = clean_pred.strip().split()
        string_check = re.compile("[\(\)\[\]]")
        if clean_words[0] in ["a", "an", "the"]:
            return False
        if clean_words[-1] in ["a", "an", "the", "in", "on", "of", "for", "and", "or"]:
            return False
        if (
            words[0][0].isalpha()
            and words[0][0].isupper()
            and string_check.search(words[0]) is None
        ):
            for kw in keywords:
                if kw in clean_pred:
                    return True
    return False


def remove_stopwords(string):
    word_tokens = word_tokenize(string)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return " ".join(filtered_sentence).strip()


def get_accepted_preds(
    preds, preds_low_confidence, cosines, cosine_threshold, tokenizer
):
    accepted_preds = []
    ########################################################
    all_accepted_preds = []
    for i in range(len(preds)):
        if cosines[i] >= cosine_threshold:
            a = preds[i].split("|")
            unique_v = np.unique(a)
            all_accepted_preds.extend(unique_v)
        else:
            preds_low_confidence_i = preds_low_confidence[i].split("|")
            preds_low_confidence_i.extend(preds[i].split("|"))
            preds_low_confidence[i] = "|".join(preds_low_confidence_i)

    counter_all_accepted_preds = Counter(all_accepted_preds)
    for k, v in counter_all_accepted_preds.items():
        k = k.strip()
        if (
            ("#" not in k)
            and len(clean_text(k).strip().split(" ")) >= 3
            and len(k.split(" ")) >= 3
            and len(remove_stopwords(k).split(" ")) >= 3
            and len(k) >= 10
            and check_special_token(k, tokenizer)
        ):
            if v >= 4:
                accepted_preds.append(clean_text(k).strip())
            else:
                if check_valid_low_confidence_pred(k):
                    accepted_preds.append(clean_text(k).strip())

    ########################################################
    all_accepted_preds = []
    for i in range(len(preds_low_confidence)):
        if cosines[i] >= -1.0:
            a = preds_low_confidence[i].split("|")
            unique_v = np.unique(a)
            all_accepted_preds.extend(unique_v)
    counter_all_accepted_preds = Counter(all_accepted_preds)
    for k, v in counter_all_accepted_preds.items():
        k = k.strip()
        if (
            ("#" not in k)
            and len(clean_text(k).strip().split(" ")) >= 3
            and len(k.split(" ")) >= 3
            and len(remove_stopwords(k).split(" ")) >= 3
            and len(k) >= 10
            and check_special_token(k, tokenizer)
        ):
            if check_valid_low_confidence_pred(k):
                accepted_preds.append(clean_text(k).strip())

    accepted_preds = list(set(accepted_preds))
    return accepted_preds


def check_special_token(string, tokenizer):
    pad_token = tokenizer.pad_token
    sep_token = tokenizer.sep_token
    cls_token = tokenizer.cls_token

    if (
        (pad_token not in string)
        and (sep_token not in string)
        and (cls_token not in string)
    ):
        return True
    return False


def calculate_iou(se_0, se_1):
    s_0, e_0 = se_0
    s_1, e_1 = se_1
    max_s = max(s_0, s_1)
    min_e = min(e_0, e_1)
    intersection = min_e - max_s
    return intersection / ((e_0 - s_0) + (e_1 - s_1) - intersection)


def find_cased_pred(
    lower_start_idx, lower_end_idx, lower_string, cased_string, lower_pred
):
    len_lower_string = len(lower_string)
    len_cased_string = len(cased_string)
    if (len_lower_string - len_cased_string) == 0:
        return cased_string[lower_start_idx:lower_end_idx]
    else:
        diff_len = abs(len_lower_string - lower_end_idx)
        for shift_idx in range(-diff_len - 1, diff_len + 1):
            cased_pred_candidate = cased_string[
                lower_start_idx
                + shift_idx : lower_start_idx
                + shift_idx
                + len(lower_pred)
            ]
            if cased_pred_candidate.lower() == lower_pred:
                return cased_pred_candidate
    return lower_pred.upper()


def find_all_pred_in_text(normed_text_cased, all_unique_preds):
    normed_text_cased = clean_text(normed_text_cased, False)
    normed_text = normed_text_cased.lower()
    preds = []
    preds_indexs = []
    for pred in all_unique_preds:
        if (
            (" " + pred + " " in normed_text)
            or (" " + pred + "," in normed_text)
            or (" " + pred + "." in normed_text)
        ):
            preds.append(pred)
    unique_preds = []  # unique in terms of index.
    preds = list(sorted(preds, key=len))
    for pred in preds:
        matchs = re.finditer(pred, normed_text)
        for match in matchs:
            start_index = match.start()
            end_index = match.end()
            pred_cased = find_cased_pred(
                start_index, end_index, normed_text, normed_text_cased, pred
            )
            if pred_cased.islower() is False:
                preds_indexs.append([start_index, end_index])
                unique_preds.append(pred)
    group_idxs = []
    for i in range(len(preds_indexs)):
        for j in range(len(preds_indexs)):
            if i != j:
                start_i, end_i = preds_indexs[i]
                start_j, end_j = preds_indexs[j]
                iou = calculate_iou(preds_indexs[i], preds_indexs[j])
                if (
                    start_i <= end_j and end_i <= end_j and start_i >= start_j
                ) or iou >= 0.1:
                    group_idxs.append([i, j])
    unique_preds = np.array(unique_preds)
    for group_idx in group_idxs:
        unique_preds[group_idx[0]] = unique_preds[group_idx[1]]
    return np.unique(unique_preds)


def find_all_acronym_candidates(labels, raw_text):
    string = clean_text_v2(raw_text)
    all_labels = labels.split("|")

    label_with_acs = []
    for label in all_labels:
        if label != "":
            acronyms_candidates = re.findall(f"{label} \((.*?)\)", string)
            acronyms_candidates.extend(re.findall(f"{label} \[(.*?)\]", string))
            acronyms_candidates = np.unique(
                [ac for ac in acronyms_candidates if len(ac.split()) >= 1]
            )
            if len(acronyms_candidates) > 0:
                for ac in acronyms_candidates:
                    ac = find_valid_ac(label, ac)
                    if ac is not None:
                        if len(ac.split(" ")) <= 2:
                            label_with_acs.append(f"|{ac}")
            else:
                label_with_acs.append(label)

    return label_with_acs


def find_valid_ac(long_form, short_form):
    long_form = "".join([w[0] for w in long_form.split()])
    short_form_candidate1 = "".join(
        [w if i == 0 else w[0] for i, w in enumerate(short_form.split())]
    )
    short_form_candidate2 = short_form.split()[0]
    short_form_accepted = None
    original_long_index = len(long_form) - 1
    for i, short_form_candidate in enumerate(
        [short_form_candidate1, short_form_candidate2]
    ):
        long_index = len(long_form) - 1
        short_index = len(short_form_candidate) - 1

        while short_index >= 0:
            current_charactor = short_form_candidate[short_index]
            if not current_charactor.isalpha():
                short_index -= 1
                continue

            while long_form[long_index] != current_charactor:
                long_index -= 1
                if long_index < 0:
                    break

            short_index -= 1
            if long_index < 0:
                break

        if (
            long_index >= 0
            and (not short_form.isdigit())
            and long_index < original_long_index
        ):
            if i == 0:
                short_form_accepted = short_form
            else:
                short_form_accepted = short_form.split()[0]

            if not (
                short_form_accepted[-1].isalpha() or short_form_accepted[-1].isdigit()
            ):
                short_form_accepted = short_form_accepted[:-1]
            return short_form_accepted

    return short_form_accepted


def predict(text: dict):
    WIN_SIZE = 200
    SEQUENCE_LENGTH = 320

    model = Model1(win_size=WIN_SIZE, sequence_legnth=SEQUENCE_LENGTH)

    with stdout_redirector():
        predictions = model.predict(model.preprocess(text))

    return predictions


if __name__ == "__main__":
    # with open("kaggle_data/test/2f392438-e215-4169-bebf-21ac4ff253e1.json") as f:
    #     text = json.load(f)

    input_json_image = sys.argv[1]
    with open(input_json_image, "r") as f:
        text = json.load(f)
    WIN_SIZE = 200
    SEQUENCE_LENGTH = 320

    model = Model1(win_size=WIN_SIZE, sequence_legnth=SEQUENCE_LENGTH)

    with stdout_redirector():
        predictions = model.predict(model.preprocess(text))

    print(
        "Model 1 dataset candidates:",
        predictions if len(predictions) else "None found.",
    )
