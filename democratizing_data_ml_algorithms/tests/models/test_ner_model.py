# BSD 3-Clause License

# Copyright (c) 2023, AUTHORS
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Any, Dict

import numpy as np
import pytest
import torch
import transformers as tfs
from transformers import AutoModelForTokenClassification, AutoTokenizer

import democratizing_data_ml_algorithms.models.base_model as bm
from democratizing_data_ml_algorithms.models.spacy_default_segementizer import (
    DefaultSpacySegmentizer,
)
import democratizing_data_ml_algorithms.tests.utils as utils
import democratizing_data_ml_algorithms.models.ner_model as ner_model


def test_train():
    pass


def test_inference():
    pass


def test_convert_ner_tags_to_ids():
    lbl_to_id = {
        "O": 0,
        "B-DAT": 1,
        "I-DAT": 2,
    }

    inputs = [
        ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O"],
        ["B-DAT", "I-DAT", "I-DAT", "O", "O", "B-DAT", "O", "O", "O", "O"],
    ]

    expeted_outputs = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 2, 2, 0, 0, 1, 0, 0, 0, 0],
    ]

    assert expeted_outputs == ner_model.convert_ner_tags_to_ids(lbl_to_id, inputs)


def test_convert_sample_ner_tags_to_ids():
    lbl_to_id = {
        "O": 0,
        "B-DAT": 1,
        "I-DAT": 2,
    }

    sample = dict(
        labels=[
            ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O"],
            ["B-DAT", "I-DAT", "I-DAT", "O", "O", "B-DAT", "O", "O", "O", "O"],
        ],
    )

    expected = dict(
        labels=[
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 2, 2, 0, 0, 1, 0, 0, 0, 0],
        ],
    )

    assert expected == ner_model.convert_sample_ner_tags_to_ids(lbl_to_id, sample)


def test_tokenize_and_align_labels():
    # TODO: This needs to be more carefully tested.

    sample = dict(
        text=[
            "In this work we perform science.",
            "We used the Really Great Dataset for our data.",
        ],
        labels=[
            ["O", "O", "O", "O", "O", "O"],
            ["O", "O", "O", "B-DAT", "I-DAT", "I-DAT", "0", "0", "0"],
        ],
    )

    actual = ner_model.tokenize_and_align_labels(utils.mock_tokenize_f, sample).labels
    expected = sample["labels"]

    assert actual == expected


def test_prepare_batch():
    pass


def test_lbl_to_color():
    label = [1.0, 0.0, 0.0]

    actual = ner_model.lbl_to_color(label).astype(np.float32)

    expected = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    print(actual, expected)

    np.testing.assert_equal(actual, expected)


def test_color_text_figure():
    pass


def test_merge_tokens_w_classifications():

    tokens = ["This", "is", "a", "data", "##set", "."]
    classifications = [0, 0, 0, 1, 1, 0]

    actual = ner_model.merge_tokens_w_classifications(tokens, classifications)

    expected = [
        ("This", 0),
        ("is", 0),
        ("a", 0),
        ("dataset", 1),
        (".", 0),
    ]

    assert actual == expected


def test_is_special_token():
    assert ner_model.is_special_token("[CLS]") == True
    assert ner_model.is_special_token("[SEP]") == True
    assert ner_model.is_special_token("[PAD]") == True
    assert ner_model.is_special_token("[UNK]") == True
    assert ner_model.is_special_token("[MASK]") == True
    assert ner_model.is_special_token("This") == False


def test_high_probablity_token_groups_thresholds():

    token_classifications = [
        ("We", 0.0),
        ("used", 0.0),
        ("the", 0.0),
        ("Really", 0.7),
        ("Great", 0.8),
        ("Dataset", 0.9),
        ("for", 0.0),
        ("our", 0.0),
        ("study", 0.0),
        (".", 0.0),
    ]

    actual_05 = ner_model.high_probablity_token_groups(token_classifications, 0.5)
    actual_06 = ner_model.high_probablity_token_groups(token_classifications, 0.6)
    actual_07 = ner_model.high_probablity_token_groups(token_classifications, 0.7)
    actual_08 = ner_model.high_probablity_token_groups(token_classifications, 0.8)
    actual_09 = ner_model.high_probablity_token_groups(token_classifications, 0.9)
    actual_10 = ner_model.high_probablity_token_groups(token_classifications, 1.0)

    expected_05 = [[("Really", 0.7), ("Great", 0.8), ("Dataset", 0.9)]]
    expected_06 = [[("Really", 0.7), ("Great", 0.8), ("Dataset", 0.9)]]
    expected_07 = [[("Really", 0.7), ("Great", 0.8), ("Dataset", 0.9)]]
    expected_08 = [[("Great", 0.8), ("Dataset", 0.9)]]
    expected_09 = [[("Dataset", 0.9)]]
    expected_10 = []

    assert actual_05 == expected_05
    assert actual_06 == expected_06
    assert actual_07 == expected_07
    assert actual_08 == expected_08
    assert actual_09 == expected_09
    assert actual_10 == expected_10


def test_high_probablity_token_groups_mulitple_groups():

    token_classifications = [
        ("We", 0.0),
        ("used", 0.0),
        ("the", 0.0),
        ("Really", 1.0),
        ("Great", 1.0),
        ("Dataset", 1.0),
        ("for", 0.0),
        ("our", 0.0),
        ("study", 0.0),
        (".", 0.0),
        ("We", 0.0),
        ("used", 0.0),
        ("the", 0.0),
        ("Really", 1.0),
        ("Great", 1.0),
        ("Dataset", 1.0),
        ("Again", 1.0),
        ("for", 0.0),
        ("our", 0.0),
        ("study", 0.0),
        (".", 0.0),
    ]

    expected = [
        [("Really", 1.0), ("Great", 1.0), ("Dataset", 1.0)],
        [("Really", 1.0), ("Great", 1.0), ("Dataset", 1.0), ("Again", 1.0)],
    ]

    actual = ner_model.high_probablity_token_groups(token_classifications, 0.5)

    assert actual == expected


@pytest.mark.uses_model_params
def test_ner_model_pytorch_get_model_objects():

    include_optimizer = False
    config = dict(
        model_tokenizer_name="distilbert-base-uncased",
    )

    actual = ner_model.NERModel_pytorch().get_model_objects(config, include_optimizer)

    expected_tokenizer = AutoTokenizer.from_pretrained(config["model_tokenizer_name"])
    expected = (
        AutoModelForTokenClassification.from_pretrained(
            config["model_tokenizer_name"],
            num_labels=len(ner_model.NERModel_pytorch.lbl_to_id),
            id2label=ner_model.NERModel_pytorch.id_to_lbl,
            label2id=ner_model.NERModel_pytorch.lbl_to_id,
        ),
        expected_tokenizer,
        tfs.data.data_collator.DataCollatorForTokenClassification(
            expected_tokenizer,
            return_tensors="pt",
        ),
    )

    assert list(map(type, actual)) == list(map(type, expected))


@pytest.mark.uses_model_params
def test_ner_model_pytorch_get_model_objects_with_optimizer():

    include_optimizer = True
    config = dict(
        model_tokenizer_name="distilbert-base-uncased",
        optimizer="torch.optim.AdamW",
    )

    actual = ner_model.NERModel_pytorch().get_model_objects(config, include_optimizer)

    expected_model = AutoModelForTokenClassification.from_pretrained(
        config["model_tokenizer_name"],
        num_labels=len(ner_model.NERModel_pytorch.lbl_to_id),
        id2label=ner_model.NERModel_pytorch.id_to_lbl,
        label2id=ner_model.NERModel_pytorch.lbl_to_id,
    )
    expected_tokenizer = AutoTokenizer.from_pretrained(config["model_tokenizer_name"])
    expected = (
        expected_model,
        expected_tokenizer,
        tfs.data.data_collator.DataCollatorForTokenClassification(
            expected_tokenizer,
            return_tensors="pt",
        ),
        torch.optim.AdamW(expected_model.parameters()),
        bm.MockLRScheduler(),
    )

    assert list(map(type, actual)) == list(map(type, expected))


@pytest.mark.uses_model_params
def test_ner_model_pytorch_get_model_objects_with_optimizer_with_scheduler():
    include_optimizer = True
    config = dict(
        model_tokenizer_name="distilbert-base-uncased",
        optimizer="torch.optim.AdamW",
        scheduler="torch.optim.lr_scheduler.ConstantLR",
    )

    actual = ner_model.NERModel_pytorch().get_model_objects(config, include_optimizer)

    expected_model = AutoModelForTokenClassification.from_pretrained(
        config["model_tokenizer_name"],
        num_labels=len(ner_model.NERModel_pytorch.lbl_to_id),
        id2label=ner_model.NERModel_pytorch.id_to_lbl,
        label2id=ner_model.NERModel_pytorch.lbl_to_id,
    )
    expected_optimizer = torch.optim.AdamW(expected_model.parameters())
    expected_tokenizer = AutoTokenizer.from_pretrained(config["model_tokenizer_name"])
    expected = (
        expected_model,
        expected_tokenizer,
        tfs.data.data_collator.DataCollatorForTokenClassification(
            expected_tokenizer,
            return_tensors="pt",
        ),
        expected_optimizer,
        torch.optim.lr_scheduler.ConstantLR(expected_optimizer),
    )

    assert list(map(type, actual)) == list(map(type, expected))


def test_ner_model_pytorch_resolve_segmentizer_config_str():
    config = dict(
        segmentizer="democratizing_data_ml_algorithms.models.spacy_default_segementizer.DefaultSpacySegmentizer",
    )

    actual = ner_model.NERModel_pytorch().resolve_segmentizer(config)

    expected = DefaultSpacySegmentizer()

    assert type(actual) == type(expected)


def test_ner_model_pytorch_resolve_segmentizer_config_class():
    config = dict(
        segmentizer=DefaultSpacySegmentizer,
    )

    actual = ner_model.NERModel_pytorch().resolve_segmentizer(config)

    expected = DefaultSpacySegmentizer()

    assert type(actual) == type(expected)


def test_ner_model_pytorch_resolve_segmentizer_default():
    config = dict()
    actual = ner_model.NERModel_pytorch().resolve_segmentizer(config)
    expected = DefaultSpacySegmentizer()

    assert type(actual) == type(expected)


def test_ner_model_pytorch_inference():
    pass


def test_ner_model_pytorch_filter_by_idx():
    pass


def test_ner_model_pytorch_train():
    pass
