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
    #TODO: This needs to be more carefully tested.

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

    assert np.testing.assert_equal(actual, expected)




def test_color_text_figure():
    pass


def test_merge_tokens_w_classifications():
    pass


def test_is_special_token():
    pass


def test_high_probablity_token_groups():
    pass
