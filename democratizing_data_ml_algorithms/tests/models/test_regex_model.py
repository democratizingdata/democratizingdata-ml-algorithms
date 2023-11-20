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

import regex as re
import pytest

import democratizing_data_ml_algorithms.tests.utils as utils

import democratizing_data_ml_algorithms.models.regex_model as regex_model


def test_train_fails():
    with pytest.raises(NotImplementedError):
        regex_model.train(None, None, None)


def test_inference():
    sample_df = utils.get_trivial_sample_dataframe()

    outs = regex_model.inference({}, sample_df)

    assert outs["model_prediction"].values[0] == "Really Great Dataset"


def test_regex_model_inference():
    sample_df = utils.get_trivial_sample_dataframe()

    outs = regex_model.RegexModel({}).inference({}, sample_df)

    assert outs["model_prediction"].values[0] == "Really Great Dataset"


def test_regex_model_train_fails():
    with pytest.raises(NotImplementedError):
        model = regex_model.RegexModel({})
        model.train(None, None, None)


def test_regex_model_init_default():
    config = {}
    model = regex_model.RegexModel(config)
    assert model.entity_pattern.pattern == regex_model.ENTITY_PATTERN


def test_regex_model_init_custom_regex():
    config = {"regex_pattern": r"(\b[A-Z][a-z]+\b)"}
    model = regex_model.RegexModel(config)
    assert model.entity_pattern.pattern == config["regex_pattern"]


def test_regex_model_init_custom_keywords():
    config = {"regex_pattern": "", "keywords": ["foo", "Bar", "USA", "Data Set"]}
    model = regex_model.RegexModel(config)

    expected = r"([F|f][O|o][O|o]|[B|b][A|a][R|r]|USA|[D|d]ata [S|s]et)"

    assert model.entity_pattern.pattern == expected


def test_regex_model_init_custom_keywords_with_regex():
    config = {
        "regex_pattern": r"(\b[A-Z][a-z]+\b)",
        "keywords": ["foo", "Bar", "USA", "Data Set"],
    }
    model = regex_model.RegexModel(config)

    expected = (
        r"(\b[A-Z][a-z]+\b|([F|f][O|o][O|o]|[B|b][A|a][R|r]|USA|[D|d]ata [S|s]et))"
    )

    assert model.entity_pattern.pattern == expected


def test_regex_model_extract_context():
    in_text = "This is a test sentence. This is the target sentence. This is not."
    expected_out = "This is the target sentence."

    match = re.search("target", in_text)
    assert regex_model.RegexModel.extract_context(in_text, match) == expected_out


def test_regex_model_regixfy_char():
    assert regex_model.RegexModel.regexify_char("a") == r"[A|a]"
    assert regex_model.RegexModel.regexify_char("A") == r"[A|a]"
    assert regex_model.RegexModel.regexify_char("1") == r"1"


def test_regex_model_regexify_first_char():
    assert regex_model.RegexModel.regexify_first_char("a") == r"[A|a]"
    assert regex_model.RegexModel.regexify_first_char("Apple") == r"[A|a]pple"


def test_regex_model_regexify_keyword():
    assert regex_model.RegexModel.regexify_keyword("USA") == r"USA"
    assert regex_model.RegexModel.regexify_keyword("(USA)") == r"\(USA\)"
    assert (
        regex_model.RegexModel.regexify_keyword("apple") == r"[A|a][P|p][P|p][L|l][E|e]"
    )
    assert regex_model.RegexModel.regexify_keyword("Foo Bar") == r"[F|f]oo [B|b]ar"
