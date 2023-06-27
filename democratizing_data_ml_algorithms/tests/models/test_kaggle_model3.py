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

import democratizing_data_ml_algorithms.models.kaggle_model3 as km3


def test_train():
    pass

def test_inference():
    pass

def test_kaggle_model3_train():
    pass

def test_kaggle_model3_inference():
    pass

def test_kaggle_model3_get_parenthesis():
    text = (
        "This model was trained on the Really Great Dataset (RGD)"
        " and the Really Bad Dataset (RBD) and it went well"
        " (though not that well)."
    )
    dataset = "Really Great Dataset"

    expected = ["RGD"]

    actual = km3.KaggleModel3.get_parenthesis(text=text, dataset=dataset)

    assert actual == expected


def test_kaggle_model3_get_index():
    texts = [
        "This model was trained on the Really Great Dataset (RGD)",
        " and the Really Bad Dataset (RBD) and it went well",
        " (though not that well).",
    ]
    words = {"Really", "Great", "Dataset"}

    expected = {
        "Really": {0, 1},
        "Great": {0},
        "Dataset": {0, 1},
    }

    actual = km3.KaggleModel3.get_index(texts=texts, words=words)

    assert actual == expected

def test_kaggle_model3_tokenize():
    text = "This model was trained on the Really Great Dataset (RGD)"

    expected = [
        "This",
        "model",
        "was",
        "trained",
        "on",
        "the",
        "Really",
        "Great",
        "Dataset",
        "RGD",
    ]

    actual = km3.KaggleModel3.tokenize(text=text)

    assert actual == expected


def test_kaggle_model3_get_word():
    pass

def test_kaggle_model3_tokenized_extract():
    pass

def test_mapfilter():
    pass

def test_mapfilter_andthe():
    pass

def test_mapfilter_stopwords():
    pass

def test_mapfilter_introssai():
    pass

def test_mapfilter_introwords():
    pass

def test_mapfilter_brlessthantwowords():
    pass

def test_mapfilter_partialmatchdatasets():
    pass

def test_mapfilter_traincounts():
    pass

def test_mapfilter_brpatsub():
    pass

def test_sentencizer():
    pass

def test_dotsplitsentencizer():
    pass

