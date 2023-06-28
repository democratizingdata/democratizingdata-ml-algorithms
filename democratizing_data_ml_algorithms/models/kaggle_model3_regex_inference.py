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

"""This model searches for a given set of keywords using a regular expression

Example:

    >>> import pandas as pd
    >>> import democratizing_data_ml_algorithms.models.kaggle_model3_regex_inference as kmr3
    >>> df = pd.DataFrame({"text": ["This is a sentence with an entity in it."]})
    >>> config = {
    >>>     "pretrained_model": "path/to/model_and_tokenizer",
    >>> }
    >>> model = kmr3.Kaggle3RegexInference(config)
    >>> df = rm.inference(config, df)

"""
import logging
from typing import Any, Dict, Optional

import pandas as pd

import democratizing_data_ml_algorithms.models.base_model as bm
from democratizing_data_ml_algorithms.data.repository import Repository
from democratizing_data_ml_algorithms.models.regex_model import RegexModel

logger = logging.getLogger("RegexModel")

EXPECTED_KEYS = ["model_path", "keywords"]

def train(
    repository: Repository,
    config: Dict[str, Any],
    training_logger: Optional[bm.SupportsLogging] = None,
) -> None:
    raise NotImplementedError("RegexModel does not support training")

def inference(
    config: Dict[str, Any], df: pd.DataFrame
) -> pd.DataFrame:

    bm.validate_config(config, EXPECTED_KEYS)

    model = Kaggle3RegexInference(config)

    return model.inference(config, df)

class Kaggle3RegexInference(RegexModel):
    def __init__(self, config: Dict[str, Any]):
        with open(config["model_path"], "r") as f:
            config["keywords"] = [l.strip() for l in f.readlines()]

        config["regex_pattern"] = ""
        super().__init__(config)
