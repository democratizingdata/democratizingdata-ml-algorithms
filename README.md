# Repository to train, retrain, and improve ML models for the Democratizing Data Project

## Introduction

Democratizing Data is a project to develop a method for extracting dataset
mentions from scientific papers. The project builds on the first through
third-place submissions to the [Show US The Data Kaggle
competition](https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/).

The top three submissions to the Kaggle provided applied a variety of techniques
to solve the problem. However, winning the competition and solving the problem
are two different things. Each model has a base approach that is wrapped with
some heuristics to improve performance. This repository applies the distilled
approach from each submission (without heuristics) and seeks to improve and
develop new approaches to the problem.

## Project Structure

The project is laid out in the following way:

- `/data` contains the data for training models. Currently, the files to train
  Kaggle model 2 are there as they don't have any copyright restrictions. To
  train more models you will likely need the Kaggle data, which is not publicly
  available. The Kaggle data can be downloaded from
  [here](https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/data),
  but you need to request access to the data. Additionally, models that train on
  sentence-level need the snippet data which can be generated using the Kaggle
  data and the SnippetRepository.

- `/models` is for storing trained model params. For example, the 3rd place
   Kaggle model (`democratizing_data_ml_algorithms/models/kagglemodel3.py`)
   builds a list of datasets by extracting entities from the training data with
   some rules. These extracted entities are saved in
   `/models/kagglemodel3/baseline/params.txt`. Saved model weights should be
   saved in directory with the same name as the model's python file.

- `/notebooks` contains Jupyter notebooks for exploring the data and models.

- `/democratizing_data_ml_algorithms` contains the source code for models, data,
  and evaluation. The code is laid out to emulate the cookie-cutter data science
  project structure. So to use it you need to install the code using `pip
  install -e .` from the root directory.

## Using Models

### Installation

To use already trained models install the package using pip

`pip install git+https://github.com/DemocratizingData/democratizingdata-ml-algorithms.git`

By default, this installs only the minimum dependencies. Each model may have its
own additionally required dependencies. To install the dependencies for a model
look for the extras in `setup.py`. For example, to install the extras for
`kaggle_model2` run:

`python -m pip install "git+https://github.com/DemocratizingData/democratizingdata-ml-algorithms.git#egg=democratizing_data_ml_algorithms[kaggle_model2]"`

To install all the extras run:

`python -m pip install "git+https://github.com/DemocratizingData/democratizingdata-ml-algorithms.git#egg=democratizing_data_ml_algorithms[all]"`

You may run into dependency hell doing this as models may have conflicting imports.

### Running Models

The use the models by importing them. Each model has its own configuration
passed as a dictionary to the inference method. See each models source code
for what parameters should be included in the configuration. The inference
method for each model accepts two arguments `config` and `df`. `config` is
the configuration dictionary and `df` is a pandas dataframe with at least the
column `text`. The inference method returns the same dataframe with additional
columns added that are the models outputs. Currently, the following models add
the following columns:

| Model | Output Columns |
| --- | --- |
| `generic_model1` | `model_prediction` |
| `kaggle_model2`  | `model_prediction`, `prediction_snippet`, `prediction_confidence` |
| `kaggle_model3_regex_inference` | `model_prediction`, `prediction_snippet`, `prediction_confidence` |
| `ner_model` | `model_prediction`, `prediction_snippet`, `prediction_confidence` |

- `model_prediction` is a `|`-delimited string of the predicted datasets.
- `prediction_snippet` is a `|`-delimited string of the snippets that contain
  the predicted datasets.
- `prediction_confidence` is a `|`-delimited string of the confidence scores
  for each predicted dataset.

The columns are ordered such that the 1st element of `model_prediction` corresponds
to the 1st element of `prediction_snippet` and `prediction_confidence`.


### Example

```python
import pandas as pd
import democratizing_data_ml_algorithms.models.regex_model as rm

df = pd.DataFrame({"text": ["This is a sentence with an entity in it."]})

config = {
  "regex_pattern": "",
  "keywords": ["entity"],
}

df_with_labels = rm.RegexModel(config=config).inference({}, df)
```

## CONTRIBUTING

### Adding New Models

The three baseline models approach the problem in different ways. This prevents
us from defining what a single *training sample* can/should be. So, there is
currently a `KaggleRepository` that can serve the Kaggle data and further
repositories can wrap functionality around it or develop their own repositories
completely (for example in entity classification for model 2).

New model code should be added `deomcratizing_data_ml_algorithms/models` and
should inherit from `deomcratizing_data_ml_algorithms.models.base_model.Model`.

### Formatting

Use `black` formatting.

`black .` (this will format the code for you)

### Testing

Testing is done using pytest and pytest-cov. To run the tests, run `pytest` from
the root directory: `python -m pytest`.

### TODO

- [ ] Add snippet return and confidences to `generic_model1`
- [ ] Add a heuristics class that can be run on model outputs to improve
  performance
- [ ] Add the first place submissions text segmentation method as an implementation
  of the `text_segmentizer_protocol`
- [ ] Explore fast text segmentation methods
- [ ] Increase test coverage
- [ ] Add integration and distributional shift tests
- [ ] In depth data label analysis

