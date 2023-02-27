# Repository to train, retrain, and improve ML models for the Democratizing Data Project

## Introduction

Democratizing Data is a project to develop a method for extracting dataset
mentions from scientific papers. The project builds on the first through
third-place submissions to the [Show US The Data Kaggle
competition](https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data/).

The top three submissions to the Kaggle provided applied a variety of
of techniques to to solve the problem. However, winning the competition
and solving the problem are two different things. Each model has a base approach
that is wrapped with some heuristics to improve performance. This repository
applies the distilled approach from each submission (without heuristics) and 
seeks to improve and develop new approaches to the problem.

## Project Structure

The project is laid out in the following way:

- `/data` contains the data for training models. Currently, the files to train
  Kaggle model 2 are there as they don't have any copyright restrictions. To
  train more models you will likely need the Kaggle data, which is not publicly
  available. The Kaggle data can be downloaded from
  [here](https://drive.google.com/file/d/1degcu_aDsxNSXx3UPwfuYMaJM47h6XW7/view?usp=share_link),
  but you need to request access to the data. Additionally, models that train on
  sentence-level need the snippet data which can be generated using the Kaggle
  data and the SnippetRepository, or directly downloaded from
  [here](https://drive.google.com/file/d/1P3ss2D5AbSoq2P-gfDPRhdzBGWZ4da3m/view?usp=share_link).
  The current version of the snippet data is 2.7.23.

- `/models` is for storing trained model params. For example, the 3rd place
   Kaggle model (`src/models/kagglemodel3.py`) builds a list of datasets by
   extracting entities from the training data with some rules. These extracted
   entities are saved in `/models/kagglemodel3/baseline/params.txt`. Saved model
   weights should be saved in directory with the same name as the model's python
   file. The directories are designed to be run with `mlflow`.

- `/notebooks`contains notebooks for data model exploration

- `/src` contains the source code for models, data, and evaluation. The code is
  laid out to emulate the cookie-cutter data science project structure. So to
  use it you need to install the code using `pip install -e .` from the root
  dir. 

- `/stubs` contains `mypy` typing stubs (These are not currently used)


## CONTRIBUTING

### Adding New Models

The three baseline models approach the problem in different ways. This prevents
us from defining what a single *training sample* can/should be. So, there is
currently a KaggleRepository that can serve the Kaggle data and further
repositories can wrap functionality around it or develop their own repositories
completely (for example in entity classification for model 2).

New model code should be added `src/models` and should inherit from
`src.models.base_model.Model` (This class is changing as requirements become
more clear).

### Formatting/Testing

As soon as the API for the classes are stable I'll start writing tests. For
now, please use `black` formatting.

`black .` (this will format the code for you)

I'd like to use `mypy` for type checking, but it's not widely supported enough
yet. So try to use type hints and we'll build from there.

## Running on SciServer

Rather than having mlflow spin up a new conda env, use:

`--env-manager=local`