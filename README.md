# Repository to train, retrain, and improve ML models for the Democratizing Data Project


To train some of the models you will need the original kaggle data plus the
kaggle validation data, which can be accessed
[here](https://drive.google.com/file/d/1degcu_aDsxNSXx3UPwfuYMaJM47h6XW7/view?usp=share_link)
(The data is not publicly available, so you need to request access to the linked
data). The downloaded file `kaggle.zip` needs to be extracted into the `data`
folder.


The project is laid out in the following way:

- `/data` contains the data for training models. You will add the kaggle data
here. Currently, the files to train Kaggle model 2 are there.

- `/models` is for storing trained model params. For example, the 3rd place
Kaggle model (`src/models/kagglemodel3.py`) builds a list of datasets by
extracting entities from the training data with some rules. These extracted
entities are saved in `/models/kagglemodel3/baseline/params.txt`. Saved model
weights should be saved in directory with the same name as the model's python
file.

- `/notebooks`contains notebooks for data model exploration

- `/src` contains the source code for models, data, and evaluation

- `/stubs` contains `mypy` typing stubs




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
now, please use `black` formatting and `mypy` typing. To check your code run
the following:

`black .` (this will format the code for you)

`MYPYPATH=stubs mypy src/` (fix any errors you get, ignore notes/warnings)


