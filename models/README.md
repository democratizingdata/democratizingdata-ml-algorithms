put saved model params here.

should go in subdirs, i.e.

models/model1/run_id/params.h5


if the model you're running is going to log in `comet-ml` you need to provide
your API key by adding the following environment variable:

`export COMET_API_KEY=YOUR_KEY`

If you have `mamba` installed and would like `mlflow` to use that for environment
creation/resolution then run the following before running `mlflow`

`export MLFLOW_CONDA_CREATE_ENV_CMD=mamba`

