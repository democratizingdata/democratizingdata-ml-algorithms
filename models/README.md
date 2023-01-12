put saved model params here.

should go in subdirs, i.e.

models/model1/run_id/params.h5


if the model you're running is going to log in `comet-ml` you need to provide
your API key by adding the following environment variable:

`export COMET_API_KEY=YOUR_KEY`

If you have `mamba` installed and would like `mlflow` to use that for environment
creation/resolution then run the following before running `mlflow`

`export MLFLOW_CONDA_CREATE_ENV_CMD=mamba`


### Running on SciServer

Generally MLflow manages a `conda` or `pyenv` environment for each project (each
subdir in `models`), however this doesn't work well with SciServer. To run well
on SciServer change the command for running a project to:

`mlflow run -e train --env-manager local models/{model_dir}/`

The flag `--env-manager local` tells mlflow to use the current python
environment, which is the SciServer environment. SciServer might be missing
dependencies for your project. If that is the case, then make a
`sciserver-requirements.txt` file in the project directory and put the
requirements in there. Then you can install them using:

`python -m pip install -r models/{model_dir}/sciserver-requirements.txt`

After your requirements are happily installed, run the previous `mlflow` command

Unless you need the multiple GPUs availble on SciServer, you should also limit
the visibility of the GPUs using:

`export CUDA_VISIBLE_DEVICES=0`

swap out zero for whatever index you want from `nvidia-smi`