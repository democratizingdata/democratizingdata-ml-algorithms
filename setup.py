# https://pythonhosted.org/an_example_pypi_project/setuptools.html

from setuptools import setup, find_packages

LOCAL_REQUIREMENTS = [
    "dbx",
    "imbalanced-learn",
    "matplotlib",
    "numpy",
    "scikit-learn",
    "pandarallel",
    "pandas",
    "pytest",
    "regex",
    "spacy",
    "thefuzz",
    "tqdm",
    "unidecode",
]

GENERIC_MODEL1_REQUIREMENTS = [
    "datasets",
    "pytorch_metric_learning",
    "torch",
    "transformers",
]

KAGGLE_MODEL1_REQUIREMENTS = [
    "datasets",
    "tensorflow",
    "transformers",
]

KAGGLE_MODEL2_REQUIREMENTS = [
    "scipy",
    "torch",
    "transformers",
]

KAGGLE_MODEL3_REQUIREMENTS = []

NER_MODEL_REQUIREMENTS = [
    "datasets",
    "spacy",
    "torch",
    "transformers",
]

REGEX_MODEL_REQUIREMENTS = []

SCHWARTZ_HEARST_MODEL_REQUIREMENTS = []

ALL = (
    GENERIC_MODEL1_REQUIREMENTS
    + KAGGLE_MODEL1_REQUIREMENTS
    + KAGGLE_MODEL2_REQUIREMENTS
    + KAGGLE_MODEL3_REQUIREMENTS
    + NER_MODEL_REQUIREMENTS
    + REGEX_MODEL_REQUIREMENTS
    + SCHWARTZ_HEARST_MODEL_REQUIREMENTS
)

DEVELOPMENT_REQUIREMENTS = [
    "black",
    "pytest",
]

setup(
    name="democratizing_data_ml_algorithms",
    version="0.0.1",
    author="Ryan Hausen and Contributors",
    install_requires=LOCAL_REQUIREMENTS,
    extras_require=dict(
        all=ALL,
        dev=DEVELOPMENT_REQUIREMENTS,
        generic_model1=GENERIC_MODEL1_REQUIREMENTS,
        kaggle_model1=KAGGLE_MODEL1_REQUIREMENTS,
        kaggle_model2=KAGGLE_MODEL2_REQUIREMENTS,
        kaggle_model3=KAGGLE_MODEL3_REQUIREMENTS,
        ner_model=NER_MODEL_REQUIREMENTS,
        regex_model=REGEX_MODEL_REQUIREMENTS,
        schwartz_hearst_model=SCHWARTZ_HEARST_MODEL_REQUIREMENTS,
    ),
    packages=find_packages(),
    include_package_data=True,
)
