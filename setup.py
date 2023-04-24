# https://pythonhosted.org/an_example_pypi_project/setuptools.html

from setuptools import setup, find_packages

ICSR_REQUIREMENTS = [
    "imbalance-learn",
    "pandarallel",
    "pyyaml",
    "spacy",
    "thefuzz",
    "unidecode",
]

LOCAL_REQUIREMENTS = [
    "dbx",
    "imbalance-learn",
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

setup(
    name="democratizing_data_ml_algorithms",
    version="0.0.1",
    author="Ryan Hausen and Contributors",
    install_requires=ICSR_REQUIREMENTS,
    extras_require=dict(
        local=LOCAL_REQUIREMENTS,
        generic_model1=GENERIC_MODEL1_REQUIREMENTS,
        kaggle_model1=KAGGLE_MODEL1_REQUIREMENTS,
        kaggle_model2=KAGGLE_MODEL2_REQUIREMENTS,
        kaggle_model3=KAGGLE_MODEL3_REQUIREMENTS,
        ner_model=NER_MODEL_REQUIREMENTS,
        regex_model=REGEX_MODEL_REQUIREMENTS,
        schwartz_hearst_model=SCHWARTZ_HEARST_MODEL_REQUIREMENTS,
    ),
    entry_points={
        "console_scripts": [
            "icsr_inference_generic_model1 = democratizing_data_ml_algorithms.inference.generic_model1:icsr_inference",
            "icsr_inference_kaggle_model1 = democratizing_data_ml_algorithms.inference.kaggle_model1:icsr_inference",
            "icsr_inference_kaggle_model2 = democratizing_data_ml_algorithms.inference.kaggle_model2:icsr_inference",
            "icsr_inference_kaggle_model3 = democratizing_data_ml_algorithms.inference.kaggle_model3:icsr_inference",
            "icsr_inference_ner_model = democratizing_data_ml_algorithms.inference.ner_model:icsr_inference",
            "icsr_inference_regex_model = democratizing_data_ml_algorithms.inference.regex_model:icsr_inference",
        ]
    },
    packages=find_packages(),
    include_package_data=True,
)
