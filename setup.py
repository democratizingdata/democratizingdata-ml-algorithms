# https://pythonhosted.org/an_example_pypi_project/setuptools.html

import os
from setuptools import setup, find_packages


def read(fname):
    """Helper for README file."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="democratizing_data_ml_algorithms",
    version="0.0.1",
    author="Ryan Hausen and Contributors",
    packages=find_packages(),
    include_package_data=True,
)
