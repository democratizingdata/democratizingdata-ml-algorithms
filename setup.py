# https://pythonhosted.org/an_example_pypi_project/setuptools.html

import os
from setuptools import setup, find_packages


def read(fname):
    """Helper for README file."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="src",
    version="0.0.0",
    author="See AUTHORS file",
    packages=find_packages(),
    include_package_data=True,
)
