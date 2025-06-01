# Federative-learning for SASRec and BERT4Rec

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains federative pipeline for training SASRec and BERT4Rec. Different pairs of [Amazon Data (2014)](https://jmcauley.ucsd.edu/data/amazon/index_2014.html) datasets were used.

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

## How to use

To train a model or two models federatively, run the following command:

```bash
python train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments. Trainer can handle both usual (one domain) and federative modes (see baseline and federative configs).

To inference models, run following:

```bash
python inference.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```


## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
