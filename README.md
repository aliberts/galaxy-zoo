<p align="center">
	<a>
		<img src="misc/banner.jpg"
             width="100%">
	</a>
</p>

<p align="center">
	<a href="https://www.python.org/downloads/release/python-380/">
		<img src="https://img.shields.io/badge/python-3.10-blue"
			 alt="Python Version">
	</a>
	<a href="https://pytorch.org/">
		<img src="https://img.shields.io/badge/framework-pytorch-red"
			 alt="License">
	</a>
	<a href="https://github.com/psf/black">
		<img src="https://img.shields.io/badge/code%20style-black-000000.svg"
			 alt="Code Style">
	</a>
</p>

This project is derived from an assignement I did during my bootcamp at Yotta Academy.
It aims to classify the morphologies of distant galaxies using deep neural networks.

It is based on the Kaggle [Galaxy Zoo Challenge](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/overview).


# Documentation
A few related papers on the topic are available here:
- [1308.3496](https://arxiv.org/abs/1308.3496)
- [1807.10406](https://arxiv.org/abs/1807.10406)
- [1809.01691](https://arxiv.org/abs/1809.01691)

To better understand the task to be learned by the model, give it a go yourself: [try it here](https://www.zooniverse.org/projects/zookeeper/galaxy-zoo/classify).


# Installation

1. Ensure your gpu driver & cuda are properly setup for pytorch to use it (the name of your device should appear):
```bash
nvidia-smi
```

2. (optional) Install [poetry](https://python-poetry.org/) if you don't have it already (I highly recommend it!):
```bash
make setup-poetry
```

3. Set yourself a virtual environment with python 3.10, e.g. using [miniconda](https://docs.conda.io/en/latest/miniconda.html) (easier IMO):
```bash
conda create --yes --name gzoo python=3.10  # miniconda
conda activate gzoo
```

or [pyenv](https://github.com/pyenv/pyenv):
```bash
pyenv install 3.10:latest  # pyenv
pyenv local 3.10:latest  # run in this directory
```

4. Within this environment, install the project's dependencies:
```bash
poetry install
```

5. To download the dataset, run:
```bash
make dataset
```
This will download and extract the archives into `dataset/`. You'll need to login with [Kaggle's API](https://github.com/Kaggle/kaggle-api) first and place your `kaggle.json` api key inside `~/.kaggle` by default.

6. (optional) Make your commands shorter with this `alias`:
```bash
alias py='poetry run python'
```

7. (optional) If you intend to contribute in this repo, run:
```bash
pre-commit install
```

You're good to go!


# Training
## Create the training labels for classification
```bash
poetry run python -m gzoo.app.make_labels
```
This will produce the files `classification_labels_train_val.csv` and `classification_labels_test.csv` inside `dataset/`, which are needed for training and testing.

## Run the classification pipeline
```bash
poetry run python -m gzoo.app.train
```
script option:
- `--config_path`: specify the `.yaml` config file to read options from.
Every run config option should be listed in this file (the default file for this is [config/train.yaml](config/train.yaml)) and every option in that file can be overloaded *on the fly* at the command line.

For instance, if you are fine with the values in the `yaml` config file but you just want to change the `epochs` number, you can either change it in the config file *or* you can directly run:
```bash
poetry run python -m gzoo.app.train --compute.epochs=50
```
This will use all config values from `config/train.yaml` except the number of epochs which will be set to `50`.

main run options:
- `--compute.seed`: seed for deterministic training. (default: `None`)
- `--compute.epochs`: total number of epochs (default: `90`)
- `--compute.batch-size`: batch size (default: `128`)
- `--compute.workers`: number of data-loading threads (default: `8`)
- `--model.arch`: model architecture to be used (default: `resnet18`)
- `--model.pretrained`: use pre-trained model (default: `False`)
- `--optimizer.lr`: optimizer learning rate (default: `3.e-4` with Adam)
- `--optimizer.momentum`: optimizer momentum (for SGD only, default: `0.9`)
- `--optimizer.weight-decay`: optimizer weights regularization (L2, default `1.e-4`)


# Prediction
```bash
poetry run python -m gzoo.app.predict
```
Config works the same way as for training, default config is at [config/predict.yaml](config/predict.yaml).

A 1-image example is provided which you can run with:
```bash
poetry run python -m gzoo.app.predict --dataset.dir=example/
```


# Config
If you make changes in [gzoo.infra.config](gzoo/infra/config.py), you should also update the related `.yaml` config files in [config/](config/) with:
```bash
poetry run python -m gzoo.app.update_config
```
