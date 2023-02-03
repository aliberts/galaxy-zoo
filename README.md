<p align="center">
	<a>
		<img src="misc/banner.jpg"
             width="100%">
	</a>
</p>

<p align="center">
	<a href="https://www.python.org/downloads/release/python-3100/">
		<img src="https://img.shields.io/badge/Python-3.10-blue"
			 alt="Python Version">
	</a>
	<a href="https://pytorch.org/">
		<img src="https://img.shields.io/badge/Framework-PyTorch-red"
			 alt="License">
	</a>
	<a href="https://github.com/psf/black">
		<img src="https://img.shields.io/badge/Code%20style-Black-000000.svg"
			 alt="Code Style">
	</a>
	<a href="https://wandb.ai/aliberts/galaxy-zoo?workspace=user-aliberts">
		<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28-gray.svg"
			 alt="Weights & Biases"
			 height="20">
	</a>
</p>

This project is derived from an assignement I did during my bootcamp at Yotta Academy.
It aims to classify the morphologies of distant galaxies using deep neural networks.

It is based on the [Kaggle Galaxy Zoo Challenge](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/overview).

Originaly posed as a regression problem in the Kaggle challenge, with formulate it here as a multiclass classification problem since this is eventually the goal behind the project. Additionaly, this has the added benefit to simplify things a bit.

To better understand the task to be learned by the model, give it a go yourself: [try it here](https://www.zooniverse.org/projects/zookeeper/galaxy-zoo/classify).


# Project & Results
Checkout my experiments and the project's report on [Weights & Biases](https://wandb.ai/aliberts/galaxy-zoo?workspace=user-aliberts).


# Documentation
A few related papers on the topic are available here:
- [1308.3496](https://arxiv.org/abs/1308.3496)
- [1807.10406](https://arxiv.org/abs/1807.10406)
- [1809.01691](https://arxiv.org/abs/1809.01691)


# Installation

#### Step 1
Ensure your gpu driver & cuda are properly setup for pytorch to use it (the name of your device should appear):
```bash
nvidia-smi
```

#### Step 2
If you don't have it already — I highly recommend it! — install [poetry](https://python-poetry.org/):
```bash
make setup-poetry
```

#### Step 3
Setup the environment with python 3.10, e.g. using [miniconda](https://docs.conda.io/en/latest/miniconda.html) (easier IMO):
```bash
git clone git@github.com:aliberts/galaxy-zoo.git
cd galaxy-zoo
conda create --yes --name gzoo python=3.10
conda activate gzoo
poetry install
```

or [pyenv](https://github.com/pyenv/pyenv):
```bash
git clone git@github.com:aliberts/galaxy-zoo.git
cd galaxy-zoo
pyenv install 3.10:latest
pyenv local 3.10:latest
poetry install
```

#### Step 4
Download the dataset:
```bash
make dataset
```
This will download and extract the archives into `dataset/`. You'll need to login with [Kaggle's API](https://github.com/Kaggle/kaggle-api#api-credentials) first and place your `kaggle.json` api key inside `~/.kaggle` by default. \
You can also do it manually by downloading it [here](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data). In that case, don't forget to update the location of the directory you put it in with the `dataset.dir` [config](config/train.yaml) option.


#### Optional
Make your commands shorter with this `alias`:
```bash
alias py='poetry run python'
```

If you intend to contribute in this repo, install the pre-commit hooks with:
```bash
pre-commit install
```

You're good to go!


# Training
### Create the training labels for classification
```bash
poetry run python -m gzoo.app.make_labels
```
This will produce the `classification_labels.csv` file inside `dataset/`, which is needed for training. These class labels are produced from the original regression labels in `training_solutions_rev1.csv`.

### Run the classification pipeline
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
