setup-poetry:
	curl -sSL https://install.python-poetry.org | python3 -
download-dataset:
	poetry run kaggle competitions download -c galaxy-zoo-the-galaxy-challenge
