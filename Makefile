setup-poetry:
	curl -sSL https://install.python-poetry.org | python3 -
dataset:
	mkdir dataset && cd dataset && \
	poetry run kaggle competitions download -c galaxy-zoo-the-galaxy-challenge && \
	  while [ "`find . -type f -name '*.zip' | wc -l`" -gt 0 ]; \
	  do find -type f -name "*.zip" -exec unzip -- '{}' \; \
	  -exec rm -- '{}' \;; done
