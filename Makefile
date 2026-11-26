.PHONY: install all stats train monitor score test lint mlflow-ui clean

install:
	pip install -e ".[tracking,dev]"

all:            ## full pipeline: download -> stats -> train -> decision -> monitoring -> MLflow
	python -m churn.pipeline all

stats:
	python -m churn.pipeline stats

train:
	python -m churn.pipeline train

monitor:
	python -m churn.pipeline monitor

score:          ## make score INPUT=new_customers.csv
	python -m churn.pipeline score --input $(INPUT) --output scored.csv

test:
	pytest -q

lint:
	ruff check src tests

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

clean:
	rm -rf models mlruns mlartifacts mlflow.db scored.csv .pytest_cache
