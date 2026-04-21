.PHONY: install lint type-check security test train serve eval docker-build docker-run clean

install:
	pip install -e ".[dev,eval,monitoring]"

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

lint-fix:
	ruff check --fix src/ tests/
	ruff format src/ tests/

type-check:
	mypy src/ --ignore-missing-imports

security:
	bandit -r src/ -ll

test:
	pytest tests/ --cov=src --cov-report=html --cov-fail-under=60

test-fast:
	pytest tests/ -x -q --no-header

train:
	python src/models/train.py

serve:
	uvicorn src.serving.app:app --reload --host 0.0.0.0 --port 8000

eval:
	python evaluation/ragas_eval.py --golden-set data/golden_set/golden_pairs.yaml

judge:
	python evaluation/llm_judge.py

docker-build:
	docker build -t fraud-api:local src/serving/

docker-run:
	docker compose up

clean:
	rm -rf .pytest_cache .mypy_cache __pycache__ htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
