.PHONY: install lint lint-fix type-check security test test-fast train serve \
        eval judge benchmark docker-build docker-run demo demo-gpu clean help

# ── Dev ───────────────────────────────────────────────────────────────────────

install: ## Instala todas as dependências (dev + eval + monitoring)
	pip install -e ".[dev,eval,monitoring]"

lint: ## Verifica lint (ruff)
	ruff check src/ tests/ evaluation/
	ruff format --check src/ tests/ evaluation/

lint-fix: ## Corrige lint automaticamente
	ruff check --fix src/ tests/ evaluation/
	ruff format src/ tests/ evaluation/

type-check: ## Type checking com mypy
	mypy src/ --ignore-missing-imports

security: ## Security scan com bandit
	bandit -r src/ -ll

test: ## Testes com coverage >= 40%
	pytest tests/ --cov=src --cov-report=html --cov-fail-under=40

test-fast: ## Testes rápidos sem coverage
	pytest tests/ -x -q --no-header --no-cov

# ── ML ────────────────────────────────────────────────────────────────────────

train: ## Treina modelo RF + MLP (loga no MLflow)
	python scripts/train.py

# ── Avaliação ─────────────────────────────────────────────────────────────────

eval: ## Avaliação RAGAS (4 métricas, golden set 20 pares)
	python evaluation/ragas_eval.py

judge: ## LLM-as-judge (5 critérios de negócio)
	python evaluation/llm_judge.py

benchmark: ## Benchmark ≥3 configs LLM — Etapa 2 checklist
	python evaluation/benchmark_llm.py

# ── Serving ───────────────────────────────────────────────────────────────────

serve: ## Sobe API local (hot reload)
	uvicorn src.serving.app:app --reload --host 0.0.0.0 --port 8000

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build: ## Build da imagem fraud-api
	docker build -t fraud-api:local -f src/serving/Dockerfile .

docker-run: ## Sobe stack completa (sem GPU)
	docker compose up

# ── Demo ─── UMA LINHA QUE SOBE TUDO ─────────────────────────────────────────

demo: ## DEMO COMPLETO: deps + Docker + treino + smoke tests (sem GPU)
	@echo ""
	@echo "================================================="
	@echo "  Datathon Grupo 42 — Demo Completo"
	@echo "================================================="
	@[ -f .env ] || (cp .env.example .env && echo "AVISO: .env criado — preencha as chaves se necessário")
	pip install -e ".[dev,eval,monitoring]" -q
	docker compose up -d mlflow prometheus grafana
	@echo "Aguardando MLflow (máx 60s)..."
	@for i in $$(seq 1 20); do \
		curl -sf http://localhost:5000/api/2.0/mlflow/experiments/list > /dev/null 2>&1 && break; \
		sleep 3; echo "  aguardando..."; \
	done
	python scripts/train.py
	docker compose up -d fraud-api
	@echo "Aguardando API (máx 60s)..."
	@for i in $$(seq 1 20); do \
		curl -sf http://localhost:8000/health > /dev/null 2>&1 && break; \
		sleep 3; echo "  aguardando..."; \
	done
	python scripts/smoke_api.py
	@echo ""
	@echo "================================================="
	@echo "  Stack pronta! Acesse:"
	@echo "  API:       http://localhost:8000/docs"
	@echo "  MLflow:    http://localhost:5000"
	@echo "  Grafana:   http://localhost:3000  (admin/datathon42)"
	@echo "  Prometheus: http://localhost:9090"
	@echo "================================================="

demo-gpu: ## DEMO COM vLLM (GPU NVIDIA obrigatória)
	@echo ""
	@echo "================================================="
	@echo "  Demo com vLLM local (GPU mode)"
	@echo "================================================="
	@[ -f .env ] || (cp .env.example .env && echo "AVISO: preencha HF_TOKEN no .env")
	pip install -e ".[dev,eval,monitoring]" -q
	docker compose --profile gpu up -d
	@echo "Aguardando vLLM (pode levar 2-5 min para baixar o modelo)..."
	@for i in $$(seq 1 60); do \
		curl -sf http://localhost:8080/health > /dev/null 2>&1 && break; \
		sleep 5; echo "  aguardando vLLM..."; \
	done
	LLM_PROVIDER=vllm docker compose up -d fraud-api
	python scripts/train.py
	python scripts/smoke_api.py
	@echo ""
	@echo "  vLLM API:  http://localhost:8080/v1"
	@echo "  API:       http://localhost:8000/docs"
	@echo "  MLflow:    http://localhost:5000"

# ── Limpeza ───────────────────────────────────────────────────────────────────

clean: ## Remove artefatos de build e cache
	rm -rf .pytest_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

help: ## Exibe esta ajuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
