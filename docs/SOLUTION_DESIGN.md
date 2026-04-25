# Solution Design — Datathon Fase 5
### Modernização MLOps — Detecção de Fraude + Saúde Bancária
**Grupo 42 | FIAP MLET Pós-Tech**

---

## Parte 1 — Contexto e Objetivos

### 1.1 Situação Atual (AS-IS)

A instituição financeira opera com:
- Modelo de Detecção de Fraude em produção no Databricks (batch inference)
- Feature Store sem governança (overwrite destrutivo)
- Notebooks como ponto único de falha em produção
- Ausência de drift detection sistemático
- Zero rastreabilidade de experimentos
- Sem documentação de compliance LGPD/BACEN
- **Maturidade MLOps: Nível 0**

### 1.2 Estado Alvo (TO-BE)

Após a modernização:
- Feature Store com upsert incremental e auditabilidade total
- Databricks Jobs isolados (sem notebook SPOF)
- MLflow Tracking + Registry com gate de promoção automático
- ReAct Agent para consultas de Saúde Bancária com ≥5 ferramentas
- Pipeline RAG sobre base de conhecimento interna
- RAGAS + LLM-as-judge para avaliação contínua
- Drift detection automático com alertas Grafana
- Guardrails OWASP com red team comprovado
- CI/CD completo com cobertura ≥60%
- **Maturidade MLOps: Nível 2**

### 1.3 Stack Tecnológica

| Camada | Tecnologia | Finalidade |
|---|---|---|
| Dados | Delta Lake + DVC | Versionamento e auditabilidade |
| Feature Store | Databricks Feature Store | Materialização incremental |
| ML Training | Scikit-Learn + PyTorch | Baseline e deep learning |
| Experiment Tracking | MLflow | Parâmetros, métricas, artefatos |
| Model Serving | FastAPI + AKS | Inferência em tempo real |
| LLM | Anthropic (Haiku/Sonnet) | Agent + RAG + Judge |
| Vector Store | FAISS | Embeddings de documentos |
| Observabilidade | Prometheus + Grafana + Langfuse | Métricas e traces |
| Drift Detection | Evidently | PSI + Chi-squared |
| Segurança | Guardrails custom + spaCy | OWASP + PII |
| CI/CD | GitHub Actions | Automação de qualidade |

---

## Parte 2 — Arquitetura de Dados

### 2.1 Medallion Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  DELTA LAKE — MEDALLION                     │
│                                                             │
│  BRONZE (Raw)          SILVER (Cleaned)     GOLD (Features) │
│  ────────────          ────────────────     ─────────────── │
│  transactions.csv  ──► transactions_clean ──► features_ml   │
│  (imutável, DVC)       (schema validated)    (upsert only)  │
│                                                             │
│  Retenção: 7 anos      Retenção: 3 anos      Rolling 90d    │
│  Nenhum delete         Anonimização PII       Versionado    │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Feature Engineering Pipeline

**Features calculadas a partir do dataset creditcardfraud:**

```python
# Grupo 1: Features temporais
hour_of_day: int           # Hora da transação (0-23)
is_weekend: bool           # Final de semana
time_since_last_tx: float  # Segundos desde última transação do cliente

# Grupo 2: Features de valor
amount_zscore: float       # Z-score do valor vs. histórico do cliente
log_amount: float          # log1p(Amount) para normalização
amount_to_avg_ratio: float # Amount / média histórica do cliente

# Grupo 3: Features originais (PCA anonimizado)
V1..V28: float             # Mantidas do dataset original

# Grupo 4: Features de comportamento (para enriquecimento)
tx_count_1h: int           # Transações na última hora
tx_count_24h: int          # Transações nas últimas 24h
unique_merchants_7d: int   # Comerciantes distintos em 7 dias
```

### 2.3 DVC Pipeline (dvc.yaml)

```
prepare     → Baixa e valida dados brutos
featurize   → Executa feature_engineering.py
train       → Executa train.py (MLflow tracking)
evaluate    → Calcula métricas no test set
```

Cada stage rastreado: inputs, outputs, parâmetros. `dvc repro` reproduz experimento completo.

---

## Parte 3 — Modelos de Machine Learning

### 3.1 Baseline — Random Forest (Scikit-Learn)

```python
# Configuração padrão (configs/model_config.yaml)
RandomForestClassifier:
  n_estimators: 200
  max_depth: 15
  class_weight: "balanced"    # Lida com desbalanceamento 99.8%/0.2%
  min_samples_leaf: 10
  random_state: 42

# Threshold de decisão calibrado por Precision-Recall
decision_threshold: 0.35      # Otimizado para F1 em dados desbalanceados
```

### 3.2 Baseline — MLP (PyTorch)

```python
# Arquitetura
MLPFraudDetector:
  input_dim: 31         # V1-V28 + Amount + hour + log_amount
  hidden_layers: [128, 64, 32]
  dropout: 0.3
  activation: ReLU
  output: sigmoid (probabilidade de fraude)

# Treinamento
optimizer: Adam(lr=0.001)
loss: BCEWithLogitsLoss(pos_weight=577)  # 577 = ratio negativo/positivo
epochs: 50
batch_size: 1024
early_stopping: patience=5
```

### 3.3 MLflow Tracking — O que é Logado

```python
mlflow.log_params({
    "model_type": "random_forest",
    "n_estimators": 200,
    "class_weight": "balanced",
    "dataset_version": dvc_hash,
    "git_sha": git_sha,
})

mlflow.log_metrics({
    "auc_roc": 0.9743,
    "f1_score": 0.8821,
    "precision": 0.8934,
    "recall": 0.8712,
    "ks_statistic": 0.7821,
    "average_precision": 0.8654,
})

mlflow.log_artifacts([
    "confusion_matrix.png",
    "roc_curve.png",
    "shap_summary.html",
    "classification_report.txt",
])

mlflow.set_tags({
    "champion": "false",
    "environment": "staging",
    "approved_by": None,
})
```

### 3.4 Champion-Challenger e Promoção

```
Novo modelo (Challenger) treinado
    │
    ▼ Comparação automática
delta_AUC = AUC_challenger - AUC_champion
    │
    ├── delta_AUC >= 0.005?
    │       YES → MLflow: tag challenger como "candidate"
    │              → Notificação Slack para aprovação humana
    │              → Aprovação manual → tag "champion", deploy AKS
    │
    └── delta_AUC < 0.005?
            → MLflow: tag "rejected", log motivo
            → Champion permanece em produção
```

---

## Parte 4 — Motor GenAI e Agente ReAct

### 4.1 Arquitetura do Agente

```python
# Configuração do Agente ReAct
class BankHealthAgent:
    model: "claude-sonnet-4-6"      # Raciocínio e síntese
    router_model: "claude-haiku-4-5-20251001"  # Seleção de ferramenta
    max_iterations: 10
    system_prompt: """
        Você é um especialista em saúde bancária e detecção de fraude.
        Analise dados com precisão. Cite fontes. Nunca exponha PII.
        Explique decisões de forma clara para stakeholders não-técnicos.
        Base legal para dados: LGPD Art. 7-IX (interesse legítimo).
    """
    tools: [
        FraudMetricsLookup,
        DriftStatusChecker,
        FeatureLookup,
        KnowledgeBaseSearch,  # RAG integrado
        AlertHistoryQuery,
    ]
```

### 4.2 Ferramentas do Agente (5 obrigatórias)

**FraudMetricsLookup:**
```python
Input:  FraudMetricsInput(period: str = "24h")
Output: FraudMetricsResult(
    auc_roc: float,
    f1_score: float,
    fraud_rate: float,
    total_transactions: int,
    blocked_transactions: int,
    last_updated: datetime,
)
```

**DriftStatusChecker:**
```python
Input:  DriftInput(features: list[str] | None = None)
Output: DriftResult(
    has_drift: bool,
    alerts: list[DriftAlert],  # feature, psi_score, severity
    overall_status: Literal["OK", "WARNING", "CRITICAL"],
    checked_at: datetime,
)
```

**FeatureLookup:**
```python
Input:  FeatureInput(transaction_id: str)
Output: FeatureResult(
    transaction_id: str,
    features: dict[str, float],
    fraud_probability: float,
    decision: Literal["APPROVED", "BLOCKED", "REVIEW"],
    shap_top_features: list[tuple[str, float]],  # (feature, shap_value)
)
```

**KnowledgeBaseSearch (RAG):**
```python
Input:  KBInput(query: str, top_k: int = 3)
Output: KBResult(
    chunks: list[KBChunk],  # content, source, relevance_score
    answer_context: str,    # chunks concatenados para o LLM
)
```

**AlertHistoryQuery:**
```python
Input:  AlertInput(hours: int = 24, severity: str | None = None)
Output: AlertResult(
    alerts: list[Alert],  # type, severity, message, timestamp
    summary: str,
    critical_count: int,
)
```

### 4.3 Pipeline RAG

```
Documentos de entrada:
├── docs/MODEL_CARD.md
├── docs/SYSTEM_CARD.md
├── docs/LGPD_PLAN.md
└── data/golden_set/knowledge_base/*.txt

         │ DocumentLoader
         ▼
    TextSplitter (chunk_size=500, overlap=50)
         │
         ▼
    Embeddings: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
         │
         ▼
    FAISS Index (persisted em data/processed/faiss_index/)
         │
         ▼ Query time
    Retriever: top-3 chunks por similaridade coseno
         │
         ▼
    Context injected no prompt do agente
```

---

## Parte 5 — API de Serving (FastAPI + AKS)

### 5.1 Endpoints

```
POST /predict
  Request:  {"transaction_id": str, "features": dict[str, float]}
  Response: {"fraud_probability": float, "decision": str, "model_version": str}
  SLA:      P95 < 100ms

POST /ask
  Request:  {"query": str, "session_id": str | None}
  Response: {"answer": str, "sources": list[str], "iterations": int}
  SLA:      P95 < 3000ms

GET /health
  Response: {"status": "healthy", "model_version": str, "uptime_s": float}

GET /metrics
  Response: Prometheus text format (para scraping)
```

### 5.2 Middleware de Observabilidade

```python
# Cada request passa por:
1. RequestLoggingMiddleware  → structured log com trace_id
2. PrometheusMiddleware      → latência, throughput, erros
3. GuardrailsMiddleware      → input check antes de chegar no handler
4. PIISanitizationMiddleware → output check antes de retornar
```

### 5.3 Deploy no AKS

```yaml
# kubernetes/deployment.yaml
resources:
  requests: {cpu: "500m", memory: "512Mi"}
  limits:   {cpu: "2000m", memory: "2Gi"}

autoscaling:
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

readinessProbe:
  httpGet: {path: /health, port: 8000}
  initialDelaySeconds: 10

livenessProbe:
  httpGet: {path: /health, port: 8000}
  periodSeconds: 30
```

---

## Parte 6 — Observabilidade

### 6.1 Métricas Prometheus (8 obrigatórias)

```python
# src/monitoring/metrics.py

request_latency_seconds      # Histogram por endpoint
request_total                # Counter por status code
ragas_faithfulness_score     # Gauge — rolling 24h
ragas_answer_relevancy_score # Gauge — rolling 24h
drift_psi_score              # Gauge por feature
model_auc_current            # Gauge — modelo champion
fraud_rate_rolling_1h        # Gauge — taxa de fraude
llm_tokens_used_total        # Counter por modelo
```

### 6.2 Traces Langfuse

```python
# Cada chamada ao agente gera:
Trace:
  ├── Span: guardrail_input_check (latência, resultado)
  ├── Span: rag_retrieval (query, chunks, latência)
  ├── Span: tool_call_1 (nome, input, output, latência)
  ├── Span: tool_call_2 (...)
  ├── Span: llm_synthesis (tokens_in, tokens_out, latência)
  └── Span: guardrail_output_check (pii_detected: bool)

Score: ragas_faithfulness (automático pós-query)
Score: llm_judge_safety (automático pós-query)
```

### 6.3 Grafana Dashboard — Painéis

```
Row 1: API Health
  ├── Request rate (req/min)
  ├── Error rate (%)
  ├── P50/P95/P99 latency
  └── Active replicas AKS

Row 2: Model Quality
  ├── AUC-ROC atual vs. baseline
  ├── Fraud rate (rolling 1h)
  ├── RAGAS faithfulness (rolling 24h)
  └── LLM judge safety score

Row 3: Data Health
  ├── PSI por feature (heatmap)
  ├── Prediction drift alert
  ├── Feature availability
  └── DVC pipeline last run

Row 4: Security
  ├── Guardrail blocks (last 24h)
  ├── PII detections (last 24h)
  ├── Injection attempts
  └── Red team status
```

---

## Parte 7 — Segurança e Compliance

### 7.1 Arquitetura de Guardrails

```python
class InputGuardrail:
    patterns: list[re.Pattern] = [
        re.compile(r"ignore (previous|all) instructions", re.I),
        re.compile(r"(jailbreak|DAN|do anything now)", re.I),
        re.compile(r"(reveal|show|print) (system prompt|training data)", re.I),
        re.compile(r"act as (if you|an AI without)", re.I),
        re.compile(r"(bypass|override) (your|all) (restrictions|filters)", re.I),
    ]

    def check(self, text: str) -> GuardrailResult:
        for pattern in self.patterns:
            if pattern.search(text):
                logger.warning("Injection detectada", extra={"pattern": pattern.pattern})
                return GuardrailResult(allowed=False, reason="prompt_injection", category="LLM01")
        return GuardrailResult(allowed=True, reason="ok", category=None)

class OutputGuardrail:
    pii_detector: PIIDetector

    def apply(self, text: str) -> tuple[str, bool]:
        sanitized, had_pii = self.pii_detector.sanitize(text)
        if had_pii:
            logger.warning("PII detectado no output — sanitizado")
        return sanitized, had_pii
```

### 7.2 PII Detection — Padrões Regex + NER

```python
PII_PATTERNS = {
    "CPF":      r"\d{3}\.?\d{3}\.?\d{3}-?\d{2}",
    "CNPJ":     r"\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}",
    "EMAIL":    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "PHONE":    r"(\+55\s?)?\(?\d{2}\)?\s?\d{4,5}-?\d{4}",
    "CEP":      r"\d{5}-?\d{3}",
}
# + spaCy pt_core_news_sm para entidades PER (nomes próprios)
```

### 7.3 Conformidade LGPD

| Artigo | Requisito | Implementação |
|---|---|---|
| Art. 7-IX | Base legal: interesse legítimo | Documentado em LGPD_PLAN.md + System Card |
| Art. 18 | Direito de acesso | Endpoint GET /explain/{tx_id} com SHAP |
| Art. 20 | Decisão automatizada — direito à explicação | SHAP values expostos via API |
| Art. 46 | Medidas de segurança | Guardrails + PII detection + audit logs |
| BACEN 4.658 | Transparência algorítmica | Model Card + SHAP + MLflow audit trail |

---

## Parte 8 — CI/CD e Qualidade de Software

### 8.1 Pipeline GitHub Actions — Detalhamento

```yaml
# Stage 1: Lint (ruff)
- ruff check src/ tests/ --output-format=github
- ruff format --check src/ tests/

# Stage 2: Type Check (mypy)
- mypy src/ --strict --ignore-missing-imports

# Stage 3: Security Scan (bandit)
- bandit -r src/ -ll -f json -o bandit-report.json
- Falha se severity HIGH > 0

# Stage 4: Tests (pytest)
- pytest tests/ --cov=src --cov-report=xml --cov-fail-under=60
- Upload coverage para PR comment

# Stage 5: RAGAS Evaluation
- python evaluation/ragas_eval.py --golden-set data/golden_set/
- Falha se faithfulness < 0.70

# Stage 6: Docker Build
- docker build -t fraud-api:$SHA src/serving/
- docker run --rm fraud-api:$SHA /health check

# Stage 7: Deploy (apenas branch main)
- kubectl set image deployment/fraud-api fraud-api=$IMAGE:$SHA
- kubectl rollout status deployment/fraud-api --timeout=5m
```

### 8.2 Cobertura de Testes — Distribuição Alvo

```
src/features/feature_engineering.py    → 80% cobertura
src/models/baseline.py                 → 75% cobertura
src/models/train.py                    → 65% cobertura
src/agent/tools.py                     → 85% cobertura
src/agent/react_agent.py               → 70% cobertura
src/agent/rag_pipeline.py              → 65% cobertura
src/serving/app.py                     → 80% cobertura (integração)
src/security/guardrails.py             → 90% cobertura
src/security/pii_detection.py          → 90% cobertura
src/monitoring/drift.py                → 70% cobertura
─────────────────────────────────────────────────────
TOTAL AGREGADO                         → ≥ 60% ✓
```

### 8.3 Makefile — Targets Disponíveis

```makefile
make install        # uv sync --all-extras
make lint           # ruff check + ruff format --check
make type-check     # mypy src/ --strict
make security       # bandit -r src/ -ll
make test           # pytest --cov=src --cov-fail-under=60
make train          # python src/models/train.py
make serve          # uvicorn src.serving.app:app --reload
make eval           # python evaluation/ragas_eval.py
make docker-build   # docker build src/serving/
make docker-run     # docker compose up
make clean          # rm -rf .pytest_cache .mypy_cache __pycache__
```

---

*Documento gerado para Datathon Grupo 42 — FIAP MLET Pós-Tech Fase 5*
*Última atualização: 2026-04-18*
