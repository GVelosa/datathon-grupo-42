# System Card — Sistema de IA para Saúde Bancária e Detecção de Fraude
**Versão:** 1.0.0 | **Data:** 2026-04-18 | **Grupo 42 — FIAP MLET Pós-Tech**

---

## 1. Propósito do Sistema

O sistema integra modelos preditivos clássicos (detecção de fraude) com IA generativa
(agente ReAct + RAG) para apoio à decisão em operações financeiras de uma instituição
bancária brasileira. O sistema não substitui a decisão humana em casos de alto impacto
— ele amplifica a capacidade analítica dos times de risco e compliance.

**Capacidades principais:**
- Classificação automática de transações financeiras (fraude / legítima)
- Resposta a consultas em linguagem natural sobre saúde bancária
- Explicação de decisões automatizadas (conformidade LGPD Art. 20)
- Detecção proativa de degradação de modelo e drift de dados

---

## 2. Mapa de Componentes

```
┌─────────────────────────────────────────────────────────────────┐
│                       SISTEMA COMPLETO                          │
│                                                                 │
│  [Dados]                                                        │
│   Delta Lake (Bronze/Silver/Gold) + DVC + Feature Store         │
│                    │                                            │
│  [ML Clássico]     │                                            │
│   RandomForest + PyTorch MLP → MLflow Registry → AKS           │
│                    │                                            │
│  [GenAI]           │                                            │
│   ReAct Agent (Anthropic SDK) + RAG (FAISS) + 5 Tools          │
│                    │                                            │
│  [Serving]         │                                            │
│   FastAPI (POST /predict, POST /ask) no AKS                    │
│                    │                                            │
│  [Segurança]       │                                            │
│   InputGuardrail + OutputGuardrail + PIIDetector               │
│                    │                                            │
│  [Observabilidade] │                                            │
│   Prometheus + Grafana + Langfuse + Evidently                   │
│                    │                                            │
│  [CI/CD]           │                                            │
│   GitHub Actions (lint → type → sec → test → eval → deploy)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Fluxo de Dados

### 3.1 Fluxo de Treinamento

```
1. Dados brutos → DVC track → Delta Lake Bronze
2. feature_engineering.py → upsert → Feature Store Silver/Gold
3. Databricks Job → train.py → MLflow tracking
4. Champion-Challenger gate (delta_AUC >= 0.005)
5. Aprovação humana → MLflow Registry → AKS deploy
```

### 3.2 Fluxo de Inferência Real-Time

```
1. Requisição POST /predict com features da transação
2. InputGuardrail.check() → BLOCK se injection detectada
3. Modelo champion → fraud_probability + SHAP
4. OutputGuardrail.apply() → sanitização PII
5. Log estruturado + Prometheus metrics
6. Resposta ao cliente
```

### 3.3 Fluxo do Agente ReAct

```
1. Requisição POST /ask com query em linguagem natural
2. InputGuardrail.check() → BLOCK se injection detectada
3. ReAct Agent: Thought/Action/Observation (max 10 iterações)
4. Tool calls: FraudMetrics, DriftStatus, FeatureLookup, KnowledgeBase, AlertHistory
5. RAG: FAISS retrieval → context injection
6. LLM synthesis (Sonnet) → resposta final
7. OutputGuardrail.apply() → PII sanitization
8. Langfuse trace + RAGAS score automático
9. Resposta ao usuário
```

---

## 4. Trust Boundaries

| Boundary | O que passa | Controle aplicado |
|---|---|---|
| Input externo → Guardrail | Query do usuário | Regex injection + comprimento |
| Guardrail → Agente | Query validada | Formato normalizado |
| Agente → Tools | Tool input (TypedDict) | Validação de schema Pydantic |
| Tools → Agente | Tool output (TypedDict) | Tipos verificados em runtime |
| Agente → OutputGuardrail | Resposta do LLM | PII regex + NER |
| OutputGuardrail → Usuário | Resposta sanitizada | Sem PII, sem dados internos brutos |
| Feature Store → Modelo | Features numéricas | Schema validation, range checks |
| Modelo → API | Probabilidade [0,1] | Clamp + type assertion |

---

## 5. Modos de Falha e Mitigações

| Modo de Falha | Probabilidade | Impacto | Mitigação |
|---|---|---|---|
| LLM alucinação | Média | Alto | RAGAS faithfulness gate + golden set |
| Drift de dados silencioso | Média | Alto | Evidently PSI 24h + Grafana alert |
| Prompt injection bem-sucedida | Baixa | Crítico | Guardrails 2 camadas + regex patterns |
| PII exposto no output | Baixa | Crítico | PIIDetector obrigatório antes de retornar |
| Feature Store corrompido | Muito baixa | Crítico | Delta Lake travel time + DVC rollback |
| Model degradation | Média | Alto | Weekly AUC check + retrain trigger |
| AKS pod crash | Baixa | Médio | minReplicas=2 + liveness probe |
| FAISS index desatualizado | Baixa | Médio | Rebuild automático no CI/CD |

---

## 6. Supervisão Humana

O sistema requer supervisão humana nos seguintes pontos:

| Decisão | Supervisão Necessária | Justificativa |
|---|---|---|
| Promoção de modelo challenger | Aprovação manual obrigatória | Gate automático não é suficiente para prod |
| Transações > R$ 50.000 | Revisão por analista de risco | Alto impacto financeiro |
| Alertas CRITICAL de drift | Análise de causa raiz | Pode indicar mudança de negócio |
| Red team scenarios novos | Teste manual + atualização de guardrails | Ameaças evoluem |
| Atualização de golden set | Revisão por especialista de domínio | Qualidade do benchmark |

---

## 7. Dados Pessoais e Privacidade

O sistema **não armazena dados pessoais identificáveis** em produção:
- Dataset de treino usa V1-V28 (PCA anonimizado pelo provedor)
- Queries ao agente são sanitizadas antes de qualquer logging
- PII detectado no output é redacted antes de retornar ao usuário
- Logs estruturados excluem campos de PII via `logging.Filter`
- Retenção de logs: 30 dias (dados operacionais) / 7 anos (audit trail regulatório)

---

*System Card elaborado conforme práticas de AI Safety e guidelines ANPD/BACEN.*
