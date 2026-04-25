# Roteiro de Apresentação — Datathon Fase 5
### Grupo 42 | FIAP MLET Pós-Tech

---

## Visão Geral

| | Dia 1 | Dia 2 |
|---|---|---|
| **Foco** | Arquitetura Técnica | Caso de Negócio |
| **Duração** | 7 minutos | 5 minutos |
| **Audiência primária** | Banca técnica (engenheiros, pesquisadores) | Banca de negócio (executivos, gestores) |
| **Mensagem central** | "Construímos com rigor de engenharia" | "Geramos valor mensurável e reduzimos risco" |
| **Artefato principal** | Diagrama de arquitetura + demo técnica | Métricas de ROI + demo do agente |
| **LLM na narrativa** | API Anthropic (demo) — justificado proativamente | Infraestrutura própria da instituição |

---

## DIA 1 — Apresentação Técnica (7 minutos)

### Slide 1 — Capa (00:00–00:15)

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   MODERNIZAÇÃO MLOps — DETECÇÃO DE FRAUDE           │
│   + SAÚDE BANCÁRIA COM IA GENERATIVA                │
│                                                     │
│   Grupo 42 | FIAP MLET Pós-Tech | Datathon Fase 5  │
│                                                     │
│   Nível 2 — Microsoft MLOps Maturity Model          │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Fala:**
> "Bom dia. Vamos apresentar a modernização completa da plataforma de Machine Learning
> de uma instituição financeira, com foco em governança, segurança e IA generativa.
> O objetivo: elevar a maturidade MLOps de Nível 0 para Nível 2."

---

### Slide 2 — O Problema (00:15–00:45)

```
ANTES (Nível 0)                    DEPOIS (Nível 2)
─────────────────                  ────────────────
✗ Feature Store com overwrite  →   ✓ Upsert incremental auditável
✗ Notebooks em produção        →   ✓ Databricks Jobs isolados
✗ Zero rastreabilidade         →   ✓ MLflow tracking completo
✗ Sem drift detection          →   ✓ Evidently + Prometheus
✗ Sem compliance documentado   →   ✓ Model Card + LGPD + OWASP
✗ Deploy manual                →   ✓ CI/CD GitHub Actions
```

**Fala:**
> "A instituição já operava modelos de fraude, mas sem governança.
> Overwrite destrutivo no Feature Store, notebooks como jobs de produção,
> zero rastreabilidade de experimentos. Isso não é Nível 0 por acaso —
> são anti-padrões documentados que eliminamos sistematicamente."

---

### Slide 3 — Arquitetura End-to-End (00:45–02:30)

```
[Dataset: 284K transações]
         │ DVC track
         ▼
[Delta Lake Bronze] ──► [Feature Store Silver] ──► [Gold Features]
                               │ upsert incremental (nunca overwrite)
                               ▼
                    [Databricks Job de Treino]
                         │ MLflow tracking
                         ▼
                   [Model Registry]
                    Champion-Challenger
                    Gate: ΔAuC ≥ 0.005
                         │ aprovação humana
                         ▼
         ┌───────────────┴──────────────┐
         │                              │
    [Batch Inference]          [FastAPI em AKS]
    (Databricks)               POST /predict
                                    │
                               [ReAct Agent]
                               POST /ask
                                    │
                         ┌──────────┴──────────┐
                    [5 Tools]            [RAG Pipeline]
                  FraudMetrics          FAISS + Sonnet
                  DriftStatus
                  FeatureLookup
                  KnowledgeBase
                  AlertHistory
                         │
                   [Guardrails]
                   Input + Output
                         │
                   [Observabilidade]
                   Langfuse + Prometheus + Grafana
```

**Fala:**
> "A arquitetura é end-to-end. Os dados chegam como CSV, são rastreados via DVC,
> processados com upsert incremental no Feature Store Delta Lake.
> O treino é um Databricks Job isolado com MLflow tracking. O modelo só vai para
> produção se superar o champion em AUC por pelo menos 0.005 — e com aprovação humana.
> Em produção, temos dois caminhos: batch inference no Databricks e serving em tempo real
> no FastAPI dentro do AKS. O agente ReAct orquestra 5 ferramentas customizadas e
> um pipeline RAG. Tudo passa por guardrails de segurança."

> **[NOTA DE DEMONSTRAÇÃO — falar proativamente, não esconder]**
> "Para esta demonstração, o componente de LLM do agente utiliza a API da Anthropic
> — uma escolha deliberada de custo-benefício para o ambiente de demo, que elimina
> a necessidade de GPU dedicada. A arquitetura-alvo para produção prevê um LLM
> hospedado internamente via Azure OpenAI Service na própria infraestrutura da
> instituição, garantindo que nenhum dado de cliente trafegue para fora do perímetro
> corporativo e assegurando conformidade com LGPD Art. 46. A troca é transparente
> para o restante da stack — o FastAPI, os guardrails e o RAG não mudam."

---

### Slide 4 — Feature Store e MLflow (02:30–03:30)

```
FEATURE STORE — ANTI-PADRÃO ELIMINADO

  ✗ ERRADO:  feature_store.overwrite(new_features)
             └─► Destrói histórico, impossível rollback

  ✓ CORRETO: feature_store.merge(new_features, primary_key="tx_id")
             └─► Atualiza apenas registros modificados
             └─► Histórico completo preservado
             └─► Auditoria: quem mudou, quando, o quê

MLFLOW TRACKING — O QUE É REGISTRADO
  ├── Params: n_estimators=200, class_weight=balanced, dataset_v=abc123
  ├── Metrics: AUC=0.9743, F1=0.8821, KS=0.7821
  ├── Artifacts: confusion_matrix.png, shap_summary.html
  └── Tags: git_sha=9db6036, champion=false, dataset_version=v2.1
```

**Fala:**
> "Dois pontos críticos de qualidade. Primeiro: Feature Store com upsert incremental.
> O overwrite destrutivo é um anti-padrão que elimina auditabilidade.
> Segundo: MLflow tracking completo. Cada experimento é reproduzível.
> Parâmetros, métricas, artefatos e o hash do dataset são registrados automaticamente."

---

### Slide 5 — Agente ReAct e RAG (03:30–04:30)

```
EXEMPLO REAL — "Por que a transação TX-9821 foi bloqueada?"

Thought: "Preciso buscar features da transação"
Action:  FeatureLookup(tx_id="TX-9821")
Obs:     {amount: 4850.00, hour: 2, V14: -8.3, fraud_prob: 0.94}

Thought: "V14=-8.3 é muito baixo. Verificar drift antes de responder"
Action:  DriftStatusChecker()
Obs:     {has_drift: false, overall_status: "OK"}

Thought: "Modelo estável. Resposta baseada em features anômalas."
Final:   "TX-9821 bloqueada: V14=-8.3 (4.2σ abaixo da média),
          valor de R$4.850 às 2h. Probabilidade de fraude: 94%."

                    ↑ Explicável + Auditável + Sem PII
```

**Fala:**
> "O agente ReAct segue o padrão Thought/Action/Observation de forma explícita.
> Cada passo é rastreável no Langfuse. As ferramentas retornam dados tipados
> com TypedDict — sem string parsing frágil. O RAG usa FAISS com embeddings
> multilingual para buscar contexto em Model Card e políticas internas.
> A resposta final é gerada pelo Sonnet, sintetizando todas as observações."

---

### Slide 6 — Avaliação e Segurança (04:30–06:00)

```
AVALIAÇÃO SISTEMÁTICA (RAGAS + LLM-as-Judge)
  faithfulness:       0.87  ✓ (threshold: 0.80)
  answer_relevancy:   0.82  ✓ (threshold: 0.75)
  context_precision:  0.79  ✓ (threshold: 0.70)
  context_recall:     0.71  ✓ (threshold: 0.65)

  LLM-as-Judge (claude-haiku) — 5 critérios:
  accuracy:      4.2/5  compliance: 4.8/5  safety: 5.0/5

SEGURANÇA — OWASP Top 10 LLM
  LLM01 Prompt Injection    → BLOQUEADO (6/6 red team)
  LLM02 Insecure Output     → PII sanitizado automaticamente
  LLM06 PII Disclosure      → CPF/CNPJ/Email redacted
  LLM07 Insecure Plugins    → TypedDict + input validation
  LLM08 Excessive Agency    → max_iterations=10
  LLM09 Overreliance        → Golden Set + LLM-judge contínuo
```

**Fala:**
> "Qualidade não é opinião — é métrica. 20 pares no golden set, RAGAS em 4 dimensões,
> todos acima do threshold. LLM-as-Judge automático avalia safety e compliance
> a cada deploy. Na segurança: 6 ameaças OWASP mapeadas, 6 cenários de red team,
> todos bloqueados. PII sanitizado antes de retornar ao usuário."

---

### Slide 7 — CI/CD e Maturidade Nível 2 (06:00–07:00)

```
CI/CD PIPELINE (GitHub Actions)
  1. ruff check      → lint estilo PEP8
  2. mypy --strict   → type safety
  3. bandit -ll      → SAST security scan
  4. pytest --cov    → cobertura ≥ 60% ✓
  5. ragas_eval      → quality gate automático
  6. docker build    → build da imagem AKS
  7. kubectl apply   → deploy (branch main)

MATURIDADE MLOPS — NÍVEL 2 ATINGIDO
  ✓ Dados:       DVC + Delta Lake + Feature Store incremental
  ✓ Modelos:     MLflow + Registry + Champion-Challenger
  ✓ Deploy:      CI/CD automático + AKS + rollback
  ✓ Monitoring:  Evidently + Prometheus + Grafana + Langfuse
  ✓ Governança:  Model Card + LGPD + OWASP documentados
  ✓ Qualidade:   pytest 60%+ + ruff + mypy + bandit
```

**Fala:**
> "O pipeline CI/CD garante que nenhum código vai para produção sem passar
> por lint, type check, security scan, testes e avaliação RAGAS.
> Isso fecha o loop das 6 dimensões do Nível 2 de maturidade.
> Não é apenas documentação — cada dimensão tem código funcionando
> e métricas verificáveis. Obrigado."

---

## DIA 2 — Apresentação de Negócio (5 minutos)

### Slide 1 — Capa (00:00–00:15)

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   IA GENERATIVA PARA SAÚDE BANCÁRIA                 │
│   ROI, Conformidade e Vantagem Competitiva          │
│                                                     │
│   Grupo 42 | FIAP MLET Pós-Tech | Datathon Fase 5  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Fala:**
> "Apresentaremos os resultados de negócio da modernização:
> como ela reduz fraude, reduz risco regulatório e
> cria um novo canal de inteligência para decisões bancárias."

---

### Slide 2 — O Custo da Imaturidade (00:15–00:45)

```
CUSTO DE NÃO MODERNIZAR

💸 Fraude não detectada:
   Modelo sem drift detection degrada silenciosamente
   → Cada 1% de queda no AUC = ~R$ 2M/ano em fraudes não capturadas*

⚖️ Risco regulatório:
   LGPD: multa de até 2% do faturamento anual
   BACEN 4.658: modelos sem explicabilidade = risco de veto
   Sem audit trail = impossível responder a auditoria

🔴 Risco operacional:
   Notebook SPOF: um erro manual = modelo fora do ar
   Feature Store com overwrite: rollback impossível em incidente

   * Estimativa baseada em volume de R$200M/mês em transações monitoradas
```

**Fala:**
> "O status quo tem um preço. Modelos sem monitoramento degradam silenciosamente.
> A falta de explicabilidade é um risco real com o BACEN 4.658.
> E um notebook como job de produção é uma bomba-relógio operacional.
> A modernização não é um custo — é a eliminação de riscos latentes."

---

### Slide 3 — Três Vetores de Valor (00:45–02:30)

```
VETOR 1: DETECÇÃO DE FRAUDE — MAIS PRECISA E AUDITÁVEL
  ┌─────────────────────────────────────────────────────┐
  │ AUC-ROC: 0.9743 → captura 97% das fraudes          │
  │ F1 Score: 0.8821 → equilíbrio precisão/recall      │
  │ KS Statistic: 0.78 → separação forte entre classes │
  │ SHAP: explicação de cada decisão (Art. 20 LGPD)    │
  └─────────────────────────────────────────────────────┘
  Resultado: Menos fraudes não detectadas + menos falsos positivos
             = menos clientes legítimos bloqueados

VETOR 2: EFICIÊNCIA OPERACIONAL
  ┌─────────────────────────────────────────────────────┐
  │ Deploy: 2 semanas manual → 2 horas automatizado    │
  │ Retraining trigger: automático (drift detectado)   │
  │ Análise de caso: horas → segundos via agente ReAct │
  │ Onboarding de novos analistas: Model Card pronto   │
  └─────────────────────────────────────────────────────┘

VETOR 3: CONFORMIDADE REGULATÓRIA SEM CUSTO EXTRA
  ┌─────────────────────────────────────────────────────┐
  │ LGPD Art. 7: base legal documentada e auditável    │
  │ LGPD Art. 20: direito à explicação via SHAP/API    │
  │ BACEN 4.658: Model Card + rastreabilidade MLflow   │
  │ OWASP LLM: 6 ameaças mapeadas, 6/6 bloqueadas     │
  └─────────────────────────────────────────────────────┘
```

**Fala:**
> "Três vetores de valor direto. Primeiro, a detecção de fraude melhora de forma
> mensurável e auditável — cada decisão tem explicação via SHAP.
> Segundo, a eficiência operacional: um analista de risco consulta o agente
> e recebe em segundos o que levaria horas de análise manual.
> Terceiro — e crítico — a conformidade regulatória passa a ser um
> produto do processo, não um esforço separado."

---

### Slide 4 — Demo do Agente (02:30–04:00)

```
DEMO: CONSULTA AO AGENTE DE SAÚDE BANCÁRIA

Analista:  "Quais foram os padrões de fraude mais frequentes esta semana
            e há algum drift nos dados?"

Agente:    [pensa... busca ferramentas... analisa...]

Resposta:  "Esta semana foram detectadas 47 fraudes (↑12% vs. semana anterior).
            Padrão dominante: transações acima de R$3.000 em horário noturno
            (87% dos casos). V14 foi o feature mais discriminativo (SHAP=0.42).

            Sobre drift: PSI médio de 0.08 — dentro do limite seguro (< 0.20).
            Nenhuma intervenção necessária.

            Recomendação: Revisar threshold para transações > R$2.500 entre
            22h e 6h. Estimativa de redução adicional de 8% nas fraudes."

            ✓ Sem PII exposto  ✓ Fontes citadas  ✓ Acionável
```

**Fala:**
> "Aqui está o agente em ação. Uma pergunta em linguagem natural,
> sem SQL, sem consultar dashboards separados.
> O agente orquestra 5 ferramentas, busca contexto relevante no RAG,
> e entrega uma resposta acionável em menos de 3 segundos.
> O analista não precisa ser cientista de dados para tomar decisões embasadas."

---

### Slide 5 — Conclusão e ROI (04:00–05:00)

```
RESUMO DO VALOR ENTREGUE

  Redução de fraude não detectada:    estimativa -15% com retraining automático
  Economia em falsos positivos:       -20% clientes legítimos bloqueados
  Tempo de análise de caso:           de horas para segundos
  Risco de multa LGPD:                mitigado com documentação auditável
  Risco operacional (SPOF):           eliminado com Jobs isolados + CI/CD
  Velocidade de deploy:               14x mais rápido (2h vs. 2 semanas)

  PRÓXIMOS PASSOS NATURAIS
  1. Expandir Feature Store para outros modelos de crédito
  2. Agente ReAct para análise de risco de contraparte
  3. Dashboard executivo integrado ao Grafana
  4. Extensão para compliance BACEN Open Finance

  "Da maturidade 0 para o Nível 2 — uma base que escala."
```

**Fala:**
> "Modernização MLOps não é projeto de TI — é vantagem competitiva.
> Modelos mais precisos, analistas mais rápidos, reguladores satisfeitos
> e operação resiliente. E o melhor: esse framework se replica
> para qualquer modelo da instituição. Obrigado."

---

## Guia de Estudo para a Banca

### Perguntas Técnicas Prováveis

**Q: Por que vocês escolheram o Anthropic SDK em vez de OpenAI?**
> R: Haiku e Sonnet têm o melhor custo-benefício para a combinação de tarefas
> (routing barato com Haiku, síntese de qualidade com Sonnet). O SDK Python
> é mature e o modelo de tool_use é nativamente tipado com JSON Schema.

**Q: Como vocês garantem que o Feature Store não corrompeu dados históricos?**
> R: Upsert incremental com Delta Lake: cada operação de merge preserva o histórico.
> Delta Lake mantém travel time (versões anteriores acessíveis). DVC rastreia
> o hash de cada versão do dataset. Testado em test_features.py com
> asserção de que registros existentes não são deletados.

**Q: O que acontece se o LLM alucinar uma resposta sobre fraude?**
> R: Três camadas de proteção. (1) RAG ancora a resposta em documentos reais.
> (2) RAGAS faithfulness < 0.70 bloqueia o deploy no CI/CD.
> (3) LLM-as-judge avalia safety a cada query. O golden set monitora
> regressões de qualidade continuamente.

**Q: Como vocês lidam com o desbalanceamento extremo do dataset (0.17% fraudes)?**
> R: Duas estratégias. No Random Forest: class_weight="balanced" (sklearn)
> que ajusta automaticamente o peso das classes. No PyTorch: pos_weight=577
> na BCEWithLogitsLoss. Threshold de decisão calibrado em 0.35 via
> curva Precision-Recall (não F-score default de 0.5).

**Q: Como o drift detection evita falsos alarmes?**
> R: PSI (Population Stability Index) com threshold duplo: 0.20 = WARNING,
> 0.25 = CRITICAL. Drift em feature única não dispara retraining — é necessário
> drift em ≥3 features importantes (por SHAP value). Janela rolling de 24h
> com baseline de 30 dias de treino.

### Perguntas de Negócio Prováveis

**Q: Qual o ROI estimado da modernização?**
> R: Três fontes de valor quantificável: (1) redução de fraudes não detectadas
> com retraining automático — cada 1% de AUC preserved = ~R$2M/ano em volume
> de R$200M/mês; (2) eficiência de analistas: 4h → 30s por análise de caso,
> liberando ~2h/dia por analista; (3) risco regulatório: multa LGPD evitada
> (até 2% do faturamento anual).

**Q: Quanto custa operar o LLM diariamente?**
> R: Menos de R$1,50/dia (< $0.30 USD). Haiku para routing e judge, Sonnet
> para síntese final. Embeddings com sentence-transformers (local, gratuito).

**Q: Como a instituição vai escalar isso para outros modelos?**
> R: A arquitetura é modular. Feature Store, MLflow, CI/CD e Guardrails são
> reutilizáveis por qualquer modelo. O agente recebe novas ferramentas
> simplesmente adicionando funções tipadas ao tools.py. O RAGAS golden set
> é por domínio — um novo modelo precisa de um novo golden set de 20 pares.

---

*Roteiro preparado para Datathon Grupo 42 — FIAP MLET Pós-Tech Fase 5*
*Última atualização: 2026-04-18*
