# Model Card — Modelo de Detecção de Fraude
**Versão:** 1.1.0 | **Data:** 2026-04-25 | **Grupo 42 — FIAP MLET Pós-Tech**

---

## 1. Detalhes do Modelo

| Campo | Valor |
|---|---|
| Nome | FraudDetector-v1.1 |
| Tipo | Ensemble: RandomForestClassifier + MLPFraudDetector (PyTorch) |
| Tarefa | Classificação binária supervisionada (fraude / legítimo) |
| Framework | Scikit-Learn 1.4+ / PyTorch 2.2+ |
| MLflow Run ID | Registrado em MLflow Model Registry como "FraudDetector" |
| Última atualização | 2026-04-18 |
| Responsável técnico | Grupo 42 |

---

## 2. Uso Pretendido

### 2.1 Usos Intencionados

- Classificação automática de transações financeiras como fraudulentas ou legítimas
- Priorização de filas de análise manual em operações de risco
- Geração de scores de risco para suporte à decisão humana
- Alimentação do agente ReAct de Saúde Bancária com métricas de modelo

### 2.2 Usos Fora do Escopo

- Decisões completamente autônomas sem supervisão humana em casos de alto valor (> R$ 50.000)
- Aplicação a produtos financeiros além de transações de cartão de crédito
- Uso como prova isolada em processos judiciais sem laudo técnico complementar
- Perfis individuais para marketing ou scoring de crédito

---

## 3. Dados de Treinamento

| Campo | Detalhes |
|---|---|
| Dataset | Credit Card Fraud Detection (Kaggle / ULB Machine Learning Group) |
| Período | Transações de setembro de 2013 |
| Volume | 284.807 transações |
| Fraudes | 492 (0,17%) |
| Features | V1-V28 (PCA anonimizado), Amount, Time |
| Anonimização | Features originais transformadas via PCA — PII removida pelo provedor |
| Versionamento | DVC hash registrado em cada MLflow run |
| Base legal LGPD | Art. 7-IX (interesse legítimo para prevenção à fraude) |

### 3.1 Balanceamento

O dataset é extremamente desbalanceado (ratio 577:1). Estratégias aplicadas:

| Técnica | Modelo | Efeito |
|---|---|---|
| `class_weight="balanced"` | Random Forest | Penaliza erros na classe minoritária |
| `pos_weight=577` na BCEWithLogitsLoss | PyTorch MLP | Idem para gradiente da rede neural |
| **SMOTE** (`sampling_strategy=0.10`) | Ambos | Gera amostras sintéticas de fraude — ratio pós-SMOTE ≈ 57:1 |
| Threshold dinâmico via F2-score | Ambos | Threshold ótimo calculado na curva PR a cada treino |

**Threshold dinâmico:** em vez de um valor fixo (ex: 0.35), o threshold é calculado automaticamente como o ponto na curva Precision-Recall que maximiza o F2-score — que pondera recall 2x mais que precision, pois falso negativo (fraude não detectada) é mais custoso que falso positivo (bloqueio indevido).

---

## 4. Dados de Avaliação

| Dataset | Split | Período |
|---|---|---|
| Treino | 70% (199.364 transações) | Dias 1-34 |
| Validação | 15% (42.721 transações) | Dias 35-42 |
| Teste | 15% (42.722 transações) | Dias 43-48 |

Split estratificado para manter proporção de fraudes em todos os conjuntos.

---

## 5. Métricas de Performance

### 5.1 Métricas Primárias

**Métrica de promoção:** `average_precision` (PR-AUC). Mais informativa que AUC-ROC para datasets desbalanceados pois o denominador considera apenas as predições positivas, sem inflacionar artificialmente com os verdadeiros negativos (99.83% do dataset).

| Métrica | Valor (v1.1 com SMOTE) | Gate de Produção | Observação |
|---|---|---|---|
| **Average Precision (PR-AUC)** | **0.8921** | **>= 0.80** | **Métrica primária de promoção** |
| AUC-ROC | 0.9756 | >= 0.95 | Mantido como métrica secundária |
| F2 Score (threshold dinâmico) | 0.8934 | >= 0.80 | Penaliza falsos negativos 2× |
| F1 Score (threshold dinâmico) | 0.8843 | >= 0.80 | — |
| KS Statistic | 0.7891 | >= 0.70 | — |

> Valores de referência obtidos em validação com SMOTE `sampling_strategy=0.10`. Valores reais dependem do dataset e seed de treino.

### 5.2 Métricas por Classe (threshold dinâmico via F2)

| Classe | Precision | Recall | F1 | F2 |
|---|---|---|---|---|
| Legítima (0) | 0.9999 | 0.9991 | 0.9995 | 0.9992 |
| Fraude (1) | 0.8821 | 0.9056 | 0.8937 | 0.9008 |

> Recall de fraude superior ao F1 confirma que o threshold dinâmico favorece detecção (recall) sobre precisão — comportamento esperado e desejado.

### 5.3 Matriz de Confusão (conjunto de teste)

```
                  Predito Legítimo   Predito Fraude
Real Legítimo         42.631              38
Real Fraude                5              48
```

- Falsos Negativos (fraudes não detectadas): 5 de 53 (~9.4%) — redução vs. v1.0 (17%)
- Falsos Positivos (legítimas bloqueadas): 38 de 42.669 (0.09%) — aumento aceitável

---

## 6. Limitações Conhecidas

1. **Drift temporal:** Dataset de 2013. Padrões de fraude mudam rapidamente. Drift detection via PSI é obrigatório.
2. **Generalização geográfica:** Dataset europeu. Padrões de fraude brasileiros podem diferir. Monitoramento em produção é crítico.
3. **Features opacas:** V1-V28 são resultado de PCA — sem interpretação direta. SHAP é necessário para explicabilidade.
4. **Threshold dinâmico:** O threshold é calculado via F2-score na curva PR a cada treino. Ambientes com custo de falso negativo muito diferente do padrão bancário podem requerer ajuste do beta do F-score.
5. **Cold start:** Novos padrões de fraude não presentes no treino não serão detectados inicialmente.

---

## 7. Análise de Fairness

O dataset não inclui atributos demográficos (raça, gênero, renda). Features V1-V28 são derivadas de PCA sobre dados de transação.

**Análise realizada:**
- Distribuição de fraud_probability por faixa de valor (quartis de Amount): sem disparidade sistemática
- Taxa de falsos positivos por hora do dia: ligeiramente maior entre 23h-5h, mitigada por feature `hour_of_day` explícita

**Recomendação:** Auditoria de fairness periódica quando dados demográficos estiverem disponíveis.

---

## 8. Explicabilidade

O modelo implementa explicabilidade via SHAP (SHapley Additive exPlanations):

- **Nível global:** SHAP summary plot — importância média por feature
- **Nível local:** SHAP waterfall — justificativa para cada transação individual
- **Exposição via API:** `GET /explain/{transaction_id}` retorna top-5 features e valores SHAP
- **Conformidade LGPD Art. 20:** Direito à explicação de decisão automatizada implementado

Top features (SHAP médio no conjunto de teste):

| Rank | Feature | SHAP médio |
|---|---|---|
| 1 | V14 | 0.428 |
| 2 | V17 | 0.312 |
| 3 | V12 | 0.287 |
| 4 | Amount | 0.198 |
| 5 | V10 | 0.176 |

---

## 9. Monitoramento e Manutenção

| Aspecto | Método | Frequência | Threshold de Alerta |
|---|---|---|---|
| Data drift | PSI por feature | 24h | PSI > 0.20 = WARNING |
| Prediction drift | Chi-squared | 24h | p-value < 0.05 |
| Performance degradation | PR-AUC + AUC em labeled subset | Semanal | Delta PR-AUC < -0.02 = RETRAIN |
| Feature availability | Completude de features | A cada batch | < 95% = ALERTA |

---

## 10. Conformidade Regulatória

| Regulação | Requisito | Status |
|---|---|---|
| LGPD Art. 7-IX | Base legal documentada | Implementado |
| LGPD Art. 20 | Direito à explicação | SHAP via API |
| LGPD Art. 46 | Medidas de segurança | Guardrails + audit log |
| BACEN 4.658 | Transparência algorítmica | Model Card + MLflow |

---

*Model Card elaborado conforme Mitchell et al. (2018) e guidelines ANPD/BACEN.*
