# Plano de Conformidade LGPD
**Sistema:** Detecção de Fraude + Agente de Saúde Bancária
**Versão:** 1.0.0 | **Data:** 2026-04-18 | **Grupo 42 — FIAP MLET Pós-Tech**

---

## 1. Escopo e Contexto

Este plano documenta a conformidade do sistema de IA com a Lei Geral de Proteção de
Dados Pessoais (LGPD — Lei 13.709/2018) e com as diretrizes do Banco Central do Brasil
(BACEN Resolução 4.658/2018) para sistemas de decisão automatizada em instituições financeiras.

---

## 2. Mapeamento de Dados Pessoais

### 2.1 Dados Processados

| Dado | Categoria LGPD | Presente no sistema | Tratamento |
|---|---|---|---|
| Valor da transação (Amount) | Dado financeiro pessoal | Sim — feature de treino | Anonimizado via Z-score |
| Timestamp da transação | Dado de comportamento | Sim — derivado como hora/dia | Anonimizado |
| Features V1-V28 | Dado financeiro (PCA) | Sim — dataset de treino | PCA aplicado pelo provedor |
| ID da transação | Identificador direto | Apenas em memória runtime | Não persistido em logs |
| Query do usuário | Dado de comportamento | Sim — temporário em RAM | Sanitizado antes de log |

### 2.2 Dados NÃO Processados pelo Sistema

- Nome completo do titular do cartão
- Número do cartão de crédito (PAN)
- CPF / CNPJ do titular
- Endereço residencial ou comercial
- Dados sensíveis (saúde, etnia, religião — Art. 5, XI da LGPD)

---

## 3. Base Legal (Art. 7 da LGPD)

### 3.1 Base Legal Principal: Art. 7-IX — Legítimo Interesse

**Aplicação:** Processamento de dados de transações financeiras para prevenção à fraude.

**Justificativa (Legitimate Interest Assessment — LIA):**

| Etapa LIA | Análise |
|---|---|
| **Finalidade** | Prevenção à fraude — interesse legítimo da instituição e proteção ao titular |
| **Necessidade** | Análise automatizada de padrões é indispensável no volume de transações (2M/dia) |
| **Balanceamento** | Benefício do titular (proteção contra fraude) supera o impacto do processamento |
| **Expectativa razoável** | Titulares esperam que transações sejam monitoradas para sua segurança |
| **Mitigação de riscos** | Anonimização PCA + guardrails + retenção limitada |

### 3.2 Base Legal Secundária: Art. 7-VI — Exercício Regular de Direitos

**Aplicação:** Processamento necessário para cumprimento de obrigações regulatórias
(BACEN 4.658/2018, COAF — Lei 9.613/1998 — prevenção à lavagem de dinheiro).

---

## 4. Direitos dos Titulares (Art. 18 da LGPD)

| Direito | Implementação no Sistema |
|---|---|
| **Acesso (Art. 18-I):** saber quais dados são tratados | `GET /explain/{tx_id}` retorna features usadas e score |
| **Correção (Art. 18-III):** corrigir dados incompletos | Processo de contestação via canal de atendimento |
| **Eliminação (Art. 18-VI):** solicitar exclusão | Delta Lake retention policy + pseudonimização de logs |
| **Portabilidade (Art. 18-V):** receber dados em formato interoperável | Endpoint de exportação em JSON (roadmap) |
| **Informação sobre decisão automatizada (Art. 20):** | SHAP values via API — explicação em linguagem natural |

### 4.1 Direito à Explicação — Art. 20 (crítico para o sistema)

O Art. 20 da LGPD confere ao titular o direito de solicitar revisão de decisões tomadas
exclusivamente por meios automatizados que afetem seus interesses.

**Implementação:**
```
GET /explain/{transaction_id}
Response: {
  "decision": "BLOCKED",
  "fraud_probability": 0.94,
  "main_factors": [
    {"feature": "V14", "shap_value": -0.42, "description": "Padrão de transação anômalo"},
    {"feature": "Amount", "shap_value": 0.19, "description": "Valor acima do histórico"},
    {"feature": "hour", "shap_value": 0.15, "description": "Horário de alto risco (2h)"}
  ],
  "human_review_available": true,
  "review_channel": "0800-XXX-XXXX"
}
```

---

## 5. Segurança de Dados (Art. 46 da LGPD)

### 5.1 Medidas Técnicas Implementadas

| Medida | Implementação |
|---|---|
| Pseudonimização | IDs de transação substituídos por hashes em logs |
| Minimização de dados | Apenas features necessárias para o modelo são processadas |
| Controle de acesso | RBAC no Databricks + AKS RBAC |
| Criptografia em trânsito | TLS 1.3 em todos os endpoints FastAPI |
| Criptografia em repouso | Azure Storage Service Encryption (AES-256) no Delta Lake |
| Sanitização de output | PIIDetector obrigatório antes de retornar resposta ao usuário |
| Audit trail | MLflow + Langfuse + structured logging (7 anos de retenção) |

### 5.2 Retenção de Dados

| Tipo de dado | Retenção | Justificativa |
|---|---|---|
| Dataset de treino (Delta Lake Bronze) | 7 anos | BACEN — obrigação regulatória |
| Features processadas (Silver/Gold) | 3 anos | Necessidade operacional de retraining |
| Logs de decisão de fraude | 7 anos | BACEN 4.658 — auditabilidade |
| Logs operacionais (Prometheus) | 90 dias | Monitoramento de performance |
| Traces do agente (Langfuse) | 30 dias | Avaliação de qualidade |
| Queries sanitizadas | Não persistidas | Minimização de dados |

---

## 6. Relatório de Impacto à Proteção de Dados (RIPD)

Conforme Art. 38 da LGPD e Resolução CD/ANPD 2/2022, o RIPD é obrigatório para
processamento de dados com potencial impacto nos direitos dos titulares.

**Status:** Em elaboração pelo DPO da instituição.

**Elementos que serão documentados:**
1. Descrição sistemática dos tratamentos
2. Finalidade e necessidade de cada tratamento
3. Avaliação de riscos aos titulares
4. Medidas de mitigação adotadas
5. Indicação do legítimo interesse (LIA completo)

---

## 7. Papel do Encarregado (DPO — Art. 41)

| Responsabilidade | Contato |
|---|---|
| DPO da Instituição | [definido pela instituição financeira] |
| Contato ANPD | [canal oficial da ANPD] |
| Registro de incidentes | Prazo: 72h após conhecimento (Art. 48) |

---

## 8. Conformidade BACEN 4.658/2018

| Requisito BACEN | Implementação |
|---|---|
| Transparência em modelos de decisão de crédito/risco | Model Card + SHAP endpoint |
| Gestão de risco de modelos | MLflow + Champion-Challenger + drift detection |
| Documentação de políticas e procedimentos | MASTER_PLAN + SOLUTION_DESIGN + Model Card |
| Testes e validação de modelos | pytest >= 60% cobertura + RAGAS + backtesting |
| Monitoramento contínuo | Evidently + Prometheus + Grafana |

---

*Plano elaborado com base na LGPD (Lei 13.709/2018), Resolução CD/ANPD 2/2022 e BACEN 4.658/2018.*
