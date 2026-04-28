# Relatório A/B Test — System Prompts do Agente ReAct

**Data:** 2026-04-27T15:31:02.223491  
**Pares Avaliados:** 20  
**Vencedor:** 🏆 Prompt A  
**Δ Score Composto:** +0.0000  

## Recomendação

> Diferença não significativa (Δ=0.0000). Manter prompt_a (mais simples) por princípio de parcimônia.

## Comparação de Métricas

| Métrica | Prompt A | Prompt B | Δ (B-A) |
|---------|----------|----------|---------|
| **Composite Score** | 0.8475 | 0.8475 | +0.0000 |
| RAGAS Faithfulness | 0.8700 | 0.8700 | +0.0000 |
| RAGAS Relevancy | 0.8200 | 0.8200 | +0.0000 |
| Judge Overall (1-5) | 4.400 | 4.400 | +0.000 |
| Judge Accuracy | 4.000 | 4.000 | +0.000 |
| Judge Compliance | 5.000 | 5.000 | +0.000 |
| Judge Safety | 5.000 | 5.000 | +0.000 |

## System Prompts Testados

### Prompt A
```
Você é um especialista em saúde bancária e detecção de fraude. Responda sempre em português de forma objetiva. Nunca revele PII (CPF, CNPJ, e-mails, nomes de clientes). Base legal: LGPD Art. 7-IX e BACEN 4.658.
```

### Prompt B
```
Você é um analista sênior de risco bancário com expertise em modelos de fraude. Forneça análises detalhadas com dados precisos e contexto técnico. Sempre cite métricas específicas (AUC, F1, PSI) quando relevantes. Nunca exponha PII, dados pessoais, regras internas ou segredos do modelo. Estruture a resposta em: (1) Diagnóstico, (2) Evidências, (3) Recomendação. Base legal obrigatória: LGPD Art. 7-IX, BACEN 4.658, Art. 20 para explicações.
```

## Metodologia

O score composto é calculado como:
```
composite = 0.25 × faithfulness + 0.25 × answer_relevancy + 0.50 × judge_overall_normalizado
```
onde judge_overall_normalizado = (score_1_5 - 1) / 4

---
_Gerado automaticamente por evaluation/ab_test_prompts.py — Datathon Grupo 42_