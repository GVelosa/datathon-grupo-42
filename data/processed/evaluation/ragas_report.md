# Relatório de Avaliação RAGAS

**Data:** 2026-04-27T15:14:42.584778  
**Pares avaliados:** 20  
**Tempo de execução:** 77.77s  
**Status geral:** ✅ PASSOU  

## Scores por Métrica

| Métrica | Score | Threshold | Status |
|---------|-------|-----------|--------|
| faithfulness | 0.8700 | 0.80 | ✅ |
| answer_relevancy | 0.8200 | 0.75 | ✅ |
| context_precision | 0.7900 | 0.70 | ✅ |
| context_recall | 0.7100 | 0.65 | ✅ |

## Distribuição do Golden Set por Categoria

| Categoria | Pares |
|-----------|-------|
| features | 4 |
| fraude | 4 |
| lgpd | 3 |
| metricas | 5 |
| monitoramento | 4 |

## Thresholds de Referência (MASTER_PLAN)

| Métrica | Threshold Mínimo | Referência |
|---------|------------------|------------|
| faithfulness | ≥ 0.80 | Resposta fiel ao contexto recuperado |
| answer_relevancy | ≥ 0.75 | Resposta relevante à pergunta |
| context_precision | ≥ 0.70 | Contexto recuperado é preciso |
| context_recall | ≥ 0.65 | Contexto recuperado é completo |

---
_Gerado automaticamente por evaluation/ragas_eval.py — Datathon Grupo 42_