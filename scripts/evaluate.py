"""DVC evaluate stage: lê métricas do treino e imprime status do gate de promoção."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

metrics = json.loads(Path("data/processed/eval_metrics.json").read_text(encoding="utf-8"))

best = metrics.get("best_pr_auc", 0.0)
metric = metrics.get("primary_metric", "average_precision")
gate = metrics.get("promotion_gate", 0.005)

rf = metrics.get("rf_metrics", {})
mlp = metrics.get("mlp_metrics", {})

print("=" * 50)
print(f"Metrica primaria : {metric}")
print(f"Gate de promocao : delta >= {gate}")
print("-" * 50)
print(f"RF  — PR-AUC: {rf.get('average_precision', 0):.4f}  AUC-ROC: {rf.get('auc_roc', 0):.4f}  F2: {rf.get('f2_score', 0):.4f}")
print(f"MLP — PR-AUC: {mlp.get('average_precision', 0):.4f}  AUC-ROC: {mlp.get('auc_roc', 0):.4f}  F2: {mlp.get('f2_score', 0):.4f}")
print("-" * 50)
print(f"Melhor modelo   : {metrics.get('best_run_id', 'N/A')}")
print(f"Melhor PR-AUC   : {best:.4f}")
print(f"Status          : {'PROMOVER' if best > 0.80 else 'REVISAR'}")
print("=" * 50)
