# Model Comparison — Illicit-Class Metrics on Temporal Test Split (time steps 35-49)

| Model | Precision | Recall | F1 | ROC-AUC | Accuracy |
|---|---|---|---|---|---|
| Logistic Regression | 0.1833 | 0.8800 | 0.3034 | 0.8812 | 0.7375 |
| **Random Forest (served)** | **0.8729** | **0.7165** | **0.7870** | **0.9291** | **0.9748** |
| XGBoost | 0.8418 | 0.7368 | 0.7858 | 0.9288 | 0.9739 |
| GCN | 0.6893 | 0.4691 | 0.5582 | 0.8639 | 0.9518 |
| GraphSAGE | 0.6132 | 0.5854 | 0.5990 | 0.8799 | 0.9491 |

All metrics are for the illicit class specifically (precision/recall/F1/ROC-AUC), computed on the same temporal held-out test split (steps 35-49) used throughout the project — never a random resample, since that would leak future graph structure into evaluation. Accuracy is included for reference only; it's a misleading standalone metric here since ~93% of the test set is licit, so a model that always predicts "licit" would score ~93% accuracy while catching zero illicit transactions.

Reproduced from the saved model checkpoints (`backend/models/saved/`) on 2026-07-08 by running `python backend/models/evaluate.py`.
