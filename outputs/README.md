# outputs/

Everything here is generated - safe to delete and rebuild.

- `processed/` - small aggregate tables (parquet) built by `python main.py`; the dashboard and the ML pipeline read only these.
- `metrics/`   - `model_comparison.csv`, `permutation_importance.csv` (written by `python train.py`).
- `figures/`   - ROC curves, confusion matrix, permutation importance (written by `python train.py`).
