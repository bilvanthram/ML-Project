"""
ml_pipeline.py - TRAIN, EVALUATE, SELECT and SAVE the reorder model.

    python train.py                       # defaults: 40k training users, 20k validation+test users
    python train.py --extended            # also Decision Tree / RF / AdaBoost / GB / XGBoost (syllabus comparison)

Core comparison: Logistic Regression (interpretable baseline) vs HistGradientBoosting.
A no-ML baseline ("rank by how often the user bought the item") is reported for context.
See feature_engineering.py for the exact target definition and leakage rules.

Split (temporal + user-disjoint):
  train : past prediction point (target = each user's LAST PRIOR order), random training users
  valid : later prediction point (target = Kaggle 'train' order), used to pick the model + threshold
  test  : later prediction point, DIFFERENT users, touched only once for the final report
"""

import argparse
import json
import platform
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data_processing import processed_data_ready
from src.evaluate_models import (best_f1_threshold, evaluate, plot_confusion, plot_importance, plot_roc)
from src.feature_engineering import (ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET,
                                     build_eval_frame, build_training_frame, select_user_splits)
from src.paths import METADATA_PATH, METRICS_DIR, MODEL_PATH, MODELS_DIR
from src.preprocessing import build_preprocessor, transformed_feature_names

SEED = 42
TARGET_DEFINITION = ("will_reorder = 1 if a product the user bought in an earlier order appears in the user's target "
                     "order, else 0. Candidates = products in the user's purchase history; features use only orders "
                     "before the target order.")


def core_models():
    hgb_cat_mask = [False] * len(NUMERIC_FEATURES) + [True] * len(CATEGORICAL_FEATURES)
    return {
        "Logistic Regression": ("linear", LogisticRegression(max_iter=500, C=1.0)),  # L2 penalty (sklearn default)
        "HistGradientBoosting": ("hgb", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=15,
            categorical_features=hgb_cat_mask, random_state=SEED)),
    }


def extended_models():
    from xgboost import XGBClassifier
    return {
        "Decision Tree": ("tree", DecisionTreeClassifier(max_depth=8, min_samples_leaf=200, random_state=SEED)),
        "Random Forest": ("tree", RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_leaf=50,
                                                         n_jobs=-1, random_state=SEED)),
        "AdaBoost": ("tree", AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=SEED)),
        "Gradient Boosting": ("tree", GradientBoostingClassifier(n_estimators=100, max_depth=3, subsample=0.8,
                                                                 random_state=SEED)),
        "XGBoost": ("tree", XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                                          colsample_bytree=0.8, tree_method="hist", eval_metric="auc",
                                          n_jobs=-1, random_state=SEED)),
    }


def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o.item() if isinstance(o, np.generic) else o


def _balance(y):
    return {"rows": int(len(y)), "positives": int(y.sum()), "negatives": int(len(y) - y.sum()),
            "positive_rate": float(y.mean())}


def run(train_users=40_000, eval_users=20_000, extended=False, extended_rows=300_000):
    if not processed_data_ready():
        raise FileNotFoundError("Processed tables not found. Run `python main.py` (or `python train.py`, which builds them) first.")
    t_start = time.time()
    tr_u, va_u, te_u = select_user_splits(train_users, eval_users, SEED)
    print(f"Users -> train {len(tr_u):,} | validation {len(va_u):,} | test {len(te_u):,} (disjoint)")

    train = build_training_frame(tr_u); valid = build_eval_frame(va_u); test = build_eval_frame(te_u)
    balance = {"train": _balance(train[TARGET]), "validation": _balance(valid[TARGET]), "test": _balance(test[TARGET])}
    print("\nClass distribution (candidate products):")
    for k, v in balance.items():
        print(f"  {k:10s} rows={v['rows']:>10,}  reordered={v['positives']:>9,}  positive rate={v['positive_rate']:.3f}")
    print(f"\nFeatures ({len(ALL_FEATURES)}): {ALL_FEATURES}\n")

    Xtr, ytr = train[ALL_FEATURES], train[TARGET]
    Xva, yva = valid[ALL_FEATURES], valid[TARGET]
    Xte, yte = test[ALL_FEATURES], test[TARGET]

    rows, val_probs, fitted = [], {}, {}

    # no-ML reference: rank candidates by share of the user's orders containing the product
    thr = best_f1_threshold(yva, valid["up_purchase_share"])
    base_val = evaluate(yva, valid["up_purchase_share"], thr); base_test = evaluate(yte, test["up_purchase_share"], thr)
    rows.append({"model": "Baseline: purchase share (no ML)", "train_rows": 0, "fit_seconds": 0.0,
                 **{f"valid_{k}": v for k, v in base_val.items()}, **{f"test_{k}": v for k, v in base_test.items()}})

    models = core_models()
    if extended:
        models.update(extended_models())
    for name, (kind, est) in models.items():
        Xfit, yfit = Xtr, ytr
        if extended and name not in core_models() and len(Xtr) > extended_rows:
            idx = np.random.default_rng(SEED).choice(len(Xtr), extended_rows, replace=False)
            Xfit, yfit = Xtr.iloc[idx], ytr.iloc[idx]
            print(f"[{name}] trained on a {extended_rows:,}-row random subsample of the training set (cost control).")
        pipe = Pipeline([("preprocessor", build_preprocessor(kind)), ("model", est)])
        t0 = time.time(); pipe.fit(Xfit, yfit); fit_s = time.time() - t0
        p_val = pipe.predict_proba(Xva)[:, 1]; p_test = pipe.predict_proba(Xte)[:, 1]
        thr = best_f1_threshold(yva, p_val)  # threshold chosen on validation only
        mv, mt = evaluate(yva, p_val, thr), evaluate(yte, p_test, thr)
        print(f"{name:22s} fit {fit_s:6.1f}s | valid AUC {mv['roc_auc']:.4f} PR-AUC {mv['pr_auc']:.4f} "
              f"F1 {mv['f1']:.4f} @thr {thr:.3f} | test AUC {mt['roc_auc']:.4f}")
        rows.append({"model": name, "train_rows": int(len(Xfit)), "fit_seconds": round(fit_s, 1),
                     **{f"valid_{k}": v for k, v in mv.items()}, **{f"test_{k}": v for k, v in mt.items()}})
        val_probs[name] = p_val; fitted[name] = (pipe, p_test, thr)

    table = pd.DataFrame(rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(METRICS_DIR / "model_comparison.csv", index=False)

    # model selection: highest VALIDATION ROC-AUC among trained models
    best = max(fitted, key=lambda n: table.loc[table["model"] == n, "valid_roc_auc"].iloc[0])
    pipe, p_test, thr = fitted[best]
    best_row = table[table["model"] == best].iloc[0]
    print(f"\nSelected model: {best} (validation ROC-AUC {best_row['valid_roc_auc']:.4f}); threshold {thr:.3f} from validation F1.")

    plot_roc(val_probs, yva, "Validation ROC curves")
    plot_confusion(yte, p_test, thr, f"{best} - test set")

    # importance for the selected model (permutation importance on original features, on validation rows)
    n = min(20_000, len(Xva)); idx = np.random.default_rng(SEED).choice(len(Xva), n, replace=False)
    pi = permutation_importance(pipe, Xva.iloc[idx], yva.iloc[idx], scoring="roc_auc", n_repeats=3,
                                random_state=SEED, n_jobs=1)
    imp = pd.DataFrame({"feature": ALL_FEATURES, "auc_drop": pi.importances_mean, "std": pi.importances_std}) \
        .sort_values("auc_drop", ascending=False)
    imp.to_csv(METRICS_DIR / "permutation_importance.csv", index=False)
    plot_importance(imp["feature"], imp["auc_drop"], f"Permutation importance - {best}", "permutation_importance.png",
                    "Drop in validation ROC-AUC when feature is shuffled")
    if best == "Logistic Regression":
        coefs = pd.DataFrame({"feature": transformed_feature_names(pipe), "coefficient": pipe.named_steps["model"].coef_[0]})
        coefs.reindex(coefs["coefficient"].abs().sort_values(ascending=False).index).to_csv(METRICS_DIR / "lr_coefficients.csv", index=False)

    # ---- save artifact + metadata ---------------------------------------------
    metadata = _json_safe({
        "model_name": best, "trained_at": datetime.now().isoformat(timespec="seconds"),
        "target": TARGET, "target_definition": TARGET_DEFINITION,
        "numeric_features": NUMERIC_FEATURES, "categorical_features": CATEGORICAL_FEATURES,
        "decision_threshold": thr,
        "threshold_rule": "probability cut-off maximising F1 on the validation users",
        "selection_rule": "highest validation ROC-AUC among trained models",
        "validation_metrics": {k: best_row[f"valid_{k}"] for k in ["roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy"]},
        "test_metrics": {k: best_row[f"test_{k}"] for k in ["roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy"]},
        "class_distribution": balance,
        "split": {"users_train": len(tr_u), "users_validation": len(va_u), "users_test": len(te_u),
                  "scheme": "temporal + user-disjoint: train target = last prior order (history = orders before it); "
                            "validation/test target = the labelled next order (history = all prior orders)",
                  "seed": SEED, "extended_models_run": bool(extended)},
        "library_versions": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__, "python": platform.python_version()},
        "total_seconds": round(time.time() - t_start, 1),
    })
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH, compress=3)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    # verify the saved artifact reproduces in-memory predictions
    reloaded = joblib.load(MODEL_PATH)
    chk = Xte.iloc[:500]
    assert np.allclose(reloaded.predict_proba(chk)[:, 1], pipe.predict_proba(chk)[:, 1]), "saved model differs!"
    json.loads(METADATA_PATH.read_text())
    print(f"Saved {MODEL_PATH} and {METADATA_PATH} (verified: reload reproduces predictions).")
    print(table[["model", "valid_roc_auc", "valid_pr_auc", "valid_precision", "valid_recall", "valid_f1", "test_roc_auc"]]
          .round(4).to_string(index=False))
    return table, metadata


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-users", type=int, default=40_000)
    ap.add_argument("--eval-users", type=int, default=20_000)
    ap.add_argument("--extended", action="store_true")
    a = ap.parse_args()
    run(a.train_users, a.eval_users, a.extended)
