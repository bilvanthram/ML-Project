"""evaluate_models.py - metrics and plots shared by training and the dashboard."""

import matplotlib
matplotlib.use("Agg")  # headless-safe (training scripts, Streamlit server)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)

from src.paths import FIGURES_DIR


def best_f1_threshold(y, p):
    """Probability cut-off that maximises F1 (chosen on VALIDATION data only)."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))])


def evaluate(y, p, threshold):
    y = np.asarray(y)
    pred = (np.asarray(p) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "accuracy": (tp + tn) / len(y), "threshold": float(threshold),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "positive_rate": float(y.mean())}


def _save(name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()
    return path


def plot_roc(curves, y, title):
    """curves: {model name: validation probabilities}"""
    plt.figure(figsize=(6, 5))
    for name, p in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="chance")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title(title); plt.legend(fontsize=8)
    return _save("roc_curves.png")


def plot_confusion(y, p, threshold, title):
    cm = confusion_matrix(y, (np.asarray(p) >= threshold).astype(int))
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", color="black")
    ax.set_xticks([0, 1], ["Not in next order", "In next order"]); ax.set_yticks([0, 1], ["Not in next order", "In next order"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    return _save("confusion_matrix.png")


def plot_importance(names, values, title, filename, xlabel):
    order = np.argsort(values)[::-1][:20][::-1]
    plt.figure(figsize=(7, 5.5))
    plt.barh(np.array(names)[order], np.array(values)[order], color="steelblue")
    plt.xlabel(xlabel); plt.title(title)
    return _save(filename)
