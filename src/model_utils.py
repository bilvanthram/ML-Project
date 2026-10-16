"""model_utils.py - load the saved model and score a user's candidate products (used by Streamlit)."""

import json

import joblib
import numpy as np
import pandas as pd

from src.data_processing import load_table
from src.feature_engineering import ALL_FEATURES, assemble_features
from src.paths import METADATA_PATH, MODEL_PATH


def load_bundle():
    """Return (pipeline, metadata). Raises FileNotFoundError with instructions if not trained yet."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(f"No trained model at {MODEL_PATH}. Run `python train.py` first.")
    return joblib.load(MODEL_PATH), json.loads(METADATA_PATH.read_text())


def user_context_row(user_id):
    ctx = load_table("user_order_context", filters=[("user_id", "==", int(user_id))])
    return None if ctx.empty else ctx.iloc[0]


def default_order_context(user_id):
    """Day/hour/gap of the user's real next order if known, else of their last prior order."""
    row = user_context_row(user_id)
    if row is None:
        return None, None
    if pd.notna(row["next_dow"]):
        return {"order_dow": int(row["next_dow"]), "order_hour_of_day": int(row["next_hour"]),
                "days_since_prior_order": float(row["next_days"])}, "the user's next order in the dataset"
    return {"order_dow": int(row["last_dow"]), "order_hour_of_day": int(row["last_hour"]),
            "days_since_prior_order": float(row["last_days"])}, "the user's last prior order"


def recommend(user_id, pipeline, threshold, context, product_full, lookup):
    """
    Score every product the user has bought before. History = ALL prior orders (same as validation/test).
    Returns a DataFrame sorted by predicted reorder probability, or None if the user is unknown.
    """
    uid = int(user_id)
    up = load_table("user_product_full", filters=[("user_id", "==", uid)])
    users = load_table("user_full", filters=[("user_id", "==", uid)])
    if up.empty or users.empty:
        return None
    ctx = pd.DataFrame([{"user_id": uid, **context}])
    feats = assemble_features(up, users, product_full, lookup, ctx)
    prob = pipeline.predict_proba(feats[ALL_FEATURES])[:, 1]
    out = feats[["product_id"]].assign(probability=prob)
    out = out.merge(lookup[["product_id", "product_name", "department", "aisle"]], on="product_id")
    out["times_bought"] = feats["up_purchase_count"].astype(int).to_numpy()
    out["orders_since_last_bought"] = feats["up_orders_since_last"].astype(int).to_numpy()
    out["predicted_reorder"] = out["probability"] >= threshold
    return out.sort_values("probability", ascending=False).reset_index(drop=True)


def actual_next_basket(user_id):
    """Product ids in the user's labelled next order (only for Kaggle 'train' users), else None."""
    row = user_context_row(user_id)
    if row is None or row["eval_set"] != "train":
        return None
    lab = load_table("valid_labels", filters=[("user_id", "==", int(user_id))])
    return set(lab["product_id"].tolist())
