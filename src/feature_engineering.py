"""
feature_engineering.py - AGGREGATES -> ML FEATURE TABLE (leakage-safe).

ML TASK
-------
Binary classification. For a user about to place a NEW order, and a product the user
has bought in a PREVIOUS order (the candidate), predict:

    will_reorder = 1  if the product appears in that new order, else 0.

(Instacart's `reordered=1` means "user bought this product before", so for a candidate
that was already bought this is simply "is it in the next basket".)

PREDICTION POINTS (this is what makes the split temporal)
---------------------------------------------------------
Each user has prior orders 1..n and one later order n+1 (Kaggle "train"/"test").

  * TRAINING example (past point):   history = orders 1..n-1,  target order = n
  * VALIDATION/TEST example (later): history = orders 1..n,    target order = n+1
    (labels from order_products__train, only available for eval_set == 'train' users)

Training and validation/test use DISJOINT users, and the validation/test point is later
in each user's timeline than the training point.

LEAKAGE RULES
-------------
1. Every history feature (user / product / user-product) is computed only from orders
   BEFORE the target order. The "hist" tables exclude every user's order n entirely
   (including from the global product statistics), so no training feature ever sees a
   training target.
2. Nothing from the target order's basket is used as a feature - in particular NOT
   add_to_cart_order of the target order, and the candidate set is defined from
   history, not from what the target order contains.
3. Target-order context (day of week, hour, days since previous order) is known when an
   order starts, so it is allowed. It is identical for every candidate of a user.
4. IDs (user_id, product_id) are never model inputs.

The very same `assemble_features` function builds training rows, evaluation rows and the
Streamlit recommendation rows, so training and serving cannot drift apart.
"""

import numpy as np
import pandas as pd

from src.data_processing import load_table

TARGET = "will_reorder"
MIN_PRIOR_ORDERS = 3          # need >= 2 history orders for the training point

CONTEXT_FEATURES = ["order_dow", "order_hour_of_day", "days_since_prior_order"]
USER_FEATURES = ["user_total_orders", "user_total_products", "user_avg_basket_size",
                 "user_reorder_rate", "user_avg_days_between_orders", "user_distinct_products"]
PRODUCT_FEATURES = ["product_total_purchases", "product_reorder_count", "product_reorder_rate",
                    "product_unique_users", "product_avg_cart_position", "product_repeat_buyer_rate"]
# up_reorder_count is deliberately NOT a feature: in Instacart every purchase after the first
# is a reorder, so it always equals up_purchase_count - 1 (a perfectly redundant column).
USER_PRODUCT_FEATURES = ["up_purchase_count", "up_purchase_share", "up_orders_since_last",
                         "up_avg_cart_position"]
NUMERIC_FEATURES = CONTEXT_FEATURES + USER_FEATURES + PRODUCT_FEATURES + USER_PRODUCT_FEATURES
CATEGORICAL_FEATURES = ["department", "aisle"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def pack_pairs(user_id, product_id):
    return (np.asarray(user_id, dtype=np.int64) << 16) | np.asarray(product_id, dtype=np.int64)


def assemble_features(up, users, products, lookup, context):
    """
    up       : user-product history rows  [user_id, product_id, up_count, up_cartpos_sum, up_last_order, ...]
    users    : user history table         (same history window as `up`)
    products : product history table      (same history window as `up`)
    lookup   : products_lookup (names, aisle, department)
    context  : one row per user with target-order context [user_id, order_dow, order_hour_of_day, days_since_prior_order]
    Returns a DataFrame with user_id, product_id and ALL_FEATURES.
    """
    df = (up.merge(users, on="user_id", how="inner")
            .merge(products, on="product_id", how="inner")
            .merge(lookup[["product_id", "department", "aisle"]], on="product_id", how="left")
            .merge(context[["user_id"] + CONTEXT_FEATURES], on="user_id", how="inner"))
    df["up_purchase_count"] = df["up_count"].astype("float32")
    df["up_purchase_share"] = (df["up_count"] / df["user_total_orders"]).astype("float32")
    df["up_orders_since_last"] = (df["user_total_orders"] - df["up_last_order"]).astype("float32")
    df["up_avg_cart_position"] = (df["up_cartpos_sum"] / df["up_count"]).astype("float32")
    df["department"] = df["department"].fillna("unknown").astype("category")
    df["aisle"] = df["aisle"].fillna("unknown").astype("category")
    out = df[["user_id", "product_id"] + ALL_FEATURES].copy()
    for c in NUMERIC_FEATURES:
        out[c] = out[c].astype("float32")
    return out


def _context_for(ctx, point):
    """Target-order context for a prediction point: 'train' = last prior order, 'eval' = the next order."""
    prefix = "last" if point == "train" else "next"
    out = ctx[["user_id", f"{prefix}_dow", f"{prefix}_hour", f"{prefix}_days"]].copy()
    out.columns = ["user_id"] + CONTEXT_FEATURES
    return out


def build_training_frame(user_ids):
    """Past prediction point: history = orders 1..n-1, label = product is in order n."""
    ids = np.asarray(user_ids)
    up = load_table("user_product_hist", filters=[("user_id", "in", ids.tolist())])
    ctx = load_table("user_order_context")
    ctx = ctx[ctx["user_id"].isin(ids)]
    frame = assemble_features(up, load_table("user_hist"), load_table("product_hist"),
                              load_table("products_lookup"), _context_for(ctx, "train"))
    lab = load_table("last_prior_basket")
    lab = lab[lab["user_id"].isin(ids)]
    frame[TARGET] = np.isin(pack_pairs(frame["user_id"], frame["product_id"]),
                            pack_pairs(lab["user_id"], lab["product_id"])).astype("int8")
    return frame


def build_eval_frame(user_ids):
    """Later prediction point: history = orders 1..n, label = product is in the Kaggle 'train' order n+1."""
    ids = np.asarray(user_ids)
    up = load_table("user_product_full", filters=[("user_id", "in", ids.tolist())])
    ctx = load_table("user_order_context")
    ctx = ctx[ctx["user_id"].isin(ids)]
    frame = assemble_features(up, load_table("user_full"), load_table("product_full"),
                              load_table("products_lookup"), _context_for(ctx, "eval"))
    lab = load_table("valid_labels")
    lab = lab[lab["user_id"].isin(ids)]
    frame[TARGET] = np.isin(pack_pairs(frame["user_id"], frame["product_id"]),
                            pack_pairs(lab["user_id"], lab["product_id"])).astype("int8")
    return frame


def select_user_splits(n_train_users, n_eval_users, seed=42):
    """
    Disjoint user sets:
      valid/test users : random users whose next order has labels (eval_set == 'train'), split 50/50
      training users   : random other users (incl. eval_set == 'test' users, whose LAST PRIOR order is labelled)
    """
    ctx = load_table("user_order_context")
    ctx = ctx[ctx["n_prior_orders"] >= MIN_PRIOR_ORDERS]
    rng = np.random.default_rng(seed)
    labelled = ctx.loc[ctx["eval_set"] == "train", "user_id"].to_numpy()
    n_eval = min(n_eval_users, len(labelled))
    eval_users = rng.choice(labelled, size=n_eval, replace=False)
    valid_users, test_users = eval_users[: n_eval // 2], eval_users[n_eval // 2:]
    pool = ctx.loc[~ctx["user_id"].isin(eval_users), "user_id"].to_numpy()
    train_users = rng.choice(pool, size=min(n_train_users, len(pool)), replace=False)
    return np.sort(train_users), np.sort(valid_users), np.sort(test_users)
