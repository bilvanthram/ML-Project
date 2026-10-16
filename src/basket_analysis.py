"""
basket_analysis.py - Market Basket Analysis (support / confidence / lift) without Apriori on 32M rows.

Method
------
1. Restrict to the TOP-N most purchased products (default 500). Rare products cannot reach a
   meaningful minimum support anyway, and this bounds memory.
2. Stream the prior file in order-aligned chunks. For each chunk build a sparse
   (orders x top-N products) 0/1 matrix X and accumulate the co-occurrence matrix X^T X.
   Diagonal = how many orders contain each product; off-diagonal = orders containing both.
3. Keep pairs with at least `min_pair_count` co-occurrences. Rules are derived at display time:

     support(A,B)      = orders with A and B / all prior orders
     confidence(A->B)  = orders with A and B / orders with A
     lift(A->B)        = confidence(A->B) / (orders with B / all prior orders)

   lift > 1: A and B are bought together more than chance; lift ~ 1: independent.
Only 1-item -> 1-item rules are produced (pairs). That is a deliberate scope limit.
"""

import json

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.data_loading import DEFAULT_CHUNK_ROWS, iter_order_chunks
from src.data_processing import load_table, save_table
from src.paths import PROCESSED_DIR

META_PATH = PROCESSED_DIR / "basket_meta.json"


def build_basket_tables(top_n=500, min_pair_count=25, chunk_rows=DEFAULT_CHUNK_ROWS):
    ps = load_table("product_summary")
    top = ps.nlargest(top_n, "product_total_purchases")[["product_id", "product_name", "product_total_purchases"]]
    top = top.reset_index(drop=True)
    col_of = np.full(int(ps["product_id"].max()) + 1, -1, dtype=np.int32)
    col_of[top["product_id"].to_numpy()] = np.arange(len(top), dtype=np.int32)

    cooc = np.zeros((len(top), len(top)), dtype=np.int64)
    for chunk in iter_order_chunks("order_products_prior", chunk_rows):
        cols = col_of[chunk["product_id"].to_numpy()]
        keep = cols >= 0
        if not keep.any():
            continue
        _, rows = np.unique(chunk["order_id"].to_numpy()[keep], return_inverse=True)
        x = csr_matrix((np.ones(keep.sum(), dtype=np.int64), (rows, cols[keep])),
                       shape=(rows.max() + 1, len(top)))
        cooc += (x.T @ x).toarray()

    item_count = np.diag(cooc).copy()
    a, b = np.triu_indices(len(top), k=1)
    counts = cooc[a, b]
    ok = counts >= min_pair_count
    pairs = pd.DataFrame({"product_a": top["product_id"].to_numpy()[a[ok]],
                          "product_b": top["product_id"].to_numpy()[b[ok]],
                          "pair_count": counts[ok]})
    items = top.assign(order_count=item_count)[["product_id", "product_name", "order_count"]]
    save_table(pairs, "basket_pairs")
    save_table(items, "basket_items")
    n_orders = json.loads((PROCESSED_DIR / "overview.json").read_text())["prior_orders"]
    META_PATH.write_text(json.dumps({"top_n": int(top_n), "min_pair_count": int(min_pair_count),
                                     "n_orders": int(n_orders), "n_pairs": int(len(pairs))}, indent=2))
    return len(pairs)


def load_basket_meta():
    return json.loads(META_PATH.read_text())


def derive_rules(pairs, items, n_orders, min_support=0.0, min_confidence=0.0, min_lift=0.0):
    """Turn undirected pair counts into directed rules A -> B with support/confidence/lift."""
    name = items.set_index("product_id")["product_name"]
    cnt = items.set_index("product_id")["order_count"]
    fwd = pairs.rename(columns={"product_a": "antecedent_id", "product_b": "consequent_id"})
    bwd = pairs.rename(columns={"product_b": "antecedent_id", "product_a": "consequent_id"})
    r = pd.concat([fwd, bwd], ignore_index=True)
    r["support"] = r["pair_count"] / n_orders
    r["confidence"] = r["pair_count"] / r["antecedent_id"].map(cnt)
    r["lift"] = r["confidence"] / (r["consequent_id"].map(cnt) / n_orders)
    r["antecedent"] = r["antecedent_id"].map(name)
    r["consequent"] = r["consequent_id"].map(name)
    r = r[(r["support"] >= min_support) & (r["confidence"] >= min_confidence) & (r["lift"] >= min_lift)]
    return r[["antecedent", "consequent", "pair_count", "support", "confidence", "lift"]]
