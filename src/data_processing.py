"""
data_processing.py - RAW DATA -> CLEANING -> AGGREGATION.

Builds every table the dashboard, market-basket analysis and ML pipeline need,
WITHOUT ever creating the 32-million-row merged DataFrame. The 32M-row prior file
is streamed in order-aligned chunks (see data_loading.iter_order_chunks) and
reduced to small aggregate tables that are saved as parquet in outputs/processed/.

Run once (about a few minutes on the full dataset):
    python main.py            (or)   python -m src.data_processing

Tables written (rows in brackets are for the full Instacart data):
  EDA / dashboard
    overview.json               dataset counts, missing values, duplicates, sizes
    product_summary             one row per product              [~50k]
    department_summary, aisle_summary                            [21 / 134]
    orders_by_dow_hour, days_since_prior_hist, orders_per_user_hist,
    basket_size_hist, reorder_by_dow_hour, reorder_by_order_number,
    reorder_by_cart_position                                     [tiny]
  ML feature inputs (all HISTORY only, see feature_engineering.py)
    user_product_full / user_product_hist    one row per (user, product) bought before
                                              [~13M]  -> stored on disk, read per user
    user_full / user_hist                    one row per user    [~206k]
    product_full / product_hist              one row per product [~50k]
    user_order_context                       last prior order + next (labelled) order context
    last_prior_basket                        (user, product) pairs in each user's LAST prior order
    valid_labels                             (user, product) pairs in each user's Kaggle "train" order

"full" = history is ALL prior orders (used to predict the user's next order).
"hist" = history EXCLUDES each user's last prior order (used to build a training example
         whose target is that last prior order). This is what makes the split temporal
         and leakage-free - see feature_engineering.py.
"""

import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

from src.data_loading import DEFAULT_CHUNK_ROWS, FILES, iter_order_chunks, read_table, require_files
from src.paths import DATA_DIR, MANIFEST_PATH, PROCESSED_DIR

KEY_BITS = 16  # (user_id << 16 | product_id) packs a pair into one int64; product_id < 65536


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _log(msg, t0):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def save_table(df, name, row_group_size=None):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False, row_group_size=row_group_size)


def load_table(name, filters=None, columns=None):
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python main.py` first to build the processed tables.")
    return pd.read_parquet(path, filters=filters, columns=columns)


def processed_data_ready():
    return MANIFEST_PATH.exists() and (PROCESSED_DIR / "user_product_full.parquet").exists()


def _pair_key(uid, pid):
    return (uid.astype(np.int64) << KEY_BITS) | pid.astype(np.int64)


def _agg_user_product(key, cartpos, reordered, order_number):
    """Partial (user, product) aggregation of one chunk (indexed by packed key)."""
    df = pd.DataFrame({"k": key, "one": 1, "cp": cartpos, "ro": reordered, "on": order_number})
    out = df.groupby("k", sort=False).agg(up_count=("one", "sum"), up_reorders=("ro", "sum"),
                                          up_cartpos_sum=("cp", "sum"), up_last_order=("on", "max"))
    return out.astype("int32")  # int32 halves the memory of the running partial aggregates


def _combine_user_product(parts):
    """Merge partial aggregates (a user's orders may sit in different chunks)."""
    df = pd.concat(parts)
    out = df.groupby(level=0, sort=False).agg(up_count=("up_count", "sum"), up_reorders=("up_reorders", "sum"),
                                              up_cartpos_sum=("up_cartpos_sum", "sum"),
                                              up_last_order=("up_last_order", "max"))
    return out.astype("int32")


def _finalize_user_product(agg):
    df = agg.reset_index().rename(columns={"k": "key"}).sort_values("key")
    out = pd.DataFrame({
        "user_id": (df["key"].to_numpy() >> KEY_BITS).astype(np.int32),
        "product_id": (df["key"].to_numpy() & ((1 << KEY_BITS) - 1)).astype(np.int32),
        "up_count": df["up_count"].astype(np.int16).to_numpy(),
        "up_reorders": df["up_reorders"].astype(np.int16).to_numpy(),
        "up_cartpos_sum": df["up_cartpos_sum"].astype(np.int32).to_numpy(),
        "up_last_order": df["up_last_order"].astype(np.int16).to_numpy(),
    })
    return out  # sorted by user_id then product_id -> parquet row-group pruning works per user


def _product_table(up):
    """Product-level history features, derived from a user-product table (no extra pass over raw data)."""
    g = up.groupby("product_id")
    t = pd.DataFrame({
        "product_total_purchases": g["up_count"].sum(),
        "product_reorder_count": g["up_reorders"].sum(),
        "product_unique_users": g.size(),
        "_cp": g["up_cartpos_sum"].sum(),
        "product_repeat_buyer_rate": (up["up_count"] > 1).groupby(up["product_id"]).mean(),
    })
    t["product_reorder_rate"] = t["product_reorder_count"] / t["product_total_purchases"]
    t["product_avg_cart_position"] = t["_cp"] / t["product_total_purchases"]
    return t.drop(columns="_cp").reset_index().astype({"product_total_purchases": "int64",
                                                       "product_reorder_count": "int64",
                                                       "product_unique_users": "int64"})


def _user_table(order_df, up):
    """User-level history features from the (already restricted) history orders + user-product table."""
    g = order_df.groupby("user_id")
    t = pd.DataFrame({
        "user_total_orders": g.size(),
        "user_total_products": g["basket_size"].sum(),
        "_reordered_items": g["reordered_items"].sum(),
        "user_avg_basket_size": g["basket_size"].mean(),
        "user_avg_days_between_orders": g["days_since_prior_order"].mean(),
        "user_distinct_products": up.groupby("user_id").size(),
    })
    t["user_reorder_rate"] = t["_reordered_items"] / t["user_total_products"]
    return t.drop(columns="_reordered_items").reset_index()


# ---------------------------------------------------------------------------
# main build
# ---------------------------------------------------------------------------

def build_all(chunk_rows=DEFAULT_CHUNK_ROWS):
    t0 = time.time()
    require_files()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ---- small tables: load + light cleaning -----------------------------------
    _log("loading small tables (orders, products, aisles, departments)", t0)
    orders = read_table("orders").drop_duplicates()
    products = read_table("products").drop_duplicates()
    products["product_name"] = products["product_name"].str.strip()
    aisles = read_table("aisles").drop_duplicates()
    aisles["aisle"] = aisles["aisle"].str.strip()
    departments = read_table("departments").drop_duplicates()
    departments["department"] = departments["department"].str.strip()
    # days_since_prior_order is NaN for a user's first order (expected, kept as NaN).
    if int(products["product_id"].max()) >= (1 << KEY_BITS):
        raise ValueError("product_id exceeds 16 bits; adjust KEY_BITS in data_processing.py")

    lookup = (products.merge(aisles, on="aisle_id", how="left").merge(departments, on="department_id", how="left")
              [["product_id", "product_name", "aisle_id", "aisle", "department_id", "department"]])
    save_table(lookup, "products_lookup")

    # ---- dense lookup arrays indexed by order_id / user_id ---------------------
    max_oid = int(orders["order_id"].max())
    max_uid = int(orders["user_id"].max())
    user_of = np.zeros(max_oid + 1, dtype=np.int32)
    onum_of = np.zeros(max_oid + 1, dtype=np.int16)
    dow_of = np.zeros(max_oid + 1, dtype=np.int8)
    hour_of = np.zeros(max_oid + 1, dtype=np.int8)
    oid_arr = orders["order_id"].to_numpy()
    user_of[oid_arr] = orders["user_id"].to_numpy()
    onum_of[oid_arr] = orders["order_number"].to_numpy()
    dow_of[oid_arr] = orders["order_dow"].to_numpy()
    hour_of[oid_arr] = orders["order_hour_of_day"].to_numpy()

    prior_orders = orders[orders["eval_set"] == "prior"]
    n_prior_of_user = np.zeros(max_uid + 1, dtype=np.int16)  # number of prior orders per user
    n_prior_of_user[prior_orders.groupby("user_id")["order_number"].max().index.to_numpy()] = \
        prior_orders.groupby("user_id")["order_number"].max().to_numpy()

    # ---- ONE streaming pass over the 32M-row prior file ------------------------
    order_size = np.zeros(max_oid + 1, dtype=np.int32)
    order_reordered = np.zeros(max_oid + 1, dtype=np.int32)
    seen_product = np.zeros(int(products["product_id"].max()) + 1, dtype=bool)
    up_full_parts, up_hist_parts, last_basket_parts = [], [], []
    stats = {"rows": 0, "duplicate_rows": 0, "missing": {c: 0 for c in ["order_id", "product_id", "add_to_cart_order", "reordered"]},
             "reordered_rows": 0, "chunks": 0}
    dowhour_items = np.zeros(168, dtype=np.int64)
    dowhour_reord = np.zeros(168, dtype=np.int64)
    onum_items = np.zeros(101, dtype=np.int64)
    onum_reord = np.zeros(101, dtype=np.int64)
    cart_items = np.zeros(51, dtype=np.int64)
    cart_reord = np.zeros(51, dtype=np.int64)

    for chunk in iter_order_chunks("order_products_prior", chunk_rows):
        oid = chunk["order_id"].to_numpy()
        pid = chunk["product_id"].to_numpy()
        cp = chunk["add_to_cart_order"].to_numpy()
        ro = chunk["reordered"].to_numpy()
        uid = user_of[oid]
        onum = onum_of[oid]
        is_last = onum == n_prior_of_user[uid]  # row belongs to the user's LAST prior order

        stats["rows"] += len(chunk)
        stats["chunks"] += 1
        stats["duplicate_rows"] += int(chunk.duplicated(["order_id", "product_id"]).sum())
        for c in stats["missing"]:
            stats["missing"][c] += int(chunk[c].isna().sum())
        stats["reordered_rows"] += int(ro.sum())
        seen_product[pid] = True

        order_size += np.bincount(oid, minlength=max_oid + 1).astype(np.int32)
        order_reordered += np.bincount(oid, weights=ro, minlength=max_oid + 1).astype(np.int32)

        dh = dow_of[oid].astype(np.int64) * 24 + hour_of[oid]
        dowhour_items += np.bincount(dh, minlength=168)
        dowhour_reord += np.bincount(dh, weights=ro, minlength=168).astype(np.int64)
        oc = np.minimum(onum, 100).astype(np.int64)
        onum_items += np.bincount(oc, minlength=101)
        onum_reord += np.bincount(oc, weights=ro, minlength=101).astype(np.int64)
        cc = np.minimum(cp, 50).astype(np.int64)
        cart_items += np.bincount(cc, minlength=51)
        cart_reord += np.bincount(cc, weights=ro, minlength=51).astype(np.int64)

        key = _pair_key(uid, pid)
        up_full_parts.append(_agg_user_product(key, cp, ro, onum))
        h = ~is_last
        up_hist_parts.append(_agg_user_product(key[h], cp[h], ro[h], onum[h]))
        last_basket_parts.append(pd.DataFrame({"user_id": uid[is_last], "product_id": pid[is_last]}))

        if len(up_full_parts) >= 6:  # keep memory bounded on the full dataset
            up_full_parts = [_combine_user_product(up_full_parts)]
            up_hist_parts = [_combine_user_product(up_hist_parts)]
        _log(f"chunk {stats['chunks']}: {stats['rows']:,} prior rows processed", t0)

    _log("finalising user-product tables", t0)
    up_full = _finalize_user_product(_combine_user_product(up_full_parts))
    up_hist = _finalize_user_product(_combine_user_product(up_hist_parts))
    del up_full_parts, up_hist_parts
    last_basket = pd.concat(last_basket_parts, ignore_index=True).astype(np.int32)
    save_table(up_full, "user_product_full", row_group_size=100_000)
    save_table(up_hist, "user_product_hist", row_group_size=100_000)
    save_table(last_basket, "last_prior_basket")

    # ---- order-level table (basket size per order) -----------------------------
    ord_df = prior_orders[["order_id", "user_id", "order_number", "days_since_prior_order"]].copy()
    ord_df["basket_size"] = order_size[ord_df["order_id"].to_numpy()]
    ord_df["reordered_items"] = order_reordered[ord_df["order_id"].to_numpy()]
    n_user = n_prior_of_user[ord_df["user_id"].to_numpy()]
    hist_orders = ord_df[ord_df["order_number"].to_numpy() < n_user]

    _log("building user / product tables", t0)
    product_full = _product_table(up_full)
    product_hist = _product_table(up_hist)
    user_full = _user_table(ord_df, up_full)
    user_hist = _user_table(hist_orders, up_hist)
    save_table(product_full, "product_full")
    save_table(product_hist, "product_hist")
    save_table(user_full, "user_full")
    save_table(user_hist, "user_hist")

    # ---- user order context (target-order attributes) --------------------------
    last_idx = prior_orders.groupby("user_id")["order_number"].idxmax()
    last = prior_orders.loc[last_idx, ["user_id", "order_number", "order_dow", "order_hour_of_day", "days_since_prior_order"]]
    last.columns = ["user_id", "n_prior_orders", "last_dow", "last_hour", "last_days"]
    nxt = orders[orders["eval_set"] != "prior"][["user_id", "eval_set", "order_number", "order_dow",
                                                  "order_hour_of_day", "days_since_prior_order"]]
    nxt.columns = ["user_id", "eval_set", "next_order_number", "next_dow", "next_hour", "next_days"]
    nxt = nxt.assign(eval_set=nxt["eval_set"].astype(str))
    context = last.merge(nxt, on="user_id", how="left")
    save_table(context, "user_order_context")

    # ---- Kaggle "train" orders: labels for the validation/test prediction point -
    train = read_table("order_products_train")
    train_uid = user_of[train["order_id"].to_numpy()]
    valid_labels = (pd.DataFrame({"user_id": train_uid, "product_id": train["product_id"].to_numpy()})
                    .drop_duplicates().astype(np.int32))
    save_table(valid_labels, "valid_labels")

    # ---- EDA tables ------------------------------------------------------------
    _log("building EDA tables", t0)
    ps = product_full.merge(lookup, on="product_id", how="right")
    for c in ["product_total_purchases", "product_reorder_count", "product_unique_users"]:
        ps[c] = ps[c].fillna(0).astype("int64")
    save_table(ps[["product_id", "product_name", "aisle", "department", "product_total_purchases",
                   "product_reorder_count", "product_reorder_rate", "product_unique_users",
                   "product_avg_cart_position", "product_repeat_buyer_rate"]], "product_summary")
    for level in ["department", "aisle"]:
        g = ps.groupby(level)
        s = pd.DataFrame({"n_products": g.size(), "purchases": g["product_total_purchases"].sum(),
                          "reorders": g["product_reorder_count"].sum()}).reset_index()
        s["reorder_rate"] = s["reorders"] / s["purchases"].replace(0, np.nan)
        save_table(s.sort_values("purchases", ascending=False), f"{level}_summary")

    dh = orders.groupby(["order_dow", "order_hour_of_day"]).size().rename("orders").reset_index()
    save_table(dh, "orders_by_dow_hour")
    days = orders["days_since_prior_order"]
    dhist = days.value_counts().sort_index().rename_axis("days_since_prior_order").rename("orders").reset_index()
    save_table(dhist, "days_since_prior_hist")
    per_user = orders.groupby("user_id")["order_number"].max()
    upu = per_user.value_counts().sort_index().rename_axis("orders_per_user").rename("users").reset_index()
    save_table(upu, "orders_per_user_hist")
    sizes = order_size[prior_orders["order_id"].to_numpy()]
    bs = np.bincount(sizes)
    save_table(pd.DataFrame({"basket_size": np.arange(len(bs)), "orders": bs}).query("orders > 0"), "basket_size_hist")

    d, h = np.divmod(np.arange(168), 24)
    save_table(pd.DataFrame({"order_dow": d, "order_hour_of_day": h, "items": dowhour_items, "reordered": dowhour_reord}),
               "reorder_by_dow_hour")
    save_table(pd.DataFrame({"order_number": np.arange(101), "items": onum_items, "reordered": onum_reord}).query("items > 0"),
               "reorder_by_order_number")
    save_table(pd.DataFrame({"cart_position": np.arange(51), "items": cart_items, "reordered": cart_reord}).query("items > 0"),
               "reorder_by_cart_position")

    # ---- overview / manifest ---------------------------------------------------
    file_info = {}
    for name, fname in FILES.items():
        p = DATA_DIR / fname
        file_info[fname] = {"size_mb": round(p.stat().st_size / 1e6, 1)}
    small_mem = {n: round(df.memory_usage(deep=True).sum() / 1e6, 1)
                 for n, df in {"orders": orders, "products": products, "aisles": aisles, "departments": departments}.items()}
    overview = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "files": file_info,
        "in_memory_mb_small_tables": small_mem,
        "orders": {"rows": int(len(orders)), "columns": int(orders.shape[1]),
                   "by_eval_set": {k: int(v) for k, v in orders["eval_set"].astype(str).value_counts().items()},
                   "missing": {k: int(v) for k, v in orders.isna().sum().items()},
                   "duplicates_removed": 0},
        "order_products_prior": {"rows": stats["rows"], "columns": 4, "missing": stats["missing"],
                                 "duplicate_rows": stats["duplicate_rows"],
                                 "reordered_rows": stats["reordered_rows"],
                                 "reorder_rate": stats["reordered_rows"] / max(stats["rows"], 1),
                                 "streamed_in_chunks": stats["chunks"],
                                 "compact_in_memory_mb_if_loaded_whole": round(stats["rows"] * 11 / 1e6, 1)},
        "order_products_train": {"rows": int(len(train)), "columns": 4, "reordered_rows": int(train["reordered"].sum()),
                                 "reorder_rate": float(train["reordered"].mean()),
                                 "orders": int(train["order_id"].nunique())},
        "products": {"rows": int(len(products)), "columns": int(products.shape[1]),
                     "purchased_in_prior": int(seen_product.sum())},
        "aisles": {"rows": int(len(aisles))}, "departments": {"rows": int(len(departments))},
        "unique_users": int(orders["user_id"].nunique()),
        "unique_orders": int(orders["order_id"].nunique()),
        "prior_orders": int(len(prior_orders)),
        "user_product_pairs_full": int(len(up_full)),
        "chunk_rows": int(chunk_rows),
    }
    (PROCESSED_DIR / "overview.json").write_text(json.dumps(overview, indent=2))
    MANIFEST_PATH.write_text(json.dumps({"built_at": overview["built_at"], "chunk_rows": chunk_rows,
                                         "prior_orders": overview["prior_orders"]}, indent=2))
    _log("done - processed tables written to " + str(PROCESSED_DIR), t0)
    return overview


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build CartIQ aggregate tables from the raw Instacart CSVs.")
    ap.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    build_all(ap.parse_args().chunk_rows)
