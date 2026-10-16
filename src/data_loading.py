"""
data_loading.py - memory-efficient loading of the raw Instacart CSVs.

Key ideas
---------
* Explicit compact dtypes (int32/int16/int8/float32) instead of pandas' default
  int64: roughly halves memory for the big tables.
* `order_products__prior.csv` (32M rows) is NEVER loaded whole. `iter_order_chunks`
  streams it in chunks that never split an order across two chunks, so every
  per-order aggregate computed on a chunk is exact.
* Small tables (orders, products, aisles, departments) are loaded whole.
"""

import pandas as pd

from src.paths import DATA_DIR

FILES = {
    "orders": "orders.csv",
    "order_products_prior": "order_products__prior.csv",
    "order_products_train": "order_products__train.csv",
    "products": "products.csv",
    "aisles": "aisles.csv",
    "departments": "departments.csv",
}

DTYPES = {
    "orders": {"order_id": "int32", "user_id": "int32", "eval_set": "category",
               "order_number": "int16", "order_dow": "int8", "order_hour_of_day": "int8",
               "days_since_prior_order": "float32"},
    "order_products_prior": {"order_id": "int32", "product_id": "int32",
                             "add_to_cart_order": "int16", "reordered": "int8"},
    "order_products_train": {"order_id": "int32", "product_id": "int32",
                             "add_to_cart_order": "int16", "reordered": "int8"},
    "products": {"product_id": "int32", "aisle_id": "int16", "department_id": "int16"},
    "aisles": {"aisle_id": "int16"},
    "departments": {"department_id": "int16"},
}

# Kaggle's order_products files are sorted by order_id; the whole chunking
# strategy depends on it, so it is verified while streaming (not assumed).
DEFAULT_CHUNK_ROWS = 4_000_000


def check_files_exist():
    return {name: (DATA_DIR / fname).exists() for name, fname in FILES.items()}


def require_files(names=None):
    """Raise a clear error listing missing raw files."""
    status = check_files_exist()
    names = names or list(FILES)
    missing = [FILES[n] for n in names if not status[n]]
    if missing:
        raise FileNotFoundError(
            f"Missing raw dataset file(s) in {DATA_DIR}: {', '.join(missing)}. "
            "Download the Instacart Market Basket Analysis dataset and place the CSVs "
            "in the data/ folder (see data/README.md)."
        )


def read_table(name, usecols=None):
    """Load one raw table with compact dtypes."""
    require_files([name])
    dtypes = {c: t for c, t in DTYPES[name].items() if usecols is None or c in usecols}
    return pd.read_csv(DATA_DIR / FILES[name], dtype=dtypes, usecols=usecols)


def iter_order_chunks(name, chunk_rows=DEFAULT_CHUNK_ROWS):
    """
    Stream an order_products file in chunks that never split an order.

    The last (possibly incomplete) order of each raw chunk is carried over to the
    next chunk. Raises if the file is not sorted by order_id.
    """
    require_files([name])
    reader = pd.read_csv(DATA_DIR / FILES[name], dtype=DTYPES[name], chunksize=chunk_rows)
    carry = None
    prev_max = -1
    for raw in reader:
        chunk = raw if carry is None else pd.concat([carry, raw], ignore_index=True)
        ids = chunk["order_id"].to_numpy()
        if not (ids[1:] >= ids[:-1]).all():
            raise ValueError(f"{FILES[name]} is not sorted by order_id; chunked processing "
                             "requires the original Kaggle ordering.")
        if ids[0] <= prev_max:
            raise ValueError(f"{FILES[name]}: an order appears in more than one place.")
        last_id = ids[-1]
        tail = ids == last_id
        carry = chunk[tail]
        done = chunk[~tail]
        if len(done):
            prev_max = int(done["order_id"].iloc[-1])
            yield done
    if carry is not None and len(carry):
        yield carry


def load_datasets(prior_rows=None):
    """
    Legacy helper for the EDA notebook. Loads the small tables + the train file (1.4M rows).
    The 32M-row prior file is only included if `prior_rows` is given, and then only its FIRST
    `prior_rows` rows (a sample) - never the whole file. The dashboard does not use this function.
    """
    out = {}
    for name in ["orders", "products", "aisles", "departments", "order_products_train"]:
        if (DATA_DIR / FILES[name]).exists():
            out[name] = read_table(name)
    if prior_rows:
        require_files(["order_products_prior"])
        out["order_products_prior"] = pd.read_csv(DATA_DIR / FILES["order_products_prior"],
                                                  dtype=DTYPES["order_products_prior"], nrows=int(prior_rows))
    return out
