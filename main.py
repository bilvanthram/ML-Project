"""
main.py - build all aggregate tables from the raw Instacart CSVs (run once, or after changing the data).

    python main.py                 # skips work already done
    python main.py --rebuild       # rebuild everything
    python main.py --top-n 800     # market-basket: number of top products considered

Never builds the 32M-row merged DataFrame: the big prior file is streamed in chunks and reduced to small
tables in outputs/processed/ (see src/data_processing.py). The old merged-DataFrame EDA in src/eda.py is kept
only for the notebook on samples/small data.
"""

import argparse

from src.basket_analysis import META_PATH, build_basket_tables
from src.data_loading import DEFAULT_CHUNK_ROWS
from src.data_processing import build_all, processed_data_ready


def build_data(rebuild=False, top_n=500, min_pair_count=25, chunk_rows=DEFAULT_CHUNK_ROWS):
    if rebuild or not processed_data_ready():
        build_all(chunk_rows)
        rebuild = True  # basket tables depend on processed tables
    else:
        print("Processed tables already exist - skipping (use --rebuild to redo).")
    if rebuild or not META_PATH.exists():
        print("Building market-basket pair counts ...")
        n = build_basket_tables(top_n, min_pair_count, chunk_rows)
        print(f"Market basket: {n:,} product pairs kept.")
    else:
        print("Market-basket tables already exist - skipping.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--top-n", type=int, default=500)
    ap.add_argument("--min-pair-count", type=int, default=25)
    ap.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    a = ap.parse_args()
    build_data(a.rebuild, a.top_n, a.min_pair_count, a.chunk_rows)
