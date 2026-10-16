"""
train.py - (re)train the reorder model and save it to models/.

    python train.py                        # builds processed tables first if missing
    python train.py --train-users 80000    # more training users (slower, usually slightly better)
    python train.py --extended             # also run Decision Tree / RF / AdaBoost / GB / XGBoost

The Streamlit app only LOADS the saved model; it never trains.
"""

import argparse

from main import build_data
from src.ml_pipeline import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-users", type=int, default=40_000)
    ap.add_argument("--eval-users", type=int, default=20_000)
    ap.add_argument("--extended", action="store_true")
    ap.add_argument("--rebuild-data", action="store_true")
    a = ap.parse_args()
    build_data(rebuild=a.rebuild_data)
    run(a.train_users, a.eval_users, a.extended)
