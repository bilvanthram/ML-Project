# CartIQ

Instacart grocery analytics, market-basket analysis and reorder prediction (Streamlit).
Group 13 - MOHNISH RAJ YADAV (2500030928), DARIPALLY BILVANTH RAM (2500031922)

## Quick start (Windows or Linux/macOS, from the project folder)

```bash
pip install -r requirements.txt
# put the 6 Instacart CSVs in data/  (see data/README.md) - they are never modified
python main.py          # once: raw CSVs -> small aggregate tables (streams the 32M-row file in chunks)
python train.py         # once (or to retrain): builds features, trains, evaluates, saves the model
streamlit run app.py    # http://localhost:8501
```

`python train.py --extended` additionally trains Decision Tree, Random Forest, AdaBoost, Gradient Boosting and XGBoost
(syllabus comparison; trained on a 300k-row subsample, printed when it happens). `--train-users N` / `--eval-users N`
change the sample sizes (defaults 40,000 / 20,000).

## Architecture (why startup is fast)

The old flow merged every table into one 32,434,489-row DataFrame before the UI could render. That merge is gone.

| Layer | Code | Output |
|---|---|---|
| Raw data (read-only) | `data/*.csv` | - |
| Loading (compact dtypes, order-aligned chunks) | `src/data_loading.py` | - |
| Cleaning + aggregation | `src/data_processing.py` | `outputs/processed/*.parquet` (product/department/aisle/time/reorder tables, user & user-product history) |
| Market basket | `src/basket_analysis.py` | pair counts of the top-N products |
| Feature table (leakage-safe) | `src/feature_engineering.py` | built on demand for training/serving |
| Preprocessing | `src/preprocessing.py` | part of the saved Pipeline |
| Train / evaluate / save | `src/ml_pipeline.py` (`train.py`) | `models/cartiq_model.pkl`, `models/model_metadata.json`, `outputs/metrics`, `outputs/figures` |
| Serving | `src/model_utils.py`, `app.py` | Streamlit |

The app imports only pandas/altair/streamlit at startup, reads small tables per page (`@st.cache_data`), loads the model
only on the ML/Predict pages (`@st.cache_resource`), and never reads the raw CSVs or trains.

## ML formulation

**Target `will_reorder`**: for a user about to place a new order and a product the user bought before (the candidate),
1 if the product appears in that order, else 0. Binary classification.

**Prediction points (temporal, user-disjoint split)**

- *Train*: history = a user's orders `1..n-1`, target = order `n` (their last prior order).
- *Validation / Test*: history = orders `1..n`, target = the labelled next order (Kaggle `train` set), **different users**,
  later in the timeline than the training point. Validation picks the model and the decision threshold; test is reported once.

**Leakage prevention**

1. All history features come only from orders before the target order. The `*_hist` tables exclude every user's order `n`
   entirely, including from global product statistics.
2. The target order's basket is never a feature - in particular not its `add_to_cart_order`. Candidates come from history, not
   from what the target order contains.
3. `user_id` / `product_id` are never model inputs; train/validation/test users are disjoint.
4. `up_reorder_count` is not a feature: in Instacart it always equals `purchase_count - 1` (redundant).
5. Preprocessing lives in the same sklearn Pipeline as the model, so serving = training.

**Features (21)**: order context (`order_dow`, `order_hour_of_day`, `days_since_prior_order` of the target order - known when an
order starts); user (`user_total_orders`, `user_total_products`, `user_avg_basket_size`, `user_reorder_rate`,
`user_avg_days_between_orders`, `user_distinct_products`); product (`product_total_purchases`, `product_reorder_count`,
`product_reorder_rate`, `product_unique_users`, `product_avg_cart_position`, `product_repeat_buyer_rate`); user-product
(`up_purchase_count`, `up_purchase_share`, `up_orders_since_last`, `up_avg_cart_position`); categorical `department`, `aisle`.

**Models**: Logistic Regression (L2, scaled + one-hot; interpretable baseline) vs HistGradientBoosting, plus a no-ML reference
("rank by share of the user's orders containing the product"). Selection: highest **validation ROC-AUC**. The reorder class is
a minority, so accuracy is not used to choose; ROC-AUC, PR-AUC, precision, recall and F1 (at the validation-tuned threshold)
and the confusion matrix are reported. The class distribution is printed by `train.py` and shown on the *ML Results* page.

## Market basket analysis

Top-N products (default 500). For each order-aligned chunk a sparse orders x products matrix `X` is built and `X^T X` is
accumulated (diagonal = orders containing a product, off-diagonal = orders containing both). Rules A->B:
support = both / all orders; confidence = both / orders with A; lift = confidence / (orders with B / all orders).
Pairs only (1 item -> 1 item); thresholds are adjustable in the UI.

## Results

No numbers are hard-coded anywhere. After `python train.py`, see `outputs/metrics/model_comparison.csv` (validation and test
metrics for every model), the figures in `outputs/figures/`, and the *ML Results* page.

## Notes and limitations

- The dashboard is verified on synthetic Instacart-schema data (scaling tests up to 5.5M prior rows); memory/time on the
  real 32M-row file were not measured by the author of this refactor. If RAM is tight use `python main.py --chunk-rows 1000000`.
- `notebooks/CartIQ_EDA.ipynb` and `src/eda.py` are the original merged-DataFrame EDA and run on a 1M-row sample only.
- Day-of-week coding (0-6) is undocumented in the dataset; the app labels them "Day 0-6".
- No hyperparameter search in this version (fixed sensible settings); no monitoring/automated retraining.
- Validation is used both for model choice and threshold; the separate test users give the unbiased estimate.
