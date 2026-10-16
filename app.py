"""
app.py - CartIQ Streamlit dashboard.

    streamlit run app.py

Design rules
* Startup loads NOTHING heavy: only pandas/altair/streamlit are imported. The 713 MB raw CSVs are never
  read by the app; it reads the small pre-aggregated tables in outputs/processed/ (built by `python main.py`).
* Each page loads only the tables it needs, inside @st.cache_data, behind a spinner.
* The model is only loaded (cache_resource) on the ML / Predict pages, and is never trained here.
"""

import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_processing import load_table, processed_data_ready  # noqa: E402
from src.paths import FIGURES_DIR, METADATA_PATH, METRICS_DIR, MODEL_PATH, PROCESSED_DIR  # noqa: E402

st.set_page_config(page_title="CartIQ", page_icon="🛒", layout="wide")

PAGES = ["Overview", "Dataset", "EDA / Insights", "Product Analysis", "Reorder Analysis",
         "Time Analysis", "Market Basket Analysis", "ML Results", "Predict / Recommend"]
DAY_LABELS = {i: f"Day {i}" for i in range(7)}


# ----------------------------------------------------------------------------- cached loaders
@st.cache_data(show_spinner=False)
def T(name):
    """Small processed table (deterministic, read-only -> safe for cache_data)."""
    return load_table(name)


@st.cache_data(show_spinner=False)
def overview():
    return json.loads((PROCESSED_DIR / "overview.json").read_text())


@st.cache_data(show_spinner=False)
def metadata():
    return json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else None


@st.cache_resource(show_spinner=False)
def model_pipeline(stamp):
    """Load the saved sklearn Pipeline once per process. `stamp` (training time) invalidates the cache on retrain."""
    import joblib
    return joblib.load(MODEL_PATH)


def bar_h(df, x, y, title, color=None, height=None):
    enc = dict(x=alt.X(x, title=x.replace("_", " ")), y=alt.Y(y, sort="-x", title=None),
               tooltip=list(df.columns))
    if color:
        enc["color"] = alt.Color(color, scale=alt.Scale(scheme="blues"), legend=None)
    st.altair_chart(alt.Chart(df, title=title).mark_bar().encode(**enc).properties(height=height or max(220, 22 * len(df))))


def bar_v(df, x, y, title, x_type="O"):
    st.altair_chart(alt.Chart(df, title=title).mark_bar().encode(
        x=alt.X(f"{x}:{x_type}", title=x.replace("_", " ")), y=alt.Y(y, title=y.replace("_", " ")),
        tooltip=list(df.columns)).properties(height=280))


def need_data():
    if not processed_data_ready():
        st.warning("Processed tables not found. Build them once from the raw CSVs (data/ folder) with:")
        st.code("python main.py", language="bash")
        st.caption("This streams the 32M-row file in chunks and writes small tables to outputs/processed/. "
                   "It takes a few minutes on the full dataset and is only needed once.")
        st.stop()


# ----------------------------------------------------------------------------- pages
def page_overview():
    st.title("CartIQ 🛒")
    st.subheader("Instacart grocery analytics, market-basket analysis and reorder prediction")
    if processed_data_ready():
        ov = overview()
        c = st.columns(4)
        c[0].metric("Users", f"{ov['unique_users']:,}")
        c[1].metric("Orders", f"{ov['unique_orders']:,}")
        c[2].metric("Products", f"{ov['products']['rows']:,}")
        c[3].metric("Prior order-product rows", f"{ov['order_products_prior']['rows']:,}")
    else:
        st.info("Run `python main.py` to build the processed tables; the dashboard fills in afterwards.")
    st.markdown("""
**What this app does**

| Stage | Where |
|---|---|
| Raw CSVs (never modified) | `data/` |
| Cleaning + chunked aggregation (no 32M-row merge) | `src/data_processing.py` → `outputs/processed/` |
| EDA / product / reorder / time analysis | pages in this app, reading small aggregate tables |
| Market basket (support, confidence, lift) | `src/basket_analysis.py` |
| Leakage-safe feature table | `src/feature_engineering.py` |
| Train + evaluate + save model | `src/ml_pipeline.py` (`python train.py`) → `models/` |
| Prediction / recommendation | *Predict / Recommend* page (loads the saved model) |

**ML task.** For a user starting a new order and a product they bought before, predict whether the product
will be in that order (binary classification, `will_reorder` = 0/1).
""")
    m = metadata()
    if m:
        st.success(f"Trained model available: **{m['model_name']}** (trained {m['trained_at']}). See *ML Results*.")


def page_dataset():
    need_data(); ov = overview()
    st.header("Dataset")
    st.caption("Counts are computed while streaming the raw files once; nothing here holds the raw data in memory.")
    rows = []
    for key, fname in [("orders", "orders.csv"), ("order_products_prior", "order_products__prior.csv"),
                       ("order_products_train", "order_products__train.csv"), ("products", "products.csv"),
                       ("aisles", "aisles.csv"), ("departments", "departments.csv")]:
        t = ov[key]
        rows.append({"file": fname, "rows": t["rows"], "columns": t.get("columns", 2),
                     "size on disk (MB)": ov["files"][fname]["size_mb"],
                     "missing cells": sum(t["missing"].values()) if isinstance(t.get("missing"), dict) else 0,
                     "duplicate rows": t.get("duplicate_rows", t.get("duplicates_removed", 0))})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    c = st.columns(4)
    c[0].metric("Unique users", f"{ov['unique_users']:,}"); c[1].metric("Unique orders", f"{ov['unique_orders']:,}")
    c[2].metric("Products", f"{ov['products']['rows']:,}")
    c[3].metric("Products bought in prior orders", f"{ov['products']['purchased_in_prior']:,}")
    st.markdown("**Memory.** Small tables in memory: " + ", ".join(f"{k} {v} MB" for k, v in ov["in_memory_mb_small_tables"].items())
                + f". The prior file would need ≈ {ov['order_products_prior']['compact_in_memory_mb_if_loaded_whole']:,} MB "
                  f"even with compact dtypes (before any merge), so it is streamed in {ov['order_products_prior']['streamed_in_chunks']} "
                  "order-aligned chunks instead.")
    st.markdown("**Missing values.** The only expected missing value is `days_since_prior_order` on each user's first order "
                f"({ov['orders']['missing'].get('days_since_prior_order', 0):,} orders) - kept as missing, not filled with 0.")
    st.subheader("Orders by `eval_set`")
    st.dataframe(pd.DataFrame({"eval_set": list(ov["orders"]["by_eval_set"]), "orders": list(ov["orders"]["by_eval_set"].values())}),
                 hide_index=True)
    st.caption("`prior` = history, `train` = each user's latest order with labels, `test` = latest order without labels.")


def page_eda():
    need_data(); ov = overview()
    st.header("EDA / Insights")
    per_user = T("orders_per_user_hist"); basket = T("basket_size_hist"); ps = T("product_summary"); dep = T("department_summary")
    with st.spinner("Preparing charts..."):
        c1, c2 = st.columns(2)
        with c1:
            bar_v(per_user, "orders_per_user", "users", "Orders per user (all eval sets)", "Q")
        with c2:
            bar_v(basket[basket["basket_size"] <= 60], "basket_size", "orders", "Basket size (items per prior order, ≤ 60 shown)", "Q")
    n_orders = basket["orders"].sum()
    mean_basket = (basket["basket_size"] * basket["orders"]).sum() / n_orders
    cum = basket["orders"].cumsum() / n_orders
    median_basket = int(basket.loc[cum >= 0.5, "basket_size"].iloc[0])
    p99 = int(basket.loc[cum >= 0.99, "basket_size"].iloc[0])
    reorder_rate = ov["order_products_prior"]["reorder_rate"]
    top_dep = dep.iloc[0]
    st.subheader("What the data says (computed from the tables, not hard-coded)")
    st.markdown(f"""
- **{reorder_rate:.1%}** of all purchased items in prior orders are reorders → most of the shopping is repeat buying,
  which is why reorder prediction is a natural ML task.
- Basket size is right-skewed: mean **{mean_basket:.1f}**, median **{median_basket}**, 99th percentile **{p99}** items.
  Very large baskets are genuine bulk orders, so they are **kept** (a tree model is robust to them; for Logistic Regression the
  features are standardised).
- Order counts per user are capped by the dataset design (users have 4-100 orders); the long right tail is real behaviour, not error.
- The busiest department is **{top_dep['department']}** with {top_dep['purchases']:,} purchases ({top_dep['reorder_rate']:.1%} reorder rate).
- Only {ov['products']['purchased_in_prior']:,} of {ov['products']['rows']:,} catalogue products appear in prior orders; product popularity is
  heavily concentrated - see *Product Analysis*.
""")
    st.subheader("Outlier decisions")
    st.markdown("""
| Variable | Observation | Decision |
|---|---|---|
| `days_since_prior_order` | capped at 30 by the dataset; large spike at 30 and at 7 | **Keep** - real signal (monthly/weekly shoppers) |
| Basket size | long right tail | **Keep**, no capping needed for tree model; scaled for Logistic Regression |
| Product purchase counts | extremely skewed (bananas etc.) | **Keep**; ratio features (reorder rate) used alongside raw counts |
""")
    st.subheader("Which variables look useful for prediction?")
    st.markdown("Judge this with the model, not with EDA alone: see the permutation-importance chart on *ML Results*. "
                "EDA suggests user-product history (how often and how recently the user bought the item) and product reorder rate.")


def page_products():
    need_data()
    st.header("Product Analysis")
    ps = T("product_summary"); dep = T("department_summary"); aisle = T("aisle_summary")
    tab1, tab2, tab3, tab4 = st.tabs(["Most purchased", "Most reordered", "Departments", "Aisles"])
    with tab1:
        n = st.slider("Number of products", 5, 50, 20, key="np1")
        bar_h(ps.nlargest(n, "product_total_purchases")[["product_name", "product_total_purchases", "product_reorder_rate"]],
              "product_total_purchases", "product_name", f"Top {n} products by purchases", color="product_reorder_rate")
    with tab2:
        c = st.columns(2)
        n = c[0].slider("Number of products", 5, 50, 20, key="np2")
        min_p = c[1].slider("Minimum purchases (avoid tiny-sample noise)", 10, 5000, 500, step=10)
        sel = ps[ps["product_total_purchases"] >= min_p].nlargest(n, "product_reorder_rate")
        bar_h(sel[["product_name", "product_reorder_rate", "product_total_purchases"]], "product_reorder_rate", "product_name",
              f"Highest reorder rate (≥ {min_p} purchases)")
        st.caption("Reorder rate = reorders / purchases for that product, over all prior orders.")
    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            bar_h(dep[["department", "purchases"]], "purchases", "department", "Purchases by department")
        with c2:
            bar_h(dep.sort_values("reorder_rate", ascending=False)[["department", "reorder_rate"]], "reorder_rate", "department",
                  "Reorder rate by department")
    with tab4:
        n = st.slider("Number of aisles", 5, 40, 20, key="na")
        c1, c2 = st.columns(2)
        with c1:
            bar_h(aisle.nlargest(n, "purchases")[["aisle", "purchases"]], "purchases", "aisle", f"Top {n} aisles by purchases")
        with c2:
            bar_h(aisle[aisle["purchases"] > 0].nlargest(n, "reorder_rate")[["aisle", "reorder_rate"]], "reorder_rate", "aisle",
                  f"Top {n} aisles by reorder rate")
    st.subheader("Product lookup")
    q = st.text_input("Search product name")
    view = ps[ps["product_name"].str.contains(q, case=False, na=False)] if q else ps.nlargest(50, "product_total_purchases")
    st.dataframe(view.head(200), hide_index=True)


def page_reorder():
    need_data(); ov = overview()
    st.header("Reorder Analysis")
    p, t = ov["order_products_prior"], ov["order_products_train"]
    c = st.columns(4)
    c[0].metric("Prior items reordered", f"{p['reorder_rate']:.1%}"); c[1].metric("Prior items first-time", f"{1 - p['reorder_rate']:.1%}")
    c[2].metric("Train-order items reordered", f"{t['reorder_rate']:.1%}"); c[3].metric("Prior reordered rows", f"{p['reordered_rows']:,}")
    st.caption("Item level. NOTE: the ML task below is candidate level (products a user bought before), so its positive "
               "rate is different - see *ML Results* for the class distribution actually used.")
    by_n = T("reorder_by_order_number").assign(reorder_rate=lambda d: d["reordered"] / d["items"])
    by_c = T("reorder_by_cart_position").assign(reorder_rate=lambda d: d["reordered"] / d["items"])
    c1, c2 = st.columns(2)
    with c1:
        bar_v(by_n, "order_number", "reorder_rate", "Reorder share of items by order number (100 = 100+)", "Q")
        st.caption("Later orders contain more repeat items: history accumulates.")
    with c2:
        bar_v(by_c, "cart_position", "reorder_rate", "Reorder share by add-to-cart position (50 = 50+)", "Q")
        st.caption("Items added first are much more often reorders. Position of the CURRENT order is not used as an ML feature (it would leak).")
    ps = T("product_summary")
    min_p = st.slider("Minimum purchases for product-level distribution", 10, 2000, 100, step=10)
    d = ps[ps["product_total_purchases"] >= min_p]
    st.altair_chart(alt.Chart(d, title=f"Product reorder rate distribution ({len(d):,} products)").mark_bar().encode(
        x=alt.X("product_reorder_rate:Q", bin=alt.Bin(maxbins=30), title="product reorder rate"), y="count()").properties(height=260))


def page_time():
    need_data()
    st.header("Time Analysis")
    dh = T("orders_by_dow_hour").assign(day=lambda d: d["order_dow"].map(DAY_LABELS))
    st.caption("The dataset does not document which day is 0 (commonly assumed to be Sunday), so days are shown as Day 0-6.")
    c1, c2 = st.columns(2)
    with c1:
        bar_v(dh.groupby("order_hour_of_day", as_index=False)["orders"].sum(), "order_hour_of_day", "orders", "Orders by hour of day", "Q")
    with c2:
        bar_v(dh.groupby("day", as_index=False)["orders"].sum(), "day", "orders", "Orders by day of week")
    st.altair_chart(alt.Chart(dh, title="Orders by day and hour").mark_rect().encode(
        x=alt.X("order_hour_of_day:O", title="hour"), y=alt.Y("day:O", title=None),
        color=alt.Color("orders:Q", scale=alt.Scale(scheme="blues")), tooltip=["day", "order_hour_of_day", "orders"]).properties(height=260))
    days = T("days_since_prior_hist")
    bar_v(days, "days_since_prior_order", "orders", "Days since prior order (spikes at 7-day multiples and the 30-day cap)", "Q")
    st.caption("30 is a cap: it means '30 or more days'. First orders have no value (missing).")
    rr = T("reorder_by_dow_hour").assign(reorder_rate=lambda d: d["reordered"] / d["items"], day=lambda d: d["order_dow"].map(DAY_LABELS))
    st.altair_chart(alt.Chart(rr, title="Reorder share of items by day and hour").mark_rect().encode(
        x=alt.X("order_hour_of_day:O", title="hour"), y=alt.Y("day:O", title=None),
        color=alt.Color("reorder_rate:Q", scale=alt.Scale(scheme="greens")), tooltip=["day", "order_hour_of_day", "reorder_rate"]).properties(height=260))


def page_basket():
    need_data()
    st.header("Market Basket Analysis")
    from src.basket_analysis import META_PATH, derive_rules, load_basket_meta
    if not META_PATH.exists():
        st.warning("Market-basket tables not built yet. Run `python main.py`.")
        st.stop()
    meta = load_basket_meta()
    st.markdown(f"""
Which products are bought **together**? Computed on the **top {meta['top_n']} products** over **{meta['n_orders']:,} prior orders**
(pair co-occurrence counted chunk by chunk; pairs seen in fewer than {meta['min_pair_count']} orders are dropped).

- **Support** = share of all orders containing both A and B.
- **Confidence** (A → B) = of the orders containing A, the share that also contain B.
- **Lift** = confidence ÷ (share of orders containing B). Lift > 1 means A and B appear together more than chance; ≈ 1 means independent.
""")
    with st.spinner("Loading rules..."):
        pairs, items = T("basket_pairs"), T("basket_items")
    c = st.columns(4)
    min_sup = c[0].select_slider("Min support", options=[0.0, 0.0005, 0.001, 0.002, 0.005, 0.01], value=0.001, format_func=lambda v: f"{v:.2%}")
    min_conf = c[1].slider("Min confidence", 0.0, 1.0, 0.1, 0.05)
    min_lift = c[2].slider("Min lift", 0.0, 10.0, 1.5, 0.1)
    top_k = c[3].slider("Rows", 10, 200, 30, 10)
    rules = derive_rules(pairs, items, meta["n_orders"], min_sup, min_conf, min_lift)
    q = st.text_input("Filter by product name (antecedent or consequent)")
    if q:
        rules = rules[rules["antecedent"].str.contains(q, case=False, na=False) | rules["consequent"].str.contains(q, case=False, na=False)]
    sort_by = st.radio("Sort by", ["lift", "confidence", "support"], horizontal=True)
    rules = rules.sort_values(sort_by, ascending=False)
    st.write(f"{len(rules):,} rules match.")
    if rules.empty:
        st.info("No rules match - lower the thresholds."); return
    top = rules.iloc[0]
    st.info(f"**Example:** {top['antecedent']} → {top['consequent']} - support {top['support']:.3%}, confidence {top['confidence']:.1%}, lift {top['lift']:.2f}. "
            f"Of orders containing *{top['antecedent']}*, {top['confidence']:.1%} also contain *{top['consequent']}*, "
            f"{top['lift']:.1f}× the base rate.")
    st.dataframe(rules.head(top_k).style.format({"support": "{:.3%}", "confidence": "{:.1%}", "lift": "{:.2f}"}), hide_index=True)
    sample = rules.head(2000)
    st.altair_chart(alt.Chart(sample, title="Support vs confidence (colour = lift)").mark_circle(size=40).encode(
        x=alt.X("support:Q", scale=alt.Scale(type="log")), y="confidence:Q", color=alt.Color("lift:Q", scale=alt.Scale(scheme="viridis")),
        tooltip=["antecedent", "consequent", "support", "confidence", "lift"]).properties(height=320))
    st.caption("Popular items (e.g. bananas) have high confidence with everything but low lift; small-support pairs can have very high "
               "lift by chance - use the support slider before trusting a rule.")


def page_ml():
    st.header("ML Results")
    m = metadata()
    if m is None:
        st.warning("No trained model yet. Train once with:"); st.code("python train.py", language="bash"); st.stop()
    st.markdown(f"**Selected model: {m['model_name']}** - trained {m['trained_at']}  \n**Target:** {m['target_definition']}")
    st.markdown(f"**Selection rule:** {m['selection_rule']}.  **Decision threshold:** {m['decision_threshold']:.3f} ({m['threshold_rule']}).")
    st.subheader("Class distribution (candidate products)")
    cd = pd.DataFrame(m["class_distribution"]).T.reset_index().rename(columns={"index": "set"})
    st.dataframe(cd.style.format({"rows": "{:,}", "positives": "{:,}", "negatives": "{:,}", "positive_rate": "{:.3f}"}), hide_index=True)
    st.caption("Imbalanced: accuracy alone would look good by predicting 'not reordered' for everything, so ROC-AUC, PR-AUC, precision, recall and F1 are reported.")
    path = METRICS_DIR / "model_comparison.csv"
    if path.exists():
        st.subheader("Model comparison")
        tbl = pd.read_csv(path)
        show = tbl[["model", "train_rows", "valid_roc_auc", "valid_pr_auc", "valid_precision", "valid_recall", "valid_f1",
                    "test_roc_auc", "test_pr_auc", "test_precision", "test_recall", "test_f1"]]
        st.dataframe(show.style.format({c: "{:.4f}" for c in show.columns if c not in ("model", "train_rows")} | {"train_rows": "{:,}"}), hide_index=True)
        st.caption("Validation numbers picked the model and threshold; test numbers come from different users and were used once. "
                   "Precision/recall/F1 use the validation-tuned threshold.")
    c1, c2 = st.columns(2)
    if (FIGURES_DIR / "roc_curves.png").exists():
        c1.image(str(FIGURES_DIR / "roc_curves.png"), caption="Validation ROC curves")
    if (FIGURES_DIR / "confusion_matrix.png").exists():
        c2.image(str(FIGURES_DIR / "confusion_matrix.png"), caption="Confusion matrix on the test users")
    if (FIGURES_DIR / "permutation_importance.png").exists():
        st.image(str(FIGURES_DIR / "permutation_importance.png"),
                 caption="Permutation importance (how much validation ROC-AUC drops when a feature is shuffled). Importance is not causation.")
    st.subheader("How leakage is prevented")
    st.markdown(f"""
- **Temporal, user-disjoint split.** {m['split']['scheme']}. Users: train {m['split']['users_train']:,}, validation {m['split']['users_validation']:,}, test {m['split']['users_test']:,}.
- Every history feature uses only orders **before** the target order; the target order's basket (including its add-to-cart positions) is never a feature.
- Candidates come from the user's purchase history, not from the target basket.
- Preprocessing is part of the saved sklearn Pipeline, so serving = training.
""")
    with st.expander("Features used"):
        st.write(m["numeric_features"] + m["categorical_features"])


def page_predict():
    st.header("Predict / Recommend")
    m = metadata()
    if m is None or not processed_data_ready():
        st.warning("Needs processed tables and a trained model:"); st.code("python main.py\npython train.py", language="bash"); st.stop()
    from src.model_utils import actual_next_basket, default_order_context, recommend
    st.markdown("Pick a user. CartIQ scores **every product that user has bought before** with the saved "
                f"**{m['model_name']}** model and ranks them by the probability they appear in the user's next order.")
    ctx_tbl = T("user_order_context")
    demo = ctx_tbl.loc[ctx_tbl["eval_set"] == "train", "user_id"].head(5).tolist()
    with st.form("predict"):
        c = st.columns([1, 1, 1])
        uid = c[0].number_input("User ID", min_value=1, value=int(demo[0]) if demo else 1, step=1)
        top_k = c[1].slider("Products to show", 5, 50, 15)
        c[2].caption(f"Example users with known next order: {demo}")
        override = st.checkbox("Override next-order context (day / hour / days since last order)")
        oc = st.columns(3)
        dow = oc[0].selectbox("Day (0-6)", list(range(7)), index=0)
        hour = oc[1].slider("Hour", 0, 23, 12)
        gap = oc[2].slider("Days since last order", 0, 30, 7)
        go = st.form_submit_button("Predict / Recommend", type="primary")
    if go:
        st.session_state["rec_args"] = (int(uid), top_k, override, int(dow), int(hour), float(gap))
    if "rec_args" not in st.session_state:
        return
    uid, top_k, override, dow, hour, gap = st.session_state["rec_args"]
    default_ctx, source = default_order_context(uid)
    if default_ctx is None:
        st.error(f"User {uid} not found in the dataset."); return
    ctx = {"order_dow": dow, "order_hour_of_day": hour, "days_since_prior_order": gap} if override else default_ctx
    with st.spinner("Scoring the user's products with the saved model..."):
        pipe = model_pipeline(m["trained_at"])
        ps_full, lookup = T("product_full"), T("products_lookup")
        recs = recommend(uid, pipe, m["decision_threshold"], ctx, ps_full, lookup)
    if recs is None:
        st.error("This user has no purchase history."); return
    st.caption(f"Order context used: day {ctx['order_dow']}, hour {ctx['order_hour_of_day']}, "
               f"{ctx['days_since_prior_order']:.0f} days since last order ({'your override' if override else 'from ' + source}).")
    actual = actual_next_basket(uid)
    view = recs.head(top_k).copy()
    view["probability"] = view["probability"].round(4)
    cols = ["product_name", "department", "aisle", "probability", "predicted_reorder", "times_bought", "orders_since_last_bought"]
    if actual is not None:
        view["actually_in_next_order"] = view["product_id"].isin(actual)
        cols.append("actually_in_next_order")
    c = st.columns(3)
    c[0].metric("Products scored", f"{len(recs):,}")
    c[1].metric(f"Predicted reorders (≥ {m['decision_threshold']:.2f})", f"{int(recs['predicted_reorder'].sum())}")
    if actual is not None:
        hits = int(view["actually_in_next_order"].sum())
        c[2].metric(f"Top-{top_k} that were really reordered", f"{hits} / {len(view)}")
    st.dataframe(view[cols], hide_index=True)
    if actual is not None:
        st.caption("`actually_in_next_order` comes from the dataset's labelled next order (Kaggle 'train' users). The model never saw it as a feature.")
    st.download_button("Download all scored products (CSV)", recs.to_csv(index=False), file_name=f"cartiq_user_{uid}.csv")


# ----------------------------------------------------------------------------- navigation
st.sidebar.title("CartIQ 🛒")
page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.caption(f"Data: {'ready' if processed_data_ready() else 'not built - run python main.py'}  \n"
                   f"Model: {'ready' if MODEL_PATH.exists() else 'not trained - run python train.py'}")

{"Overview": page_overview, "Dataset": page_dataset, "EDA / Insights": page_eda, "Product Analysis": page_products,
 "Reorder Analysis": page_reorder, "Time Analysis": page_time, "Market Basket Analysis": page_basket,
 "ML Results": page_ml, "Predict / Recommend": page_predict}[page]()
