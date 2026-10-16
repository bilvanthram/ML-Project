"""
LEGACY / NOTEBOOK ONLY: these functions build a merged DataFrame (order rows x product rows). That is fine on a
sample, but on the full 32M-row prior file it is exactly the memory problem the dashboard avoids. For the full data use
src/data_processing.py + app.py.

eda.py

Data integration + exploratory data analysis for CartIQ.

CartIQ - Group 13

This module:
1. Merges the cleaned Instacart tables into useful combined datasets.
2. Runs univariate, product, basket, reorder, time, and customer analyses.
3. Saves each plot into outputs/figures/.
4. Prints a final EDA Insights summary built from the real numbers.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.paths import FIGURES_DIR
sns.set_style("whitegrid")


def _save_figure(filename):
    """Helper to save the current matplotlib figure into outputs/figures/."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(str(FIGURES_DIR), filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved figure: {path}")


# ---------------------------------------------------------------------------
# DATA INTEGRATION
# ---------------------------------------------------------------------------

def build_merged_data(orders, order_products_prior, products, aisles, departments):
    """
    Builds the main merged dataset used throughout EDA:

    order_products_prior
        -> orders          (adds order-level info: user, day, hour, etc.)
        -> products         (adds product name, aisle_id, department_id)
        -> aisles           (adds aisle name)
        -> departments      (adds department name)

    We use order_products__prior because it is the largest, most complete
    set of historical orders, which makes it best suited for EDA.
    """
    merged_data = order_products_prior.merge(orders, on="order_id", how="left")
    merged_data = merged_data.merge(products, on="product_id", how="left")
    merged_data = merged_data.merge(aisles, on="aisle_id", how="left")
    merged_data = merged_data.merge(departments, on="department_id", how="left")

    print("merged_data created.")
    print(f"Shape: {merged_data.shape}\n")

    return merged_data


# ---------------------------------------------------------------------------
# 1. UNIVARIATE ANALYSIS
# ---------------------------------------------------------------------------

def univariate_analysis(orders, merged_data):
    print("=== Univariate Analysis ===\n")

    # Orders per customer
    orders_per_customer = orders.groupby("user_id")["order_number"].max()
    plt.figure(figsize=(8, 5))
    sns.histplot(orders_per_customer, bins=30, color="steelblue")
    plt.title("Distribution of Orders per Customer")
    plt.xlabel("Number of Orders")
    plt.ylabel("Number of Customers")
    _save_figure("orders_per_customer.png")

    # Products per order
    products_per_order = merged_data.groupby("order_id")["product_id"].count()
    plt.figure(figsize=(8, 5))
    sns.histplot(products_per_order, bins=30, color="darkorange")
    plt.title("Distribution of Products per Order")
    plt.xlabel("Number of Products in Order")
    plt.ylabel("Number of Orders")
    _save_figure("products_per_order.png")

    # Order day of week
    plt.figure(figsize=(8, 5))
    sns.countplot(x="order_dow", data=orders, color="mediumseagreen")
    plt.title("Number of Orders by Day of Week")
    plt.xlabel("Day of Week (0 = Saturday, ... dataset-defined)")
    plt.ylabel("Number of Orders")
    _save_figure("orders_by_day.png")

    # Order hour of day
    plt.figure(figsize=(10, 5))
    sns.countplot(x="order_hour_of_day", data=orders, color="mediumpurple")
    plt.title("Number of Orders by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Orders")
    _save_figure("orders_by_hour.png")

    # Days since prior order
    plt.figure(figsize=(8, 5))
    sns.histplot(orders["days_since_prior_order"].dropna(), bins=30, color="crimson")
    plt.title("Distribution of Days Since Prior Order")
    plt.xlabel("Days Since Prior Order")
    plt.ylabel("Number of Orders")
    _save_figure("days_since_prior_order.png")

    return {
        "orders_per_customer": orders_per_customer,
        "products_per_order": products_per_order,
    }


# ---------------------------------------------------------------------------
# 2. PRODUCT ANALYSIS
# ---------------------------------------------------------------------------

def product_analysis(merged_data):
    print("=== Product Analysis ===\n")

    # Top 15 most ordered products
    top_products = merged_data["product_name"].value_counts().head(15)
    plt.figure(figsize=(9, 6))
    sns.barplot(x=top_products.values, y=top_products.index, color="steelblue")
    plt.title("Top 15 Most Frequently Ordered Products")
    plt.xlabel("Number of Times Ordered")
    plt.ylabel("Product")
    _save_figure("top_products.png")

    # Department-wise order counts
    department_counts = merged_data["department"].value_counts()
    plt.figure(figsize=(9, 6))
    sns.barplot(x=department_counts.values, y=department_counts.index, color="darkorange")
    plt.title("Number of Products Ordered by Department")
    plt.xlabel("Number of Orders")
    plt.ylabel("Department")
    _save_figure("department_distribution.png")

    # Top 15 aisles
    aisle_counts = merged_data["aisle"].value_counts().head(15)
    plt.figure(figsize=(9, 6))
    sns.barplot(x=aisle_counts.values, y=aisle_counts.index, color="mediumseagreen")
    plt.title("Top 15 Most Popular Aisles")
    plt.xlabel("Number of Orders")
    plt.ylabel("Aisle")
    _save_figure("top_aisles.png")

    # Most reordered products (only products with a reasonable order count)
    reorder_by_product = merged_data.groupby("product_name")["reordered"].agg(["mean", "count"])
    reorder_by_product = reorder_by_product[reorder_by_product["count"] >= 50]
    top_reordered_products = reorder_by_product.sort_values("mean", ascending=False).head(15)

    plt.figure(figsize=(9, 6))
    sns.barplot(x=top_reordered_products["mean"], y=top_reordered_products.index, color="crimson")
    plt.title("Top 15 Most Frequently Reordered Products (min. 50 orders)")
    plt.xlabel("Reorder Rate")
    plt.ylabel("Product")
    _save_figure("top_reordered_products.png")

    return {
        "top_products": top_products,
        "department_counts": department_counts,
        "aisle_counts": aisle_counts,
        "top_reordered_products": top_reordered_products,
    }


# ---------------------------------------------------------------------------
# 3. BASKET ANALYSIS
# ---------------------------------------------------------------------------

def basket_analysis(merged_data):
    print("=== Basket Analysis ===\n")

    basket_sizes = merged_data.groupby("order_id")["product_id"].count()

    basket_stats = {
        "average_basket_size": basket_sizes.mean(),
        "min_basket_size": basket_sizes.min(),
        "max_basket_size": basket_sizes.max(),
        "median_basket_size": basket_sizes.median(),
    }

    plt.figure(figsize=(8, 5))
    sns.histplot(basket_sizes, bins=40, color="teal")
    plt.title("Basket Size Distribution")
    plt.xlabel("Number of Products in Order")
    plt.ylabel("Number of Orders")
    _save_figure("basket_size_distribution.png")

    print(f"Average basket size: {basket_stats['average_basket_size']:.2f}")
    print(f"Minimum basket size: {basket_stats['min_basket_size']}")
    print(f"Maximum basket size: {basket_stats['max_basket_size']}")
    print(f"Median basket size: {basket_stats['median_basket_size']}\n")

    return basket_stats, basket_sizes


# ---------------------------------------------------------------------------
# 4. REORDER ANALYSIS
# ---------------------------------------------------------------------------

def reorder_analysis(merged_data):
    print("=== Reorder Analysis ===\n")
    print("Note: 'reordered' is the existing target/label column for a future")
    print("machine learning phase. No model is trained on it in this project.\n")

    # Overall reordered vs not reordered
    reorder_counts = merged_data["reordered"].value_counts()
    overall_reorder_rate = merged_data["reordered"].mean() * 100

    plt.figure(figsize=(6, 5))
    sns.countplot(x="reordered", data=merged_data, color="slateblue")
    plt.title("Reordered vs Not Reordered (Overall)")
    plt.xlabel("Reordered (0 = No, 1 = Yes)")
    plt.ylabel("Count")
    _save_figure("reorder_distribution.png")

    # Reorder rate by department
    reorder_by_department = merged_data.groupby("department")["reordered"].mean().sort_values(ascending=False) * 100

    plt.figure(figsize=(9, 6))
    sns.barplot(x=reorder_by_department.values, y=reorder_by_department.index, color="darkorange")
    plt.title("Reorder Rate by Department")
    plt.xlabel("Reorder Rate (%)")
    plt.ylabel("Department")
    _save_figure("reorder_by_department.png")

    # Reorder rate by aisle (top 15)
    reorder_by_aisle = merged_data.groupby("aisle")["reordered"].mean().sort_values(ascending=False) * 100
    reorder_by_aisle_top = reorder_by_aisle.head(15)

    plt.figure(figsize=(9, 6))
    sns.barplot(x=reorder_by_aisle_top.values, y=reorder_by_aisle_top.index, color="mediumseagreen")
    plt.title("Top 15 Aisles by Reorder Rate")
    plt.xlabel("Reorder Rate (%)")
    plt.ylabel("Aisle")
    _save_figure("reorder_by_aisle.png")

    return {
        "reorder_counts": reorder_counts,
        "overall_reorder_rate": overall_reorder_rate,
        "reorder_by_department": reorder_by_department,
    }


# ---------------------------------------------------------------------------
# 5. TIME-BASED ANALYSIS
# ---------------------------------------------------------------------------

def time_based_analysis(orders):
    print("=== Time-Based Analysis ===\n")

    busiest_day = orders["order_dow"].value_counts().idxmax()
    busiest_hour = orders["order_hour_of_day"].value_counts().idxmax()

    # Orders by day and hour together (heatmap)
    day_hour_counts = orders.groupby(["order_dow", "order_hour_of_day"]).size().unstack()

    plt.figure(figsize=(12, 6))
    sns.heatmap(day_hour_counts, cmap="YlOrRd")
    plt.title("Orders by Day of Week and Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Day of Week")
    _save_figure("orders_day_hour_heatmap.png")

    return {
        "busiest_day": busiest_day,
        "busiest_hour": busiest_hour,
    }


# ---------------------------------------------------------------------------
# 6. CUSTOMER / ORDER BEHAVIOR
# ---------------------------------------------------------------------------

def customer_behavior_analysis(orders, merged_data):
    print("=== Customer / Order Behavior Analysis ===\n")

    avg_products_per_order = merged_data.groupby("order_id")["product_id"].count().mean()

    # Reorder rate by order_number (does reorder rate change as customers order more?)
    reorder_by_order_number = merged_data.groupby("order_number")["reordered"].mean() * 100
    reorder_by_order_number = reorder_by_order_number.head(30)  # keep it readable

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=reorder_by_order_number.index, y=reorder_by_order_number.values, color="crimson")
    plt.title("Reorder Rate vs Order Number (first 30 orders)")
    plt.xlabel("Order Number")
    plt.ylabel("Reorder Rate (%)")
    _save_figure("reorder_rate_vs_order_number.png")

    print(f"Average products per order: {avg_products_per_order:.2f}\n")

    return {
        "avg_products_per_order": avg_products_per_order,
        "reorder_by_order_number": reorder_by_order_number,
    }


# ---------------------------------------------------------------------------
# CORRELATION ANALYSIS
# ---------------------------------------------------------------------------

def correlation_analysis(orders, merged_data):
    """
    Correlation is only calculated between meaningful numerical variables
    (not ID columns, since correlating IDs is meaningless).
    """
    print("=== Correlation Analysis ===\n")

    numeric_columns = [
        "order_number",
        "order_dow",
        "order_hour_of_day",
        "days_since_prior_order",
    ]
    numeric_columns = [col for col in numeric_columns if col in orders.columns]

    corr_matrix = orders[numeric_columns].corr()

    plt.figure(figsize=(7, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Order-Level Numerical Variables)")
    _save_figure("correlation_heatmap.png")

    print("Reminder: correlation does NOT imply causation. These values only")
    print("describe linear relationships between variables, not cause and effect.\n")

    return corr_matrix


# ---------------------------------------------------------------------------
# EDA INSIGHTS
# ---------------------------------------------------------------------------

def print_eda_insights(product_results, basket_stats, reorder_results, time_results):
    """
    Prints a final summary of insights, using only real numbers computed
    earlier in the pipeline (nothing is invented).
    """
    top_product = product_results["top_products"].index[0]
    top_department = product_results["department_counts"].index[0]
    top_reorder_department = reorder_results["reorder_by_department"].index[0]
    top_reorder_department_rate = reorder_results["reorder_by_department"].iloc[0]

    print("=" * 60)
    print("EDA INSIGHTS")
    print("=" * 60)
    print(f"1. The most frequently ordered product is '{top_product}'.")
    print(f"2. The department with the highest number of orders is '{top_department}'.")
    print(f"3. The average basket size is {basket_stats['average_basket_size']:.2f} products.")
    print(f"4. Approximately {reorder_results['overall_reorder_rate']:.2f}% of product orders are reorders.")
    print(f"5. The highest reorder rate occurs in the '{top_reorder_department}' department "
          f"({top_reorder_department_rate:.2f}%).")
    print(f"6. The busiest ordering hour is {time_results['busiest_hour']}:00.")
    print(f"7. The busiest ordering day is day code {time_results['busiest_day']} "
          f"(0/1 = weekend in this dataset's encoding).")
    print("=" * 60)
    print("\nNote: correlation and frequency findings above describe patterns in")
    print("the data only. They do not imply causation.\n")
