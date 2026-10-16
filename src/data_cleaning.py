"""
data_cleaning.py

Basic, classroom-level cleaning for the Instacart dataset.

CartIQ - Group 13
"""

import pandas as pd


def explore_dataframe(df, name):
    """
    Prints a simple understanding summary for one dataframe:
    shape, columns, dtypes, missing values, and duplicate rows.

    This is meant to be readable in a classroom setting, not exhaustive.
    """
    print(f"--- {name} ---")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print("-" * 60, "\n")


def clean_orders(orders):
    """
    Cleans the orders dataframe.

    - 'days_since_prior_order' is missing for a customer's very first order
      (there is no "prior" order to compare to). This is expected, not an
      error, so instead of dropping these rows we keep them and simply note
      that a missing value here means "first order for this customer".
    - Removes exact duplicate rows, if any.
    """
    orders_clean = orders.copy()

    duplicates_found = orders_clean.duplicated().sum()
    if duplicates_found > 0:
        orders_clean = orders_clean.drop_duplicates()
        print(f"Removed {duplicates_found} duplicate rows from orders.")

    # order_dow (day of week) and order_hour_of_day should be small integers.
    # We just make sure they are integer type for consistent plotting later.
    if "order_dow" in orders_clean.columns:
        orders_clean["order_dow"] = orders_clean["order_dow"].astype(int)
    if "order_hour_of_day" in orders_clean.columns:
        orders_clean["order_hour_of_day"] = orders_clean["order_hour_of_day"].astype(int)

    print("orders cleaned.")
    print(f"Missing 'days_since_prior_order' values (expected for first-time orders): "
          f"{orders_clean['days_since_prior_order'].isnull().sum() if 'days_since_prior_order' in orders_clean.columns else 'N/A'}\n")

    return orders_clean


def clean_products(products):
    """
    Cleans the products dataframe.

    - Removes duplicate rows if any.
    - Strips extra whitespace from product names, since inconsistent
      spacing can cause the same product to look "different" in groupby
      operations.
    """
    products_clean = products.copy()

    duplicates_found = products_clean.duplicated().sum()
    if duplicates_found > 0:
        products_clean = products_clean.drop_duplicates()
        print(f"Removed {duplicates_found} duplicate rows from products.")

    if "product_name" in products_clean.columns:
        products_clean["product_name"] = products_clean["product_name"].str.strip()

    print("products cleaned.\n")
    return products_clean


def clean_order_products(order_products):
    """
    Cleans an order_products dataframe (prior or train).

    - Removes duplicate rows if any.
    - Ensures 'reordered' is an integer (0 or 1) for correct aggregation.
    """
    order_products_clean = order_products.copy()

    duplicates_found = order_products_clean.duplicated().sum()
    if duplicates_found > 0:
        order_products_clean = order_products_clean.drop_duplicates()
        print(f"Removed {duplicates_found} duplicate rows from order_products.")

    if "reordered" in order_products_clean.columns:
        order_products_clean["reordered"] = order_products_clean["reordered"].astype(int)

    print("order_products cleaned.\n")
    return order_products_clean


def clean_aisles_departments(aisles, departments):
    """
    Cleans the small lookup tables: aisles and departments.
    These are small dataframes, so cleaning is minimal.
    """
    aisles_clean = aisles.drop_duplicates().copy()
    departments_clean = departments.drop_duplicates().copy()

    if "aisle" in aisles_clean.columns:
        aisles_clean["aisle"] = aisles_clean["aisle"].str.strip()
    if "department" in departments_clean.columns:
        departments_clean["department"] = departments_clean["department"].str.strip()

    print("aisles and departments cleaned.\n")
    return aisles_clean, departments_clean
