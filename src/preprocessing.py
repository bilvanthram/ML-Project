"""
preprocessing.py - one place that defines how features are prepared for each model family.

Every model is trained as a single sklearn Pipeline(preprocessor -> estimator) and the whole
Pipeline is what gets saved, so the Streamlit app applies EXACTLY the training-time
imputation/scaling/encoding (no training-serving skew).

  "linear" (Logistic Regression): median impute + StandardScaler for numbers, one-hot for
           department/aisle. Scaling matters here: the L2 penalty and the optimiser are sensitive
           to feature magnitude.
  "hgb"    (HistGradientBoosting): numbers untouched (it handles missing values natively),
           department/aisle ordinal-coded and declared categorical.
  "tree"   (other tree models): median impute, ordinal-coded categoricals. No scaling - tree
           splits are invariant to monotonic rescaling.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor(kind):
    if kind == "linear":
        num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        cat = OneHotEncoder(handle_unknown="ignore")
    elif kind == "hgb":
        num = "passthrough"
        cat = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                             encoded_missing_value=np.nan)
    elif kind == "tree":
        num = SimpleImputer(strategy="median")
        cat = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        raise ValueError(f"unknown preprocessor kind: {kind}")
    return ColumnTransformer([("num", num, NUMERIC_FEATURES), ("cat", cat, CATEGORICAL_FEATURES)])


def transformed_feature_names(pipeline):
    """Readable names of the columns the estimator actually sees."""
    pre = pipeline.named_steps["preprocessor"]
    names = list(NUMERIC_FEATURES)
    cat = pre.named_transformers_["cat"]
    if hasattr(cat, "get_feature_names_out"):
        names += list(cat.get_feature_names_out(CATEGORICAL_FEATURES))
    return names
