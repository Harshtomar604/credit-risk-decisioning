import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SEED = 42

DELAY_COLS = ["max_delay", "recent_delay"]
TREND_COLS = ["delay_trend"]
UTIL_COLS = ["util_recent", "util_avg", "util_max"]
CAT_COLS = ["education"]
OTHER_NUM_COLS = ["limit_bal", "age", "months_late", "pay_ratio_recent", "bill_trend", "log_pay_amt1"]

FEATURES = DELAY_COLS + TREND_COLS + UTIL_COLS + CAT_COLS + OTHER_NUM_COLS


def cap_delay(X):
    return np.clip(X, 0, 3)


def cap_trend(X):
    return np.clip(X, -3, 3)


class UpperClipper(BaseEstimator, TransformerMixin):
    def __init__(self, quantile=0.99):
        self.quantile = quantile

    def fit(self, X, y=None):
        self.upper_ = np.quantile(np.asarray(X, dtype=float), self.quantile, axis=0)
        return self

    def transform(self, X):
        return np.minimum(np.asarray(X, dtype=float), self.upper_)


def make_logreg():
    prep = ColumnTransformer([
        ("delay", Pipeline([("cap", FunctionTransformer(cap_delay)), ("scale", StandardScaler())]), DELAY_COLS),
        ("trend", Pipeline([("cap", FunctionTransformer(cap_trend)), ("scale", StandardScaler())]), TREND_COLS),
        ("util", Pipeline([("clip", UpperClipper(0.99)), ("scale", StandardScaler())]), UTIL_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ("num", StandardScaler(), OTHER_NUM_COLS),
    ])
    return Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=1000))])


def make_gbm():
    return HistGradientBoostingClassifier(random_state=SEED)


def make_final_gbm():
    from sklearn.calibration import CalibratedClassifierCV
    return CalibratedClassifierCV(make_gbm(), method="isotonic", cv=5)
