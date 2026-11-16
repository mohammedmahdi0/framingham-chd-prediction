"""
common.py - shared code used by every experiment script.

It does four things:
  1. loads framingham.csv and makes ONE fixed, stratified train / test split
  2. defines the cross-validation scheme used for every comparison
  3. defines the preprocessing building blocks (imputers, transformations, scalers)
  4. defines small helpers for printing / saving tables and figures

RULE: the test set (X_test, y_test) is only used by the final-evaluation scripts
(06h and 07). All experiments use cross-validation inside the training set.
"""
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import (train_test_split, StratifiedKFold, RepeatedStratifiedKFold,
                                     cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer,
                                   QuantileTransformer, FunctionTransformer)
from sklearn.linear_model import LogisticRegression

sklearn.set_config(transform_output="pandas")     # keep column names through the pipelines
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.width", 220, "display.max_columns", 60, "display.max_rows", 200,
              "display.float_format", "{:.4f}".format)

# --------------------------------------------------------------------------- paths
BASE = Path(__file__).resolve().parent
FIG_DIR, RES_DIR, MODEL_DIR = BASE / "figures", BASE / "results", BASE / "model"
for d in (FIG_DIR, RES_DIR, MODEL_DIR):
    d.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- data
SEED = 42
TARGET = "TenYearCHD"
BINARY = ["male", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"]
ORDINAL = ["education"]
CONT = ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]

df = pd.read_csv(BASE / "framingham.csv")
X, y = df.drop(columns=TARGET), df[TARGET]
FEATURES = list(X.columns)

# One fixed split, made before anything else. 80% train / 20% test, stratified.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)

BASELINE_TRAIN = 1 - y_train.mean()     # accuracy of always answering "no CHD"
BASELINE_TEST = 1 - y_test.mean()

# Continuous columns whose skewness on the training set is above 0.5 (see 03_distribution.py)
SKEWED = ["cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]

# Feature subsets found in step 5 (05a and 05c). They were chosen on the training set only.
FEATURE_SETS = {
    "all15": FEATURES,
    "lasso10": ["male", "age", "cigsPerDay", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
                "totChol", "sysBP", "glucose"],
    "aic8": ["male", "age", "cigsPerDay", "BPMeds", "prevalentStroke", "totChol", "sysBP", "glucose"],
    "bic5": ["male", "age", "cigsPerDay", "sysBP", "glucose"],
}

# --------------------------------------------------------------------------- cross-validation
CV5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
RCV = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)   # 15 fits per experiment
# A second, independent set of folds. Used in step 6 to score models AFTER tuning, because the folds
# used by the hyper-parameter search itself give a slightly optimistic score.
FRESH_CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED + 1)


def cv_eval(model, X=None, y=None, cv=RCV, n_jobs=-1):
    """Cross-validated accuracy and AUC on the training set."""
    X = X_train if X is None else X
    y = y_train if y is None else y
    s = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "roc_auc"], n_jobs=n_jobs)
    return {"CV accuracy": s["test_accuracy"].mean(), "CV acc std": s["test_accuracy"].std(),
            "CV AUC": s["test_roc_auc"].mean()}


def LR(**kw):
    # random_state makes the 'saga' solver (used for L1 / Elastic Net) give identical results on every run
    return LogisticRegression(max_iter=5000, random_state=SEED, **kw)


# --------------------------------------------------------------------------- custom transformers
class Clipper(BaseEstimator, TransformerMixin):
    """Winsorizing: clip each column to the [low, high] percentiles learned on the training data."""

    def __init__(self, low=1, high=99):
        self.low, self.high = low, high

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lo_, self.hi_ = X.quantile(self.low / 100), X.quantile(self.high / 100)
        return self

    def transform(self, X):
        return pd.DataFrame(X).clip(self.lo_, self.hi_, axis=1)


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Keep one of the named feature subsets in FEATURE_SETS."""

    def __init__(self, feature_set="all15"):
        self.feature_set = feature_set

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X)[FEATURE_SETS[self.feature_set]]


class StepwiseSelector(BaseEstimator, TransformerMixin):
    """Forward / backward stepwise feature selection for logistic regression using AIC or BIC."""

    def __init__(self, criterion="aic", direction="forward"):
        self.criterion, self.direction = criterion, direction

    def _score(self, X, y, cols):
        Xc = sm.add_constant(X[cols], has_constant="add") if cols else pd.DataFrame({"const": np.ones(len(y))})
        try:
            return getattr(sm.Logit(y, Xc).fit(disp=0, maxiter=200), self.criterion)
        except Exception:
            return np.inf

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y)
        cols = list(X.columns)
        current = [] if self.direction == "forward" else cols[:]
        best = self._score(X, y, current)
        self.history_ = [(tuple(current), best)]
        while True:
            if self.direction == "forward":
                cand = [(self._score(X, y, current + [c]), current + [c]) for c in cols if c not in current]
            else:
                cand = [(self._score(X, y, [k for k in current if k != c]), [k for k in current if k != c])
                        for c in current]
            if not cand:
                break
            score, new = min(cand, key=lambda t: t[0])
            if score < best - 1e-9:
                best, current = score, new
                self.history_.append((tuple(current), best))
            else:
                break
        self.selected_, self.best_score_ = current, best
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.selected_]


# --------------------------------------------------------------------------- preprocessing blocks
def _on_cols(transformer, cols):
    return ColumnTransformer([("t", transformer, cols)], remainder="passthrough", verbose_feature_names_out=False)


def _log():
    return FunctionTransformer(np.log1p, feature_names_out="one-to-one")


IMPUTERS = {
    "Mean": SimpleImputer(strategy="mean"),
    "Median": SimpleImputer(strategy="median"),
    "Most frequent": SimpleImputer(strategy="most_frequent"),
    "Median (numeric) + mode (categorical)": ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), CONT),
         ("cat", SimpleImputer(strategy="most_frequent"), BINARY + ORDINAL)], verbose_feature_names_out=False),
    "Median + missing-indicator columns": SimpleImputer(strategy="median", add_indicator=True),
    "KNN imputer (k=5)": Pipeline([("s", StandardScaler()), ("i", KNNImputer(n_neighbors=5))]),
    "Iterative imputer (MICE)": IterativeImputer(max_iter=10, random_state=SEED),
}

TRANSFORMS = {
    "None": "passthrough",
    "log(1+x) on skewed columns": _on_cols(_log(), SKEWED),
    "Yeo-Johnson power transform": _on_cols(PowerTransformer(method="yeo-johnson"), CONT),
    "Quantile transform to normal": _on_cols(
        QuantileTransformer(output_distribution="normal", n_quantiles=200, random_state=SEED), CONT),
    "Winsorize (clip to 1st-99th percentile)": _on_cols(Clipper(1, 99), CONT),
    "Winsorize + log(1+x)": Pipeline([("w", _on_cols(Clipper(1, 99), CONT)), ("l", _on_cols(_log(), SKEWED))]),
}

SCALERS = {"None": "passthrough", "StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(),
           "RobustScaler": RobustScaler()}


def make_pre(imputer="Median", transform="None", scaler="StandardScaler"):
    """Preprocessing pipeline: impute -> reshape distributions -> scale."""
    t, s = TRANSFORMS[transform], SCALERS[scaler]
    return Pipeline([("impute", clone(IMPUTERS[imputer])),
                     ("transform", t if isinstance(t, str) else clone(t)),
                     ("scale", s if isinstance(s, str) else clone(s))])


def PRE():
    """The preprocessing chosen in steps 2-4: median imputation, no transformation, StandardScaler."""
    return make_pre(imputer="Median", transform="None", scaler="StandardScaler")


# --------------------------------------------------------------------------- output helpers
def header(title):
    line = "=" * 100
    print(f"\n{line}\n{title}\n{line}")
    print(f"train = {len(X_train)} rows | test = {len(X_test)} rows | "
          f"baseline accuracy (always 'no CHD'): train {BASELINE_TRAIN:.4f}, test {BASELINE_TEST:.4f}\n")


def show(table, title=None):
    if title:
        print(f"\n--- {title}")
    print(table.to_string())


def save_table(table, name, index=False):
    path = RES_DIR / f"{name}.csv"
    table.to_csv(path, index=index)
    print(f"[saved] results/{name}.csv")


def save_json(obj, name):
    import json
    with open(RES_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"[saved] results/{name}.json")


def load_json(name):
    import json
    path = RES_DIR / f"{name}.json"
    if not path.exists():
        sys.exit(f"results/{name}.json not found - run the earlier script that creates it first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def record(rows, method, family, pipe, n_feat, n_jobs=-1):
    """Cross-validate one dimensionality-reduction pipeline and append the result to `rows`."""
    r = cv_eval(pipe, n_jobs=n_jobs)
    rows.append({"Method": method, "Family": family, "Features used": str(n_feat), **r})
    print(f"  {method:42s} features={str(n_feat):>6s}  CV acc={r['CV accuracy']:.4f} +/- {r['CV acc std']:.4f}"
          f"  AUC={r['CV AUC']:.4f}")


def savefig(name):
    """Save the current figure to figures/. The window is shown unless NO_SHOW=1 is set."""
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    print(f"[saved] figures/{name}.png")
    if os.environ.get("NO_SHOW") == "1":
        plt.close()
    else:
        plt.show()
