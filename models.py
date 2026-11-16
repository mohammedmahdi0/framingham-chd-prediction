"""
models.py - the 17 model families tried in step 6, with their hyper-parameter search spaces.

Every model is wrapped in the same pipeline:
    preprocessing (median imputation + StandardScaler)  ->  feature subset  ->  classifier
The feature subset (all15 / lasso10 / aic8 / bic5, from step 5) is tuned like any other hyper-parameter.
"""
from common import *

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier,
                              VotingClassifier, StackingClassifier)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

LOG = lambda a, b, n: [float(v) for v in np.logspace(a, b, n)]

# name -> (estimator with default settings, search space)
ZOO = {
    "Logistic Regression": (
        LogisticRegression(max_iter=5000, solver="saga", random_state=SEED),
        {"C": LOG(-3, 2, 20), "l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0]}),
    "LDA": (
        LinearDiscriminantAnalysis(solver="lsqr"),
        {"shrinkage": [None, "auto", 0.01, 0.05, 0.1, 0.3, 0.6, 0.9]}),
    "QDA": (
        QuadraticDiscriminantAnalysis(),
        {"reg_param": [0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0]}),
    "Naive Bayes": (
        GaussianNB(),
        {"var_smoothing": LOG(-9, 0, 10)}),
    "KNN": (
        KNeighborsClassifier(),
        {"n_neighbors": [5, 9, 15, 21, 31, 41, 51, 75, 101], "weights": ["uniform", "distance"], "p": [1, 2]}),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=SEED),
        {"max_depth": [1, 2, 3, 4, 5, 6, 8, None], "min_samples_leaf": [1, 5, 10, 20, 50, 100],
         "criterion": ["gini", "entropy"]}),
    "Random Forest": (
        RandomForestClassifier(random_state=SEED, n_jobs=1),
        {"n_estimators": [200, 500], "max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 3, 5, 10, 20],
         "max_features": ["sqrt", 0.3, 0.5]}),
    "Extra Trees": (
        ExtraTreesClassifier(random_state=SEED, n_jobs=1),
        {"n_estimators": [200, 500], "max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 3, 5, 10, 20],
         "max_features": ["sqrt", 0.3, 0.5]}),
    "AdaBoost": (
        AdaBoostClassifier(random_state=SEED),
        {"n_estimators": [50, 100, 200, 400], "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0]}),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=SEED),
        {"n_estimators": [50, 100, 200, 300], "learning_rate": [0.01, 0.03, 0.05, 0.1], "max_depth": [1, 2, 3],
         "subsample": [0.7, 0.85, 1.0], "min_samples_leaf": [1, 10, 30]}),
    "Hist Gradient Boosting": (
        HistGradientBoostingClassifier(random_state=SEED),
        {"max_iter": [50, 100, 200], "learning_rate": [0.01, 0.03, 0.05, 0.1], "max_depth": [2, 3, 4, None],
         "l2_regularization": [0.0, 1.0, 10.0], "min_samples_leaf": [10, 20, 50]}),
    "XGBoost": (
        XGBClassifier(random_state=SEED, n_jobs=1, eval_metric="logloss"),
        {"n_estimators": [50, 100, 200, 300], "max_depth": [1, 2, 3, 4], "learning_rate": [0.01, 0.03, 0.05, 0.1],
         "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.6, 0.8, 1.0], "reg_lambda": [1, 5, 10],
         "min_child_weight": [1, 5, 10]}),
    "LightGBM": (
        LGBMClassifier(random_state=SEED, n_jobs=1, verbose=-1, subsample_freq=1),
        {"n_estimators": [50, 100, 200, 300], "num_leaves": [3, 5, 7, 15], "learning_rate": [0.01, 0.03, 0.05, 0.1],
         "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.6, 0.8, 1.0], "min_child_samples": [10, 20, 50],
         "reg_lambda": [0, 1, 10]}),
    "CatBoost": (
        CatBoostClassifier(random_state=SEED, thread_count=1, verbose=0, allow_writing_files=False),
        {"iterations": [200, 400], "depth": [2, 3, 4, 6], "learning_rate": [0.01, 0.03, 0.05, 0.1],
         "l2_leaf_reg": [1, 3, 10]}),
    "SVM (linear)": (
        SVC(kernel="linear", probability=True, random_state=SEED),
        {"C": [0.001, 0.01, 0.1, 1.0]}),
    "SVM (RBF)": (
        SVC(kernel="rbf", probability=True, random_state=SEED),
        {"C": [0.1, 0.3, 1.0, 3.0, 10.0], "gamma": ["scale", 0.001, 0.01, 0.03, 0.1]}),
    "Neural Network (MLP)": (
        MLPClassifier(max_iter=1000, random_state=SEED),
        {"hidden_layer_sizes": [(8,), (16,), (32,), (16, 8), (32, 16)], "alpha": [1e-4, 1e-2, 0.1, 1.0, 3.0],
         "learning_rate_init": [0.001, 0.01], "early_stopping": [True, False]}),
}

FEATURE_SET_CHOICES = ["all15", "lasso10", "aic8", "bic5"]


def make_pipeline(name, feature_set="all15", **clf_params):
    """preprocessing -> feature subset -> classifier"""
    est = clone(ZOO[name][0]).set_params(**clf_params)
    return Pipeline([("pre", PRE()), ("sel", ColumnSelector(feature_set)), ("clf", est)])


def search_space(name):
    space = {f"clf__{k}": v for k, v in ZOO[name][1].items()}
    space["sel__feature_set"] = FEATURE_SET_CHOICES
    return space


def build_tuned(name, best=None):
    """Rebuild a model with the hyper-parameters found by 06b_hyperparameter_tuning.py."""
    best = best or load_json("step6b_best_params")
    p = dict(best[name]["params"])
    fs = p.pop("feature_set")
    if "hidden_layer_sizes" in p:
        p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
    return make_pipeline(name, fs, **p)


# The seven base models of the stacking example in "Lab 2 - NLP / Stacking"
LECTURE_MEMBERS = ["KNN", "SVM (RBF)", "Decision Tree", "Random Forest", "XGBoost", "Naive Bayes",
                   "Neural Network (MLP)"]


def lecture_stacking():
    """The stacked model exactly as built in Lab 2: seven classifiers with the lab's settings,
    combined by an XGBoost meta-model. Only random_state is added, so that results are repeatable."""
    estimators = [
        ("knn", KNeighborsClassifier(3)),
        ("svm_rbf", SVC(gamma=2, C=1)),
        ("dt", DecisionTreeClassifier(max_depth=5, random_state=SEED)),
        ("rf", RandomForestClassifier(n_estimators=10, random_state=SEED)),
        ("XGB_C", XGBClassifier(random_state=SEED, n_jobs=1)),
        ("NB_C", GaussianNB(priors=None, var_smoothing=1e-09)),
        ("mlp", MLPClassifier(alpha=1, max_iter=1000, random_state=SEED)),
    ]
    stack = StackingClassifier(estimators=estimators, final_estimator=XGBClassifier(random_state=SEED, n_jobs=1))
    return Pipeline([("pre", PRE()), ("stack", stack)])


def build_ensemble(kind, members, best=None):
    """Ensembles of models. Each tuned member keeps its own pipeline.
       voting        soft voting (average of the predicted probabilities)
       hard_voting   majority vote of the predicted classes
       stacking      stacking, logistic regression as the meta-model
       stacking_xgb  stacking, XGBoost as the meta-model (as in Lab 2)
       lecture       the Lab 2 stacked model with the lab's own settings (`members` is ignored)"""
    if kind == "lecture":
        return lecture_stacking()
    best = best or load_json("step6b_best_params")
    estimators = [(m, build_tuned(m, best)) for m in members]
    if kind == "voting":
        return VotingClassifier(estimators, voting="soft", n_jobs=1)
    if kind == "hard_voting":
        return VotingClassifier(estimators, voting="hard", n_jobs=1)
    if kind == "stacking":
        return StackingClassifier(estimators, final_estimator=LogisticRegression(max_iter=5000, random_state=SEED), cv=5,
                                  stack_method="predict_proba", n_jobs=1)
    if kind == "stacking_xgb":
        return StackingClassifier(estimators, final_estimator=XGBClassifier(random_state=SEED, n_jobs=1), cv=5,
                                  stack_method="predict_proba", n_jobs=1)
    raise ValueError(kind)
