"""
STEP 5d - OTHER FEATURE-SELECTION METHODS
  - RFE with cross-validation (wrapper method)
  - ANOVA F-test and mutual information (filter methods; the number of features k is tuned by inner CV)
  - Random-Forest importance (embedded method)

Run:  python 05d_other_feature_selection.py
"""
from common import *
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFECV, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from functools import partial

# mutual information is estimated with random noise; fixing the seed makes it repeatable
mutual_info = partial(mutual_info_classif, random_state=SEED)


def kbest(score_func):
    inner = Pipeline([("kb", SelectKBest(score_func)), ("clf", LR())])
    return GridSearchCV(inner, {"kb__k": list(range(2, 16))}, scoring="accuracy", cv=CV5, n_jobs=1)


def rf_selector():
    return SelectFromModel(RandomForestClassifier(300, min_samples_leaf=5, random_state=SEED, n_jobs=1),
                           threshold="median")


def main():
    header("STEP 5d - OTHER FEATURE-SELECTION METHODS")
    Xp = PRE().fit_transform(X_train)
    rows, selected = [], {}

    # RFE
    rfe = RFECV(LR(), cv=CV5, scoring="accuracy").fit(Xp, y_train)
    selected["RFE (RFECV)"] = list(Xp.columns[rfe.support_])
    record(rows, "RFE (RFECV)", "Feature selection",
           Pipeline([("pre", PRE()), ("sel", RFECV(LR(), cv=CV5, scoring="accuracy")), ("clf", LR())]),
           int(rfe.n_features_))

    # filters
    for name, func in {"ANOVA F-test (SelectKBest)": f_classif, "Mutual information (SelectKBest)": mutual_info}.items():
        g = kbest(func).fit(Xp, y_train)
        selected[name] = list(Xp.columns[g.best_estimator_.named_steps["kb"].get_support()])
        record(rows, name, "Feature selection", Pipeline([("pre", PRE()), ("g", kbest(func))]), len(selected[name]))

    # Random-Forest importance
    s = rf_selector().fit(Xp, y_train)
    selected["Random-Forest importance"] = list(Xp.columns[s.get_support()])
    record(rows, "Random-Forest importance", "Feature selection",
           Pipeline([("pre", PRE()), ("sel", rf_selector()), ("clf", LR())]), len(selected["Random-Forest importance"]))

    res = pd.DataFrame(rows)
    save_table(res, "step5d_other_selection")
    save_json(selected, "step5d_selected")
    show(res, "Results")

    scores = pd.DataFrame({"ANOVA F": f_classif(Xp, y_train)[0],
                           "mutual information": mutual_info_classif(Xp, y_train, random_state=SEED),
                           "RF importance": s.estimator_.feature_importances_,
                           "RFE rank (1 = kept)": rfe.ranking_}, index=Xp.columns).sort_values("ANOVA F", ascending=False)
    show(scores, "Feature scores given by each method")
    save_table(scores, "step5d_feature_scores", index=True)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for a, c in zip(ax, ["ANOVA F", "mutual information", "RF importance"]):
        d = scores[c].sort_values()
        a.barh(d.index, d.values, color="#55A868")
        a.set_title(c)
    fig.suptitle("Feature importance according to three different methods", y=1.02)
    savefig("14_feature_scores")
    for k, v in selected.items():
        print(f"{k}: kept {len(v)} -> {v}")


if __name__ == "__main__":
    main()
