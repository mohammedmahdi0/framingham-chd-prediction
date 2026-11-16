"""
STEP 6f - FEATURE-ENGINEERING EXPERIMENT
Can new features built from the existing ones raise accuracy?

  bp     : pulse pressure (sysBP - diaBP) and mean arterial pressure
  log    : log of glucose, sysBP, totChol, BMI
  flags  : clinical cut-offs (obese, diabetic-range glucose, stage-2 hypertension, heavy smoker, ...)
  inter  : interactions (age x sysBP, age x smoking, male x smoking, ...)
  poly   : squared terms of age, sysBP, glucose, cigsPerDay
  all    : everything above

Pipeline: median imputation -> add features -> StandardScaler -> model.
Three probe models: plain logistic regression, Lasso logistic regression (C tuned by inner CV, so it
can drop useless new features), and gradient boosting.

Run:  python 06f_feature_engineering.py      (about 2 minutes)
"""
from common import *
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier


class FeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, groups=()):
        self.groups = groups

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        g = self.groups
        if "bp" in g:
            X["pulse_pressure"] = X.sysBP - X.diaBP
            X["mean_arterial_pressure"] = X.diaBP + (X.sysBP - X.diaBP) / 3
        if "log" in g:
            for c in ["glucose", "sysBP", "totChol", "BMI"]:
                X[f"log_{c}"] = np.log(X[c])
        if "flags" in g:
            X["obese"] = (X.BMI >= 30).astype(int)
            X["glucose_ge_126"] = (X.glucose >= 126).astype(int)
            X["sysBP_ge_160"] = (X.sysBP >= 160).astype(int)
            X["heavy_smoker"] = (X.cigsPerDay >= 20).astype(int)
            X["chol_ge_240"] = (X.totChol >= 240).astype(int)
            X["age_ge_60"] = (X.age >= 60).astype(int)
        if "inter" in g:
            X["age_x_sysBP"] = X.age * X.sysBP
            X["age_x_cigs"] = X.age * X.cigsPerDay
            X["male_x_cigs"] = X.male * X.cigsPerDay
            X["male_x_age"] = X.male * X.age
            X["age_x_glucose"] = X.age * X.glucose
            X["hyp_x_sysBP"] = X.prevalentHyp * X.sysBP
        if "poly" in g:
            for c in ["age", "sysBP", "glucose", "cigsPerDay"]:
                X[f"{c}_squared"] = X[c] ** 2
        return X


def pipe(groups, clf):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("add", FeatureAdder(groups)),
                     ("scale", StandardScaler()), ("clf", clf)])


def main():
    header("STEP 6f - FEATURE ENGINEERING")
    variants = {"Original 15 features": (), "+ bp": ("bp",), "+ log": ("log",), "+ flags": ("flags",),
                "+ inter": ("inter",), "+ poly": ("poly",), "+ all": ("bp", "log", "flags", "inter", "poly")}
    probes = {
        "Logistic Regression": lambda: LR(),
        "Lasso LR (C tuned)": lambda: GridSearchCV(LR(solver="saga", l1_ratio=1.0), {"C": np.logspace(-2.5, 0.5, 8)},
                                                   scoring="accuracy", cv=CV5, n_jobs=1),
        "Hist Gradient Boosting": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=100,
                                                                         random_state=SEED),
    }
    rows = []
    for vname, groups in variants.items():
        n = pipe(groups, LR())[:-2].fit_transform(X_train).shape[1]
        for pname, make in probes.items():
            r = cv_eval(pipe(groups, make()))
            rows.append({"Features": vname, "n features": n, "Model": pname, **r})
        print(f"  done: {vname} ({n} features)")
    res = pd.DataFrame(rows)
    save_table(res, "step6f_feature_engineering")
    acc = res.pivot(index="Features", columns="Model", values="CV accuracy").loc[list(variants)]
    acc.insert(0, "n features", res.drop_duplicates("Features").set_index("Features")["n features"])
    show(acc, "CV accuracy with engineered features")
    show(res.pivot(index="Features", columns="Model", values="CV AUC").loc[list(variants)], "CV AUC")
    base = acc.iloc[0, 1:]
    show((acc.iloc[:, 1:] - base), "Change in CV accuracy vs. the original 15 features")

    fig, ax = plt.subplots(figsize=(9, 3.8))
    sns.heatmap(acc.iloc[:, 1:], annot=True, fmt=".4f", cmap="YlGnBu", ax=ax)
    ax.set_title("Step 6f - CV accuracy with engineered features")
    savefig("21_feature_engineering")


if __name__ == "__main__":
    main()
