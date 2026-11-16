"""
STEP 2 - MISSING DATA
Count the missing values, then compare 8 ways of handling them by cross-validated accuracy.

Run:  python 02_missing_data.py
"""
from common import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


def cv_drop_rows(model, cv=RCV):
    """'Drop incomplete rows' strategy.
    Rows are dropped from the TRAINING folds only. The validation fold keeps every row,
    because a real model must give a prediction for every person."""
    acc, auc = [], []
    for tr, va in cv.split(X_train, y_train):
        Xt, yt = X_train.iloc[tr], y_train.iloc[tr]
        keep = Xt.notna().all(axis=1)
        p = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", clone(model))])
        p.fit(Xt[keep], yt[keep])
        pr = p.predict_proba(X_train.iloc[va])[:, 1]
        acc.append(accuracy_score(y_train.iloc[va], pr >= .5))
        auc.append(roc_auc_score(y_train.iloc[va], pr))
    return {"CV accuracy": np.mean(acc), "CV acc std": np.std(acc), "CV AUC": np.mean(auc)}


def main():
    header("STEP 2 - MISSING DATA")

    miss = pd.DataFrame({"missing (train)": X_train.isna().sum(), "% (train)": X_train.isna().mean() * 100,
                         "missing (test)": X_test.isna().sum()})
    miss = miss[miss["missing (train)"] > 0].sort_values("% (train)", ascending=False)
    show(miss, "Missing values per column")
    n_bad = X_train.isna().any(axis=1).sum()
    print(f"\nTraining rows with at least one missing value: {n_bad} of {len(X_train)} ({n_bad/len(X_train):.1%})")

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(miss.index[::-1], miss["% (train)"][::-1], color="#4C72B0")
    ax.set_xlabel("% missing (train)")
    ax.set_title("Missing values per column")
    savefig("05_missing")

    related = pd.DataFrame({c: {"CHD rate when missing": y_train[X_train[c].isna()].mean(),
                                "CHD rate when present": y_train[X_train[c].notna()].mean(),
                                "n missing": X_train[c].isna().sum()} for c in miss.index}).T
    show(related, "Is missingness related to the target?")

    # ---------------- experiment
    probes = {"Logistic Regression": LR(),
              "Random Forest": RandomForestClassifier(300, min_samples_leaf=5, random_state=SEED, n_jobs=1)}
    rows = []
    for pname, probe in probes.items():
        rows.append({"Strategy": "Drop incomplete rows", "Model": pname, **cv_drop_rows(probe)})
        for iname, imp in IMPUTERS.items():
            p = Pipeline([("imp", clone(imp)), ("sc", StandardScaler()), ("m", clone(probe))])
            rows.append({"Strategy": iname, "Model": pname, **cv_eval(p)})
            print(f"  done: {pname:20s} | {iname}")
    res = pd.DataFrame(rows)
    save_table(res, "step2_missing_data")

    piv = res.pivot(index="Strategy", columns="Model", values="CV accuracy").sort_values(
        "Logistic Regression", ascending=False)
    show(piv, "CV accuracy of each missing-data strategy (repeated 5-fold, 15 fits)")

    fig, ax = plt.subplots(figsize=(9, 4))
    piv.iloc[::-1].plot.barh(ax=ax, color=["#4C72B0", "#55A868"])
    ax.axvline(BASELINE_TRAIN, ls="--", c="k", lw=1, label="baseline")
    ax.set_xlim(0.84, 0.862)
    ax.set_xlabel("CV accuracy")
    ax.set_title("Step 2 - missing-data strategies")
    ax.legend(loc="lower right")
    savefig("06_missing_strategies")

    print(f"""
CONCLUSIONS
- Best strategy for logistic regression: {piv.index[0]} ({piv.iloc[0, 0]:.4f}).
- Dropping incomplete rows throws away {n_bad/len(X_train):.0%} of the training data and is among the weakest options.
- The differences are tiny (smaller than the CV standard deviation, about 0.005).
DECISION: median imputation (simple, and the best or tied-best).""")


if __name__ == "__main__":
    main()
