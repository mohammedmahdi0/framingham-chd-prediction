"""
STEP 4 - FEATURE SCALING
Compare four scaling options (none, Standard, MinMax, Robust) on four model families.

Run:  python 04_feature_scaling.py
"""
from common import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def main():
    header("STEP 4 - FEATURE SCALING")

    ranges = pd.DataFrame({"min": X_train.min(), "max": X_train.max(), "mean": X_train.mean(), "std": X_train.std()})
    show(ranges, "Raw feature ranges - very different scales")

    models = {"Logistic Regression": LR(), "KNN (k=25)": KNeighborsClassifier(25), "SVM (RBF)": SVC(C=1.0),
              "Random Forest": RandomForestClassifier(200, min_samples_leaf=5, random_state=SEED, n_jobs=1)}
    rows = []
    for sname in SCALERS:
        for mname, m in models.items():
            p = Pipeline([("pre", make_pre(scaler=sname)), ("m", clone(m))])
            s = cross_validate(p, X_train, y_train, cv=RCV, scoring="accuracy", n_jobs=-1)
            rows.append({"Scaler": sname, "Model": mname, "CV accuracy": s["test_score"].mean(),
                         "CV acc std": s["test_score"].std()})
        print(f"  done: {sname}")
    res = pd.DataFrame(rows)
    save_table(res, "step4_scaling")
    piv = res.pivot(index="Scaler", columns="Model", values="CV accuracy").loc[list(SCALERS)]
    show(piv, "CV accuracy by scaler and model")

    fig, ax = plt.subplots(figsize=(8, 3.2))
    sns.heatmap(piv, annot=True, fmt=".4f", cmap="YlGnBu", ax=ax)
    ax.set_title("Step 4 - CV accuracy by scaler and model")
    savefig("10_scaling")

    print("""
CONCLUSIONS
- StandardScaler is best (or tied) for logistic regression and SVM.
- Random Forest gives the same accuracy with every scaler: tree models do not need scaling.
DECISION: StandardScaler.""")


if __name__ == "__main__":
    main()
