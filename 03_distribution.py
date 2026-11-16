"""
STEP 3 - DISTRIBUTIONS
Look at the shape of every continuous feature (skewness, outliers), then test whether
correcting the shape (log, Yeo-Johnson, quantile, winsorizing) improves accuracy.

Run:  python 03_distribution.py
"""
from common import *
from sklearn.neighbors import KNeighborsClassifier


def main():
    header("STEP 3 - DISTRIBUTIONS")

    fig, axes = plt.subplots(2, 4, figsize=(16, 6.5))
    for a, c in zip(axes.ravel(), CONT):
        sns.histplot(X_train[c].dropna(), kde=True, ax=a, color="#4C72B0")
        a.set_title(f"{c}  (skew = {X_train[c].skew():.2f})")
        a.set_xlabel("")
    fig.suptitle("Distributions of the continuous features (train)", y=1.02)
    savefig("07_distributions")

    fig, axes = plt.subplots(1, 8, figsize=(17, 3.8))
    for a, c in zip(axes, CONT):
        sns.boxplot(y=X_train[c], ax=a, color="#8DA0CB", fliersize=2)
        a.set_title(c)
        a.set_ylabel("")
    fig.suptitle("Box plots - outliers", y=1.04)
    savefig("08_boxplots")

    q1, q3 = X_train[CONT].quantile(.25), X_train[CONT].quantile(.75)
    iqr = q3 - q1
    out = ((X_train[CONT] < q1 - 1.5 * iqr) | (X_train[CONT] > q3 + 1.5 * iqr)).sum()
    stats = pd.DataFrame({"skewness": X_train[CONT].skew(), "kurtosis": X_train[CONT].kurt(),
                          "outliers (1.5 IQR)": out, "outliers %": out / len(X_train) * 100,
                          "min": X_train[CONT].min(), "max": X_train[CONT].max()}).sort_values("skewness", ascending=False)
    show(stats, "Shape of each continuous feature (train)")
    save_table(stats, "step3_distribution_stats", index=True)
    print("\nColumns with skewness > 0.5:", [c for c in CONT if X_train[c].skew() > 0.5])

    # ---------------- experiment
    rows = []
    for tname in TRANSFORMS:
        for mname, m in {"Logistic Regression": LR(), "KNN (k=25)": KNeighborsClassifier(25)}.items():
            p = Pipeline([("pre", make_pre(transform=tname)), ("m", m)])
            rows.append({"Transformation": tname, "Model": mname, **cv_eval(p)})
        print(f"  done: {tname}")
    res = pd.DataFrame(rows)
    save_table(res, "step3_distribution_transforms")
    piv = res.pivot(index="Transformation", columns="Model", values="CV accuracy").sort_values(
        "Logistic Regression", ascending=False)
    show(piv, "CV accuracy after each transformation")

    skew_after = pd.DataFrame({"raw": X_train[CONT].skew()})
    for tname in ["log(1+x) on skewed columns", "Yeo-Johnson power transform", "Winsorize (clip to 1st-99th percentile)"]:
        skew_after[tname] = make_pre(transform=tname, scaler="None").fit_transform(X_train)[CONT].skew()
    show(skew_after, "Skewness before / after (the transformations do fix the shape)")

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    raw = make_pre(scaler="None").fit_transform(X_train)["glucose"]
    sns.histplot(raw, kde=True, ax=ax[0], color="#4C72B0")
    ax[0].set_title(f"glucose - raw (skew {raw.skew():.2f})")
    for a, tname in zip(ax[1:], ["log(1+x) on skewed columns", "Yeo-Johnson power transform"]):
        v = make_pre(transform=tname, scaler="None").fit_transform(X_train)["glucose"]
        sns.histplot(v, kde=True, ax=a, color="#55A868")
        a.set_title(f"glucose - {tname.split(' ')[0]} (skew {v.skew():.2f})")
    savefig("09_glucose_transforms")

    print(f"""
CONCLUSIONS
- glucose is extremely right-skewed; cigsPerDay, sysBP and BMI are moderately skewed.
- The transformations fix the skew, but none of them beats "no transformation" for logistic regression
  (best: {piv.index[0]} = {piv.iloc[0]['Logistic Regression']:.4f}).
- Reason: very high glucose / blood-pressure values are themselves a sign of disease. Flattening them removes signal.
DECISION: keep the original distributions (no transformation).""")


if __name__ == "__main__":
    main()
