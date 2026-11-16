"""
STEP 4b - CATEGORICAL ENCODING OF `education`  (method from "Lab 1 - Data Preprocessing")
`education` has four levels (1, 2, 3, 4). So far it was used as a number, which assumes that the
step from level 1 to 2 has the same effect as the step from 3 to 4.
One-Hot Encoding (OneHotEncoder + ColumnTransformer, as in the lab) replaces the column by four
0/1 columns, so the model can learn a separate effect for every level.

Four variants are compared:
    numeric          education kept as one number (what the rest of the project uses)
    one-hot          four 0/1 columns
    one-hot, drop 1  three 0/1 columns (the first level is the reference - avoids redundant columns)
    without          the column is removed

The encoder is inside the pipeline, so it is fitted on the training folds only.

Run:  python 04b_onehot_encoding.py
"""
from common import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

OTHERS = [c for c in FEATURES if c != "education"]


def preprocess(variant):
    """impute -> encode education -> scale"""
    num = ("num", SimpleImputer(strategy="median"), OTHERS)
    if variant == "numeric":
        cols = ColumnTransformer([num, ("edu", SimpleImputer(strategy="most_frequent"), ["education"])],
                                 verbose_feature_names_out=False)
    elif variant == "without":
        cols = ColumnTransformer([num], verbose_feature_names_out=False)
    else:
        onehot = OneHotEncoder(drop=("first" if variant == "one-hot, drop 1" else None), sparse_output=False,
                               handle_unknown="ignore")
        edu = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("onehot", onehot)])
        cols = ColumnTransformer([num, ("edu", edu, ["education"])], verbose_feature_names_out=False)
    return Pipeline([("cols", cols), ("scale", StandardScaler())])


def main():
    header("STEP 4b - ONE-HOT ENCODING OF education")
    train = X_train.assign(TenYearCHD=y_train)
    by_level = train.groupby("education")[TARGET].agg(["count", "mean"]).rename(columns={"mean": "CHD rate"})
    show(by_level, "CHD rate for each education level (train)")
    print("\nThe CHD rate does not fall in equal steps from level to level -> the 'numeric' assumption is"
          "\nonly approximately true, so one-hot encoding is worth testing.")

    sample = X_train.dropna(subset=["education"]).groupby("education").head(1).sort_values("education")
    encoded = preprocess("one-hot")[:-1].fit(X_train).transform(sample)
    show(sample[["education"]].join(encoded[[c for c in encoded.columns if c.startswith("education")]]),
         "Example: one person from each level, before and after One-Hot Encoding")

    models = {"Logistic Regression": LR(), "KNN (k=25)": KNeighborsClassifier(25),
              "Random Forest": RandomForestClassifier(300, min_samples_leaf=5, random_state=SEED, n_jobs=1)}
    rows = []
    for variant in ["numeric", "one-hot", "one-hot, drop 1", "without"]:
        n = preprocess(variant).fit_transform(X_train).shape[1]
        for mname, m in models.items():
            rows.append({"education encoding": variant, "n features": n, "Model": mname,
                         **cv_eval(Pipeline([("pre", preprocess(variant)), ("m", clone(m))]))})
        print(f"  done: {variant} ({n} features)")
    res = pd.DataFrame(rows)
    save_table(res, "step4b_onehot_encoding")
    order = ["numeric", "one-hot", "one-hot, drop 1", "without"]
    acc = res.pivot(index="education encoding", columns="Model", values="CV accuracy").loc[order]
    acc.insert(0, "n features", res.drop_duplicates("education encoding").set_index("education encoding")["n features"])
    show(acc, "CV accuracy for each encoding of education")
    show(res.pivot(index="education encoding", columns="Model", values="CV AUC").loc[order], "CV AUC")

    fig, ax = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1, 1.6]})
    ax[0].bar(by_level.index.astype(int).astype(str), by_level["CHD rate"], color="#C44E52")
    ax[0].axhline(y_train.mean(), ls="--", c="k", lw=1)
    ax[0].set_xlabel("education level"); ax[0].set_ylabel("CHD rate"); ax[0].set_title("CHD rate by education level")
    sns.heatmap(acc.iloc[:, 1:], annot=True, fmt=".4f", cmap="YlGnBu", ax=ax[1])
    ax[1].set_title("Step 4b - CV accuracy by encoding of education"); ax[1].set_ylabel("")
    savefig("10b_onehot_encoding")

    lr = acc["Logistic Regression"]
    print(f"""
CONCLUSIONS
- Logistic regression: numeric {lr['numeric']:.4f} | one-hot {lr['one-hot']:.4f} | one-hot drop 1 {lr['one-hot, drop 1']:.4f} | without {lr['without']:.4f}
- The differences are far below the CV standard deviation (about 0.005): education carries very little
  information about CHD once age and blood pressure are known (step 5 also ranks it among the least useful).
DECISION: keep education as a single numeric column. Step 5 (Lasso) removes it anyway.""")


if __name__ == "__main__":
    main()
