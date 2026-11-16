"""
STEP 5b - DIMENSIONALITY REDUCTION BY FEATURE EXTRACTION: PCA
Replace the 15 features with k uncorrelated principal components, for k = 1 ... 15.
PCA is inside the pipeline, so it is fitted on the training folds only.

Run:  python 05b_pca.py
"""
from common import *
from sklearn.decomposition import PCA


def main():
    header("STEP 5b - PCA")
    Xp = PRE().fit_transform(X_train)
    pca = PCA().fit(Xp)
    evr = pca.explained_variance_ratio_

    rows = []
    for k in range(1, 16):
        r = cv_eval(Pipeline([("pre", PRE()), ("pca", PCA(n_components=k)), ("clf", LR())]))
        rows.append({"components": k, "explained variance": evr[:k].sum(), **r})
        print(f"  k={k:2d}  explained variance={evr[:k].sum():.3f}  CV acc={r['CV accuracy']:.4f}  AUC={r['CV AUC']:.4f}")
    res_k = pd.DataFrame(rows)
    save_table(res_k, "step5b_pca_by_k")

    fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
    ax[0].bar(range(1, 16), evr, color="#8DA0CB", label="each component")
    ax[0].plot(range(1, 16), evr.cumsum(), "o-", c="#C44E52", label="cumulative")
    ax[0].axhline(.95, ls="--", c="k", lw=1)
    ax[0].set_xlabel("component"); ax[0].set_ylabel("explained variance")
    ax[0].set_title("PCA - explained variance"); ax[0].legend()
    ax[1].errorbar(res_k["components"], res_k["CV accuracy"], yerr=res_k["CV acc std"], fmt="o-", capsize=3)
    ax[1].axhline(BASELINE_TRAIN, ls="--", c="k", lw=1, label="baseline")
    ax[1].set_xlabel("number of components"); ax[1].set_ylabel("CV accuracy")
    ax[1].set_title("Accuracy vs. number of components"); ax[1].legend()
    pcs = pca.transform(Xp)
    ax[2].scatter(pcs.iloc[:, 0], pcs.iloc[:, 1], c=y_train.map({0: "#4C72B0", 1: "#C44E52"}), s=6, alpha=.5)
    ax[2].set_xlabel("PC1"); ax[2].set_ylabel("PC2"); ax[2].set_title("First two components (red = CHD)")
    savefig("12_pca")

    loadings = pd.DataFrame(pca.components_[:3].T, index=Xp.columns, columns=["PC1", "PC2", "PC3"])
    show(loadings.sort_values("PC1", key=abs, ascending=False), "Loadings of the first three components")

    k95 = int(np.argmax(evr.cumsum() >= .95) + 1)
    k_small = int(res_k.loc[res_k["components"] < 15, "CV accuracy"].idxmax() + 1)
    by_k = res_k.set_index("components")
    out = []
    for k, label in [(2, "PCA (2 components)"), (k95, f"PCA ({k95} components = 95% variance)"),
                     (k_small, f"PCA ({k_small} components = best k below 15)")]:
        out.append({"Method": label, "Family": "Feature extraction", "Features used": f"{k} PCs",
                    "CV accuracy": by_k.loc[k, "CV accuracy"], "CV acc std": by_k.loc[k, "CV acc std"],
                    "CV AUC": by_k.loc[k, "CV AUC"]})
    res = pd.DataFrame(out).drop_duplicates("Method")
    save_table(res, "step5b_pca")
    show(res, "Results")

    print(f"""
CONCLUSIONS
- The variance is spread over many components: {k95} are needed for 95% (the first two hold only {evr[:2].sum():.0%}).
- Accuracy only reaches its maximum when (almost) all components are kept; with few components it falls to the baseline.
- PCA looks for directions of high VARIANCE, not directions that separate CHD from non-CHD, so it does not help here.""")


if __name__ == "__main__":
    main()
