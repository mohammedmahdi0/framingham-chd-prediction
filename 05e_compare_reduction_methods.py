"""
STEP 5e - COMPARE ALL DIMENSIONALITY-REDUCTION METHODS AND CHOOSE ONE
Reads the results saved by 05a, 05b, 05c and 05d (run those first) and puts them side by side.

Run:  python 05e_compare_reduction_methods.py
"""
from common import *


def main():
    header("STEP 5e - COMPARISON OF ALL DIMENSIONALITY-REDUCTION METHODS")
    parts = ["step5a_regularization", "step5b_pca", "step5c_stepwise", "step5d_other_selection"]
    missing = [p for p in parts if not (RES_DIR / f"{p}.csv").exists()]
    if missing:
        sys.exit(f"Missing results: {missing}. Run scripts 05a, 05b, 05c and 05d first.")
    res = (pd.concat([pd.read_csv(RES_DIR / f"{p}.csv") for p in parts])
           .sort_values("CV accuracy", ascending=False).reset_index(drop=True))
    res.index += 1
    save_table(res, "step5_all_methods")
    show(res, "All methods, best first (same classifier, same repeated 5-fold CV)")

    pal = {"Baseline": "#7F7F7F", "Regularization": "#4C72B0", "Feature extraction": "#DD8452",
           "Feature selection": "#55A868"}
    d = res.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(d["Method"], d["CV accuracy"], xerr=d["CV acc std"], color=d["Family"].map(pal), capsize=2,
            error_kw={"lw": .8})
    ax.axvline(BASELINE_TRAIN, ls="--", c="k", lw=1)
    ax.set_xlim(0.84, 0.866)
    ax.set_xlabel("CV accuracy (error bar = std over 15 folds)")
    ax.set_title("Step 5 - dimensionality-reduction methods (dashed = baseline)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in pal.values()], labels=list(pal),
              loc="lower right", fontsize=8)
    savefig("15_reduction_comparison")

    # which features does each method keep?
    selected = {}
    for n in ["step5a_selected", "step5c_selected", "step5d_selected"]:
        selected.update(load_json(n))
    m = pd.DataFrame({k: [int(f in v) for f in FEATURES] for k, v in selected.items()}, index=FEATURES)
    m["times selected"] = m.sum(axis=1)
    m = m.sort_values("times selected", ascending=False)
    save_table(m, "step5_selected_features", index=True)
    show(m[["times selected"]].T, f"How many of the {len(selected)} methods keep each feature")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.heatmap(m.drop(columns="times selected"), cmap=["#F0F0F0", "#55A868"], cbar=False, linewidths=.5,
                linecolor="white", ax=ax)
    ax.set_title("Which features does each method keep? (green = kept)")
    plt.xticks(rotation=35, ha="right")
    savefig("16_selected_features")

    # Methods whose accuracy is within 0.0005 of the best are a tie (the CV std is ten times larger).
    # Among tied methods the one that keeps the FEWEST features is preferred.
    top = res[res["CV accuracy"] >= res["CV accuracy"].max() - 0.0005].copy()
    top["n"] = top["Features used"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    show(top.drop(columns="n"), "Tied for first place (within 0.0005 of the best)")
    best = top.sort_values(["n", "CV accuracy"], ascending=[True, False]).iloc[0]
    print(f"""
CONCLUSIONS
- {len(top)} methods are tied for first place at about {res['CV accuracy'].max():.4f}.
- Among them, {best['Method']} uses the fewest features ({best['Features used']}), so it is the preferred one.
- All methods are within about {res['CV accuracy'].iloc[0] - res['CV accuracy'].iloc[-2]:.3f} of each other, except PCA with few components.
- Features kept by every method: {list(m.index[m['times selected'] == len(selected)])}
- Least useful features: {list(m.index[-2:])}
DECISION: regularization (Lasso / Elastic Net) is the preferred reduction method. The feature subsets
'lasso10', 'aic8' and 'bic5' are stored in common.py and are tried with every model in step 6.""")


if __name__ == "__main__":
    main()
