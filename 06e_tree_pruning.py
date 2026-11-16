"""
STEP 6e - DECISION-TREE PRUNING  (method from "Lecture 5 - DT / RF / XGB")
A decision tree that is allowed to grow without limits memorizes the training data (over-fitting):
very high training accuracy, poor accuracy on new people. The lecture gives two cures:

    Pre-pruning  : stop the tree early (limit the depth / the minimum number of samples in a leaf)
    Post-pruning : let the tree grow fully, then cut back the branches that do not pay for themselves
                   (cost-complexity pruning, parameter ccp_alpha: larger alpha = more cutting)

Four trees are compared:
    1) no pruning            2) pre-pruning            3) post-pruning            4) both together

The pruning parameters are chosen by an inner 5-fold CV, and every variant is scored by the outer
repeated CV (nested cross-validation). The test set is not used. No class balancing is used.

Run:  python 06e_tree_pruning.py      (about 1 minute)
"""
from common import *
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree

PRE_GRID = {"max_depth": [1, 2, 3, 4, 5, 6, 8, 10], "min_samples_leaf": [1, 5, 10, 20, 50, 100]}


def tree(**kw):
    return DecisionTreeClassifier(random_state=SEED, **kw)


def main():
    header("STEP 6e - DECISION-TREE PRUNING")
    Xp = PRE().fit_transform(X_train)

    # ---- the unpruned tree and its pruning path
    full = tree().fit(Xp, y_train)
    print(f"Unpruned tree: depth = {full.get_depth()}, leaves = {full.get_n_leaves()}, "
          f"training accuracy = {full.score(Xp, y_train):.4f}  <- it memorized the training set")
    path = full.cost_complexity_pruning_path(Xp, y_train)
    alphas = np.unique(np.round(np.quantile(path.ccp_alphas[:-1], np.linspace(0, 1, 30)), 6))
    alphas = np.unique(np.concatenate([[0.0], alphas, [0.001, 0.002, 0.003, 0.005]]))
    print(f"The pruning path has {len(path.ccp_alphas)} possible alphas; {len(alphas)} of them are tried.")

    # ---- how accuracy changes with the strength of post-pruning
    curve = []
    for a in alphas:
        s = cross_validate(Pipeline([("pre", PRE()), ("clf", tree(ccp_alpha=a))]), X_train, y_train, cv=RCV,
                           scoring="accuracy", return_train_score=True, n_jobs=-1)
        leaves = tree(ccp_alpha=a).fit(Xp, y_train).get_n_leaves()
        curve.append({"ccp_alpha": a, "leaves": leaves, "train accuracy": s["train_score"].mean(),
                      "validation accuracy": s["test_score"].mean()})
    curve = pd.DataFrame(curve)
    save_table(curve, "step6e_pruning_curve")
    show(curve, "Effect of post-pruning strength (ccp_alpha)")

    # ---- the four variants, nested CV
    POST_GRID = {"ccp_alpha": list(alphas)}
    variants = {
        "1) No pruning": tree(),
        "2) Pre-pruning (depth, leaf size)": GridSearchCV(tree(), PRE_GRID, scoring="accuracy", cv=CV5, n_jobs=1),
        "3) Post-pruning (ccp_alpha)": GridSearchCV(tree(), POST_GRID, scoring="accuracy", cv=CV5, n_jobs=1),
        "4) Pre + post pruning": GridSearchCV(tree(), {**PRE_GRID, "ccp_alpha": [0.0, 0.0005, 0.001, 0.002, 0.003]},
                                              scoring="accuracy", cv=CV5, n_jobs=1),
    }
    rows, fitted = [], {}
    for name, est in variants.items():
        s = cross_validate(Pipeline([("pre", PRE()), ("clf", clone(est))]), X_train, y_train, cv=RCV,
                           scoring=["accuracy", "roc_auc"], return_train_score=True, n_jobs=-1)
        m = clone(est).fit(Xp, y_train)
        t = m.best_estimator_ if hasattr(m, "best_estimator_") else m
        fitted[name] = t
        rows.append({"Tree": name, "Train accuracy": s["train_accuracy"].mean(), "CV accuracy": s["test_accuracy"].mean(),
                     "CV acc std": s["test_accuracy"].std(), "CV AUC": s["test_roc_auc"].mean(),
                     "depth": t.get_depth(), "leaves": t.get_n_leaves(),
                     "chosen settings": str(getattr(m, "best_params_", "-"))})
        print(f"  {name:36s} train = {rows[-1]['Train accuracy']:.4f}  CV = {rows[-1]['CV accuracy']:.4f}  "
              f"leaves = {rows[-1]['leaves']}")
    res = pd.DataFrame(rows)
    save_table(res, "step6e_tree_pruning")
    show(res, f"Decision tree with and without pruning (baseline = {BASELINE_TRAIN:.4f})")

    fig, ax = plt.subplots(1, 2, figsize=(15, 4.4))
    ax[0].plot(curve["ccp_alpha"], curve["train accuracy"], "o-", ms=3, label="train", c="#4C72B0")
    ax[0].plot(curve["ccp_alpha"], curve["validation accuracy"], "o-", ms=3, label="validation (CV)", c="#C44E52")
    ax[0].axhline(BASELINE_TRAIN, ls="--", c="k", lw=1, label="baseline")
    ax[0].set_xlabel("ccp_alpha (pruning strength)"); ax[0].set_ylabel("accuracy")
    ax[0].set_title("Post-pruning: the gap between train and validation closes"); ax[0].legend()
    x = np.arange(len(res))
    ax[1].bar(x - .2, res["Train accuracy"], width=.4, label="train", color="#4C72B0")
    ax[1].bar(x + .2, res["CV accuracy"], width=.4, label="validation (CV)", color="#C44E52")
    ax[1].axhline(BASELINE_TRAIN, ls="--", c="k", lw=1)
    ax[1].set_xticks(x); ax[1].set_xticklabels([n.split(") ")[1] for n in res["Tree"]], fontsize=8, rotation=10)
    ax[1].set_ylim(0.70, 1.02); ax[1].set_title("Step 6e - four trees"); ax[1].legend()
    savefig("20_tree_pruning")

    best_name = res.sort_values("CV accuracy", ascending=False).iloc[0]["Tree"]
    best_tree = fitted[best_name]
    if best_tree.get_n_leaves() > 1:
        fig, ax = plt.subplots(figsize=(min(4 + 1.6 * best_tree.get_n_leaves(), 22), 3 + 1.4 * best_tree.get_depth()))
        plot_tree(best_tree, feature_names=list(Xp.columns), class_names=["No CHD", "CHD"], filled=True, rounded=True,
                  impurity=False, fontsize=8, ax=ax)
        ax.set_title(f"The best pruned tree: {best_name} (features are standardized)")
        savefig("20b_pruned_tree")
    else:
        print("\nThe best pruned tree is a single leaf: it predicts 'no CHD' for everyone (= the baseline).")

    none, top = res.iloc[0], res.sort_values("CV accuracy", ascending=False).iloc[0]
    print(f"""
CONCLUSIONS
- Without pruning the tree has {int(none['leaves'])} leaves: train accuracy {none['Train accuracy']:.4f} but CV accuracy only {none['CV accuracy']:.4f}
  -> clear over-fitting, exactly as described in the lecture.
- Pruning removes the over-fitting: the best variant ({top['Tree']}) has {int(top['leaves'])} leaves,
  train {top['Train accuracy']:.4f} and CV {top['CV accuracy']:.4f} (+{top['CV accuracy'] - none['CV accuracy']:.4f} over the unpruned tree).
- Even the best pruned tree stays close to the baseline ({BASELINE_TRAIN:.4f}) and below logistic regression,
  so a single tree is not the final model. Pruning fixes over-fitting; it cannot add information.""")


if __name__ == "__main__":
    main()
