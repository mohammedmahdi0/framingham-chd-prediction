"""
STEP 5a - DIMENSIONALITY REDUCTION BY REGULARIZATION: Ridge (L2), Lasso (L1), Elastic Net
Penalize large coefficients. Lasso can push coefficients to exactly zero = it removes features.

The penalty strength C is tuned by an inner 5-fold CV; the whole thing is scored by the outer
repeated CV (nested cross-validation), so the scores are not optimistic.

Run:  python 05a_regularization_ridge_lasso_elasticnet.py
"""
from common import *
from sklearn.model_selection import GridSearchCV

C_GRID = np.logspace(-3, 2, 16)


def tuned_lr(l1_ratio):
    """Logistic regression with C (and, for Elastic Net, l1_ratio) tuned by inner CV."""
    if l1_ratio is None:     # Elastic Net: tune the L1 / L2 mix as well
        grid, start = {"C": C_GRID, "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}, 0.5
    else:
        grid, start = {"C": C_GRID}, l1_ratio
    return GridSearchCV(LR(solver="saga", l1_ratio=start), grid, scoring="accuracy", cv=CV5, n_jobs=1)


def main():
    header("STEP 5a - REGULARIZATION: RIDGE / LASSO / ELASTIC NET")
    Xp = PRE().fit_transform(X_train)

    # ---- regularization paths (illustration)
    Cs = np.logspace(-3, 2, 30)
    kinds = {"Ridge (L2)": 0.0, "Lasso (L1)": 1.0, "Elastic Net (l1_ratio = 0.5)": 0.5}
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), sharey=True)
    for a, (name, l1r) in zip(axes, kinds.items()):
        path = pd.DataFrame([LR(C=C, l1_ratio=l1r, solver="saga").fit(Xp, y_train).coef_[0] for C in Cs],
                            index=Cs, columns=Xp.columns)
        path.plot(ax=a, legend=False, logx=True, lw=1.6, colormap="tab20")
        a.set_title(name)
        a.set_xlabel("C  (small C = strong penalty)")
        a.axhline(0, c="k", lw=.6)
    axes[0].set_ylabel("coefficient")
    axes[2].legend(loc="center left", bbox_to_anchor=(1.01, .5), fontsize=8)
    fig.suptitle("Regularization paths - how each penalty shrinks the coefficients", y=1.02)
    savefig("11_regularization_paths")

    # ---- experiment
    rows, selected, coefs = [], {}, {}
    record(rows, "All 15 features (no reduction)", "Baseline", Pipeline([("pre", PRE()), ("clf", LR())]), 15)
    for name, l1r in {"Ridge (L2)": 0.0, "Lasso (L1)": 1.0, "Elastic Net": None}.items():
        full = Pipeline([("pre", PRE()), ("clf", tuned_lr(l1r))]).fit(X_train, y_train)
        gs = full.named_steps["clf"]
        coef = pd.Series(gs.best_estimator_.coef_[0], index=FEATURES)
        coefs[name] = coef
        selected[name] = list(coef[coef.abs() > 1e-6].index)
        print(f"\n{name}: best parameters = {gs.best_params_} | non-zero coefficients = {len(selected[name])}")
        record(rows, name, "Regularization", Pipeline([("pre", PRE()), ("clf", tuned_lr(l1r))]), len(selected[name]))

    res = pd.DataFrame(rows)
    save_table(res, "step5a_regularization")
    save_json(selected, "step5a_selected")
    show(res.sort_values("CV accuracy", ascending=False), "Results")
    show(pd.DataFrame(coefs).assign(order=lambda d: d["Lasso (L1)"].abs()).sort_values("order", ascending=False)
         .drop(columns="order"), "Coefficients at the best C (0 = feature removed)")
    print("\nFeatures removed by Lasso:", [f for f in FEATURES if f not in selected["Lasso (L1)"]])


if __name__ == "__main__":
    main()
