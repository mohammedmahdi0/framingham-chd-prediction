"""
STEP 5c - FEATURE SELECTION WITH AIC AND BIC (stepwise logistic regression)

    AIC = 2k - 2 ln(L)          BIC = k ln(n) - 2 ln(L)
    k = number of parameters, L = likelihood of the model, n = number of rows.  Lower is better.
    Both reward goodness of fit and punish extra features; BIC punishes them harder.

Forward : start with no features, keep adding the one that lowers the criterion most.
Backward: start with all features, keep removing the one whose removal lowers the criterion most.

The selector (class StepwiseSelector in common.py) sits inside the pipeline, so the selection is
repeated on every training fold.

Run:  python 05c_stepwise_aic_bic.py
"""
from common import *


def main():
    header("STEP 5c - STEPWISE SELECTION WITH AIC / BIC")
    Xp = PRE().fit_transform(X_train)

    rows, selected = [], {}
    for crit in ["aic", "bic"]:
        for direction in ["forward", "backward"]:
            name = f"Stepwise {direction} {crit.upper()}"
            sel = StepwiseSelector(crit, direction).fit(Xp, y_train)
            selected[name] = sel.selected_
            print(f"\n{name}: {crit.upper()} = {sel.best_score_:.2f} | kept {len(sel.selected_)}: {sel.selected_}")
            record(rows, name, "Feature selection",
                   Pipeline([("pre", PRE()), ("sel", StepwiseSelector(crit, direction)), ("clf", LR())]),
                   len(sel.selected_))
    res = pd.DataFrame(rows)
    save_table(res, "step5c_stepwise")
    save_json(selected, "step5c_selected")
    show(res, "Results")

    # ---- the forward AIC path, step by step
    fwd = StepwiseSelector("aic", "forward").fit(Xp, y_train)
    steps = pd.DataFrame([{"step": i, "feature added": (h[0][-1] if h[0] else "(intercept only)"), "AIC": h[1]}
                          for i, h in enumerate(fwd.history_)])
    show(steps, "Forward selection: AIC after each added feature")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(steps["step"], steps["AIC"], "o-", c="#4C72B0")
    for _, r in steps.iterrows():
        ax.annotate(r["feature added"], (r["step"], r["AIC"]), textcoords="offset points", xytext=(4, 6),
                    fontsize=8, rotation=20)
    ax.set_xlabel("step"); ax.set_ylabel("AIC (lower is better)")
    ax.set_title("Forward stepwise selection - AIC after each added feature")
    savefig("13_aic_forward")

    # ---- the statistical model on the AIC-selected features
    yv = y_train.reset_index(drop=True)
    full_model = sm.Logit(yv, sm.add_constant(Xp.reset_index(drop=True))).fit(disp=0)
    aic_model = sm.Logit(yv, sm.add_constant(Xp[fwd.selected_].reset_index(drop=True))).fit(disp=0)
    print(f"\nFull model (15 features): AIC = {full_model.aic:.2f} | BIC = {full_model.bic:.2f}")
    print(f"AIC model  ({len(fwd.selected_)} features): AIC = {aic_model.aic:.2f} | BIC = {aic_model.bic:.2f}"
          f" | pseudo R2 = {aic_model.prsquared:.4f}")
    odds = pd.DataFrame({"coef": aic_model.params, "odds ratio": np.exp(aic_model.params),
                         "p-value": aic_model.pvalues}).sort_values("odds ratio", ascending=False)
    show(odds, "Logistic regression on the AIC-selected features (odds ratio per +1 standard deviation)")
    save_table(odds, "step5c_aic_model_odds_ratios", index=True)


if __name__ == "__main__":
    main()
