"""
STEP 6a - TRY EVERY MODEL WITH ITS DEFAULT SETTINGS
17 model families x 4 feature subsets (all 15 features, Lasso 10, AIC 8, BIC 5).
Nothing is tuned here: this shows what each model does "out of the box".
Scored on the same folds (FRESH_CV) as the tuned models in 06b, so the two tables are comparable.

Run:  python 06a_models_default.py        (about 2 minutes)
"""
import time
from common import *
from models import ZOO, FEATURE_SET_CHOICES, make_pipeline


def main():
    header("STEP 6a - ALL MODELS, DEFAULT SETTINGS")
    rows = []
    for name in ZOO:
        for fs in FEATURE_SET_CHOICES:
            t = time.time()
            r = cv_eval(make_pipeline(name, fs), cv=FRESH_CV)
            rows.append({"Model": name, "Feature set": fs, **r, "seconds": time.time() - t})
        best = max(rows[-len(FEATURE_SET_CHOICES):], key=lambda d: d["CV accuracy"])
        print(f"  {name:24s} best feature set = {best['Feature set']:8s} CV acc = {best['CV accuracy']:.4f}"
              f"   AUC = {best['CV AUC']:.4f}")
    res = pd.DataFrame(rows)
    save_table(res, "step6a_models_default")

    acc = res.pivot(index="Model", columns="Feature set", values="CV accuracy")[FEATURE_SET_CHOICES]
    acc["best"] = acc.max(axis=1)
    acc = acc.sort_values("best", ascending=False)
    show(acc, "CV accuracy, default settings (rows = models, columns = feature subsets)")
    auc = res.pivot(index="Model", columns="Feature set", values="CV AUC")[FEATURE_SET_CHOICES].loc[acc.index]
    show(auc, "CV AUC, default settings")
    print(f"\nBaseline accuracy (always 'no CHD') = {BASELINE_TRAIN:.4f}")
    print("Models below the baseline with default settings:", list(acc.index[acc["best"] < BASELINE_TRAIN]))

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(acc[FEATURE_SET_CHOICES], annot=True, fmt=".4f", cmap="RdYlGn", center=BASELINE_TRAIN, ax=ax)
    ax.set_title(f"Step 6a - CV accuracy with default settings (baseline = {BASELINE_TRAIN:.4f})")
    savefig("17_models_default")


if __name__ == "__main__":
    main()
