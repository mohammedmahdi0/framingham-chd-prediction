"""
STEP 6b - HYPER-PARAMETER TUNING OF EVERY MODEL
For each of the 17 models: randomized search (30 settings x 5-fold CV), optimizing ACCURACY.
The feature subset from step 5 is searched together with the model's own hyper-parameters.

The best setting of each model is then re-scored on FRESH cross-validation folds (different random
seed), because the score reported by the search itself is slightly optimistic (it is the maximum of
30 noisy numbers).

The test set is not used.

Run:  python 06b_hyperparameter_tuning.py      (about 2 minutes)
"""
import time
from common import *
from models import ZOO, make_pipeline, search_space, build_tuned
from sklearn.model_selection import RandomizedSearchCV

N_ITER = 30


def main():
    header("STEP 6b - HYPER-PARAMETER TUNING")
    best, rows = {}, []
    for name in ZOO:
        t = time.time()
        search = RandomizedSearchCV(make_pipeline(name), search_space(name), n_iter=N_ITER, scoring="accuracy",
                                    cv=CV5, random_state=SEED, n_jobs=-1, refit=False)
        search.fit(X_train, y_train)
        params = {k.split("__", 1)[1]: v for k, v in search.best_params_.items()}
        best[name] = {"params": params, "search_score": float(search.best_score_)}
        fresh = cv_eval(build_tuned(name, best), cv=FRESH_CV)
        best[name].update(cv_accuracy=float(fresh["CV accuracy"]), cv_std=float(fresh["CV acc std"]),
                          cv_auc=float(fresh["CV AUC"]))
        rows.append({"Model": name, "Feature set": params["feature_set"], "Search score": search.best_score_,
                     "CV accuracy (fresh folds)": fresh["CV accuracy"], "CV acc std": fresh["CV acc std"],
                     "CV AUC": fresh["CV AUC"], "seconds": time.time() - t})
        print(f"  {name:24s} CV acc = {fresh['CV accuracy']:.4f} +/- {fresh['CV acc std']:.4f}  "
              f"AUC = {fresh['CV AUC']:.4f}  ({time.time() - t:.0f}s)  {params}")
    save_json(best, "step6b_best_params")

    res = pd.DataFrame(rows).sort_values("CV accuracy (fresh folds)", ascending=False).reset_index(drop=True)
    res.index += 1
    save_table(res, "step6b_tuned_models")
    show(res, "Tuned models, best first")

    default_path = RES_DIR / "step6a_models_default.csv"
    if default_path.exists():
        d = pd.read_csv(default_path).groupby("Model")["CV accuracy"].max().rename("default (best feature set)")
        cmp = res.set_index("Model")[["CV accuracy (fresh folds)"]].join(d)
        cmp["gain from tuning"] = cmp["CV accuracy (fresh folds)"] - cmp["default (best feature set)"]
        show(cmp, "Tuned vs. default")

    d = res.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(d["Model"], d["CV accuracy (fresh folds)"], xerr=d["CV acc std"], color="#4C72B0", capsize=2,
            error_kw={"lw": .8})
    ax.axvline(BASELINE_TRAIN, ls="--", c="k", lw=1, label="baseline")
    ax.set_xlim(0.835, 0.868)
    ax.set_xlabel("CV accuracy after tuning (error bar = std over 15 folds)")
    ax.set_title("Step 6b - tuned models")
    ax.legend(loc="lower right")
    savefig("18_models_tuned")


if __name__ == "__main__":
    main()
