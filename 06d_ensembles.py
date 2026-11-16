"""
STEP 6d - ENSEMBLES: COMBINING SEVERAL MODELS
  - Soft voting : average the predicted probabilities of several models
  - Hard voting : every model votes "CHD" / "no CHD", the majority wins
  - Stacking    : a meta-model learns how to combine the models' predictions

Two groups of ensembles are tried:
  A) the best tuned models from 06b (top 3 / 5 / 8 by cross-validated accuracy)
  B) the stacked model of "Lab 2 - NLP / Stacking": KNN, SVC, Decision Tree, Random Forest, XGBoost,
     Naive Bayes and MLP, combined by an XGBoost meta-model
       - once with the lab's own settings, exactly as in the lab
       - once with the same seven models after tuning (06b)

The test set is not used. No class balancing is used.

Run:  python 06d_ensembles.py      (needs 06b first, about 4 minutes)
"""
import time
from common import *
from models import build_ensemble, LECTURE_MEMBERS


def main():
    header("STEP 6d - ENSEMBLES")
    best = load_json("step6b_best_params")
    ranking = sorted(best, key=lambda n: best[n]["cv_accuracy"], reverse=True)
    show(pd.DataFrame({"CV accuracy": {n: best[n]["cv_accuracy"] for n in ranking}}), "Tuned models, ranked")

    configs = {
        "Voting (top 3)": ("voting", ranking[:3]),
        "Voting (top 5)": ("voting", ranking[:5]),
        "Voting (top 8)": ("voting", ranking[:8]),
        "Hard voting (top 5)": ("hard_voting", ranking[:5]),
        "Stacking (top 3)": ("stacking", ranking[:3]),
        "Stacking (top 5)": ("stacking", ranking[:5]),
        "Lab 2 stacking (lab settings)": ("lecture", LECTURE_MEMBERS),
        "Lab 2 stacking (tuned models)": ("stacking_xgb", LECTURE_MEMBERS),
        "Lab 2 models, LR meta-model": ("stacking", LECTURE_MEMBERS),
    }
    rows, out = [], {}
    for name, (kind, members) in configs.items():
        t = time.time()
        if kind == "hard_voting":       # hard voting has no probabilities, so no AUC
            s = cross_validate(build_ensemble(kind, members, best), X_train, y_train, cv=FRESH_CV,
                               scoring="accuracy", n_jobs=-1)
            r = {"CV accuracy": s["test_score"].mean(), "CV acc std": s["test_score"].std(), "CV AUC": np.nan}
        else:
            r = cv_eval(build_ensemble(kind, members, best), cv=FRESH_CV)
        out[name] = {"kind": kind, "members": members, "cv_accuracy": float(r["CV accuracy"]),
                     "cv_std": float(r["CV acc std"]), "cv_auc": float(r["CV AUC"])}
        rows.append({"Ensemble": name, **r, "seconds": time.time() - t, "Members": ", ".join(members)})
        print(f"  {name:32s} CV acc = {r['CV accuracy']:.4f} +/- {r['CV acc std']:.4f}  AUC = {r['CV AUC']:.4f}"
              f"  ({time.time() - t:.0f}s)")
    save_json(out, "step6d_ensembles")
    res = pd.DataFrame(rows).sort_values("CV accuracy", ascending=False).reset_index(drop=True)
    save_table(res, "step6d_ensembles")
    show(res.drop(columns="Members"), "Ensembles")

    single = ranking[0]
    single_acc = best[single]["cv_accuracy"]
    print(f"\nBest single model: {single} = {single_acc:.4f}")
    print(f"Best ensemble    : {res.loc[0, 'Ensemble']} = {res.loc[0, 'CV accuracy']:.4f}"
          f"  (difference {res.loc[0, 'CV accuracy'] - single_acc:+.4f})")

    d = res.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = ["#DD8452" if "Lab 2" in n else "#4C72B0" for n in d["Ensemble"]]
    ax.barh(d["Ensemble"], d["CV accuracy"], xerr=d["CV acc std"], color=colors, capsize=2, error_kw={"lw": .8})
    ax.axvline(single_acc, c="#C44E52", lw=1.5, label=f"best single model ({single})")
    ax.axvline(BASELINE_TRAIN, ls="--", c="k", lw=1, label="baseline")
    ax.set_xlim(0.82, 0.866)
    ax.set_xlabel("CV accuracy (error bar = std over 15 folds)")
    ax.set_title("Step 6d - ensembles (orange = the Lab 2 stacking recipe)")
    ax.legend(loc="lower left", fontsize=8)
    savefig("19b_ensembles")


if __name__ == "__main__":
    main()
