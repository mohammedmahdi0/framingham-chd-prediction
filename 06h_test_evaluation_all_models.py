"""
STEP 6h - TEST-SET RESULTS OF EVERY TUNED MODEL AND ENSEMBLE
This is the first script that touches the test set.

Every model is trained on the full training set with the settings found in 06b (and the threshold
from 06c) and then scored once on the 848 test people.

IMPORTANT: the final model is the one with the best CROSS-VALIDATED accuracy, not the one that
happens to score highest in this table. Picking the winner by looking at test scores would turn the
test set into training data and make the reported accuracy optimistic.

Run:  python 06h_test_evaluation_all_models.py      (needs 06b, 06c, 06d first, about 3 minutes)
"""
from common import *
from models import build_tuned, build_ensemble
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve)


def evaluate(name, model, threshold, cv_acc):
    model.fit(X_train, y_train)
    if hasattr(model, "predict_proba"):
        p_tr, p_te = model.predict_proba(X_train)[:, 1], model.predict_proba(X_test)[:, 1]
        pred_tr, pred, auc = (p_tr >= threshold).astype(int), (p_te >= threshold).astype(int), roc_auc_score(y_test, p_te)
    else:       # hard voting gives classes only: no probabilities, no threshold, no AUC
        p_te, pred_tr, pred, auc, threshold = None, model.predict(X_train), model.predict(X_test), np.nan, np.nan
    tn = int(((pred == 0) & (y_test == 0)).sum()); tp = int(((pred == 1) & (y_test == 1)).sum())
    return {"Model": name, "CV accuracy": cv_acc, "Train accuracy": accuracy_score(y_train, pred_tr),
            "TEST accuracy": accuracy_score(y_test, pred), "Test AUC": auc,
            "Precision": precision_score(y_test, pred, zero_division=0), "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred), "Threshold": threshold, "CHD found": f"{tp}/{int(y_test.sum())}",
            "Correct": f"{tp + tn}/{len(y_test)}"}, p_te


def main():
    header("STEP 6h - TEST-SET RESULTS OF ALL MODELS")
    best, thresholds, ensembles = (load_json("step6b_best_params"), load_json("step6c_thresholds"),
                                   load_json("step6d_ensembles"))
    rows, probas = [], {}
    for name in best:
        r, probas[name] = evaluate(name, build_tuned(name, best), thresholds[name]["threshold"], best[name]["cv_accuracy"])
        rows.append(r)
        print(f"  {name:24s} CV = {r['CV accuracy']:.4f} | test = {r['TEST accuracy']:.4f}")
    for name, e in ensembles.items():
        r, probas[name] = evaluate(name, build_ensemble(e["kind"], e["members"], best), 0.50, e["cv_accuracy"])
        rows.append(r)
        print(f"  {name:24s} CV = {r['CV accuracy']:.4f} | test = {r['TEST accuracy']:.4f}")

    res = pd.DataFrame(rows).sort_values("CV accuracy", ascending=False).reset_index(drop=True)
    res.index += 1
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    base = {"Model": "BASELINE (always 'no CHD')", "CV accuracy": BASELINE_TRAIN, "Train accuracy": BASELINE_TRAIN,
            "TEST accuracy": accuracy_score(y_test, dummy.predict(X_test)), "Test AUC": 0.5, "Precision": 0.0,
            "Recall": 0.0, "F1": 0.0, "Threshold": np.nan, "CHD found": f"0/{int(y_test.sum())}",
            "Correct": f"{int((y_test == 0).sum())}/{len(y_test)}"}
    res = pd.concat([res, pd.DataFrame([base], index=["-"])])
    save_table(res, "step6h_test_results_all_models", index=True)
    show(res, "All models, ordered by CROSS-VALIDATED accuracy (the selection criterion)")

    # Selection rule (uses CV accuracy only): models within 0.0005 of the best CV accuracy are a tie -
    # that is one tenth of the CV standard deviation. Among tied models the simplest one is taken:
    # a single model before an ensemble.
    models_only = res.iloc[:-1]
    tied = models_only[models_only["CV accuracy"] >= models_only["CV accuracy"].max() - 0.0005]
    single = tied[~tied["Model"].str.contains("voting|Voting|Stacking|stacking|Lab 2")]
    chosen = (single if len(single) else tied).iloc[0]
    print(f"\nTied for the best cross-validated accuracy (within 0.0005): {list(tied['Model'])}")
    print(f"Model selected (simplest of the tied models): {chosen['Model']}")
    print(f"   CV accuracy = {chosen['CV accuracy']:.4f} | TEST accuracy = {chosen['TEST accuracy']:.4f}"
          f" | baseline = {BASELINE_TEST:.4f}")
    se = np.sqrt(chosen["TEST accuracy"] * (1 - chosen["TEST accuracy"]) / len(y_test))
    print(f"   With 848 test people, one test accuracy has a standard error of about {se:.4f} "
          f"(+/- {1.96 * se:.3f} at 95%).\n   Differences between models smaller than that are not meaningful.")

    d = res.iloc[:-1].iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 9.5))
    ypos = np.arange(len(d))
    ax.barh(ypos + .2, d["CV accuracy"], height=.4, color="#8DA0CB", label="CV accuracy (train)")
    ax.barh(ypos - .2, d["TEST accuracy"], height=.4, color="#C44E52", label="TEST accuracy")
    ax.set_yticks(ypos); ax.set_yticklabels(d["Model"])
    ax.axvline(BASELINE_TEST, ls="--", c="k", lw=1, label="baseline")
    ax.set_xlim(0.80, 0.875); ax.set_xlabel("accuracy")
    ax.set_title("Step 6h - cross-validated vs. test accuracy of every model")
    ax.legend(loc="lower right")
    savefig("22_all_models_test")

    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name in [n for n in res["Model"] if probas.get(n) is not None][:6]:
        fpr, tpr, _ = roc_curve(y_test, probas[name])
        ax.plot(fpr, tpr, lw=1.5, label=f"{name} (AUC {roc_auc_score(y_test, probas[name]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("ROC curves on the test set (top 6 models by CV)")
    ax.legend(fontsize=8)
    savefig("23_roc_curves")


if __name__ == "__main__":
    main()
