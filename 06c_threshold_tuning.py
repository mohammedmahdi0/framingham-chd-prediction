"""
STEP 6c - DECISION-THRESHOLD TUNING
A classifier outputs a probability; by default "CHD" is predicted when it is >= 0.50.
Because the classes are imbalanced, another cut-off can give higher accuracy.

For every tuned model, out-of-fold probabilities are computed on the TRAINING set
(3 x 5-fold) and the threshold with the highest out-of-fold accuracy is kept.
The test set is not used.

Run:  python 06c_threshold_tuning.py      (needs 06b first, about 1 minute)
"""
from common import *
from models import build_tuned
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score

THRESHOLDS = np.round(np.arange(0.20, 0.801, 0.01), 2)


def oof_accuracy_curve(model, repeats=3):
    """Mean out-of-fold accuracy at every threshold, averaged over `repeats` different 5-fold splits."""
    curves = []
    for r in range(repeats):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 10 + r)
        proba = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        curves.append([accuracy_score(y_train, proba >= t) for t in THRESHOLDS])
    return np.mean(curves, axis=0)


def main():
    header("STEP 6c - DECISION-THRESHOLD TUNING")
    best = load_json("step6b_best_params")
    rows, curves, out = [], {}, {}
    i50 = int(np.where(THRESHOLDS == 0.50)[0][0])
    for name in best:
        c = oof_accuracy_curve(build_tuned(name, best))
        curves[name] = c
        i = int(np.argmax(c))
        # keep 0.50 unless another threshold is better by a real margin (0.001 = about 3 people)
        th = float(THRESHOLDS[i]) if c[i] - c[i50] >= 0.001 else 0.50
        out[name] = {"threshold": th, "oof_acc_at_050": float(c[i50]), "oof_acc_at_best": float(c[i])}
        rows.append({"Model": name, "OOF acc @0.50": c[i50], "best threshold": THRESHOLDS[i],
                     "OOF acc @best": c[i], "gain": c[i] - c[i50], "threshold kept": th})
        print(f"  {name:24s} acc@0.50 = {c[i50]:.4f} | best threshold = {THRESHOLDS[i]:.2f} -> {c[i]:.4f}"
              f" (gain {c[i] - c[i50]:+.4f}) | kept: {th:.2f}")
    save_json(out, "step6c_thresholds")
    res = pd.DataFrame(rows).sort_values("OOF acc @best", ascending=False).reset_index(drop=True)
    save_table(res, "step6c_thresholds")
    show(res, "Threshold tuning (out-of-fold, training set)")

    top = list(res["Model"][:6])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for name in top:
        ax.plot(THRESHOLDS, curves[name], label=name, lw=1.6)
    ax.axvline(0.5, ls=":", c="k", lw=1)
    ax.axhline(BASELINE_TRAIN, ls="--", c="k", lw=1, label="baseline")
    ax.set_xlabel("decision threshold"); ax.set_ylabel("out-of-fold accuracy")
    ax.set_title("Step 6c - accuracy as a function of the decision threshold (top 6 models)")
    ax.legend(fontsize=8)
    savefig("19_threshold_curves")

    print("""
NOTE
A low threshold labels many people as CHD (more false alarms); a high threshold labels almost nobody
as CHD and the accuracy falls back to the baseline. The maximum sits close to 0.50, so the gain from
threshold tuning is small.""")


if __name__ == "__main__":
    main()
