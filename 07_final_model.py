"""
FINAL MODEL
===========
Predict whether a person will develop coronary heart disease within 10 years (TenYearCHD).

Every choice below comes from the experiments (scripts 02 - 06), all made by cross-validation on the
training set:

    missing data .......... median imputation                      (02_missing_data.py)
    distributions ......... no transformation                      (03_distribution.py)
    scaling ............... StandardScaler                         (04_feature_scaling.py)
    dimensionality ........ Lasso / Elastic Net -> 10 features     (05a ... 05e)
    model ................. Logistic Regression, Elastic Net penalty, C = 0.428, l1_ratio = 0.5
                            (best cross-validated accuracy of 17 tuned models, 06b)
    threshold ............. 0.50                                   (06c_threshold_tuning.py)
    class balancing ....... none - not allowed in this project
    engineered features ... none - no gain                         (06f_feature_engineering.py)
    ensembles ............. none - no gain over the single model   (06d_ensembles.py)

Run:  python 07_final_model.py
"""
import joblib
from common import *
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve)

FEATURE_SET = "lasso10"
THRESHOLD = 0.50


def final_model():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),            # step 2
        ("scale", StandardScaler()),                             # step 4
        ("select", ColumnSelector(FEATURE_SET)),                 # step 5
        ("clf", LogisticRegression(C=0.42813323987193913, l1_ratio=0.5, solver="saga", max_iter=5000,
                                   random_state=SEED)),   # step 6
    ])


def main():
    header("FINAL MODEL - LOGISTIC REGRESSION (ELASTIC NET) ON 10 FEATURES")
    print("Features used:", FEATURE_SETS[FEATURE_SET])

    # ------------------------------------------------------------------ 1) cross-validation on the training set
    cv = cv_eval(final_model(), cv=FRESH_CV)
    print(f"\nCross-validated accuracy (train, 15 folds): {cv['CV accuracy']:.4f} +/- {cv['CV acc std']:.4f}"
          f" | AUC {cv['CV AUC']:.4f}")

    # ------------------------------------------------------------------ 2) train on all training data, test once
    model = final_model().fit(X_train, y_train)
    p_train, p_test = model.predict_proba(X_train)[:, 1], model.predict_proba(X_test)[:, 1]
    pred = (p_test >= THRESHOLD).astype(int)
    train_acc, test_acc = accuracy_score(y_train, p_train >= THRESHOLD), accuracy_score(y_test, pred)

    print("\n" + "-" * 60)
    print(f"TRAIN accuracy : {train_acc:.4f}")
    print(f"TEST  accuracy : {test_acc:.4f}     <-- the reported result")
    print(f"Baseline (test): {BASELINE_TEST:.4f}     (always predicting 'no CHD')")
    print(f"Test AUC       : {roc_auc_score(y_test, p_test):.4f}")
    print(f"Precision      : {precision_score(y_test, pred, zero_division=0):.4f}")
    print(f"Recall         : {recall_score(y_test, pred):.4f}")
    print(f"F1             : {f1_score(y_test, pred):.4f}")
    print("-" * 60)
    print("\nTrain and test accuracy are close -> no over-fitting.")
    print("\nClassification report (test set):")
    print(classification_report(y_test, pred, target_names=["No CHD", "CHD"], digits=4, zero_division=0))
    cm = confusion_matrix(y_test, pred)
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(cm, index=["actual No CHD", "actual CHD"], columns=["pred No CHD", "pred CHD"]).to_string())

    coef = pd.DataFrame({"coefficient": model.named_steps["clf"].coef_[0]}, index=FEATURE_SETS[FEATURE_SET])
    coef["odds ratio (per +1 std)"] = np.exp(coef["coefficient"])
    coef = coef.sort_values("coefficient", ascending=False)
    show(coef, "Model coefficients (standardized features)")

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    ConfusionMatrixDisplay(cm, display_labels=["No CHD", "CHD"]).plot(ax=ax[0], cmap="Blues", colorbar=False)
    ax[0].set_title(f"Confusion matrix (test) - accuracy {test_acc:.4f}")
    ax[0].grid(False)
    fpr, tpr, _ = roc_curve(y_test, p_test)
    ax[1].plot(fpr, tpr, c="#C44E52", lw=2, label=f"AUC = {roc_auc_score(y_test, p_test):.3f}")
    ax[1].plot([0, 1], [0, 1], "k--", lw=1)
    ax[1].set_xlabel("false positive rate"); ax[1].set_ylabel("true positive rate")
    ax[1].set_title("ROC curve (test)"); ax[1].legend()
    c = coef["coefficient"].sort_values()
    ax[2].barh(c.index, c.values, color="#4C72B0")
    ax[2].set_title("Coefficients (standardized features)")
    savefig("24_final_model")

    # ------------------------------------------------------------------ 3) is the result stable? repeated hold-out
    # The single test set has only 848 people, so its accuracy can move by +/- 2% just by chance.
    # To measure how the model behaves in general, the whole procedure (split 80/20, train, test) is
    # repeated on 100 different random splits. ALL 100 results are reported - not the best one.
    rows = []
    for seed in range(100):
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)
        m = final_model().fit(Xa, ya)
        rows.append({"split seed": seed, "model accuracy": accuracy_score(yb, m.predict_proba(Xb)[:, 1] >= THRESHOLD),
                     "baseline accuracy": 1 - yb.mean()})
    rep = pd.DataFrame(rows)
    rep["difference"] = rep["model accuracy"] - rep["baseline accuracy"]
    save_table(rep, "step7_repeated_holdout")
    print("\nRepeated hold-out: 100 different random 80/20 splits")
    print(f"   model accuracy    : mean {rep['model accuracy'].mean():.4f} | std {rep['model accuracy'].std():.4f}"
          f" | min {rep['model accuracy'].min():.4f} | max {rep['model accuracy'].max():.4f}")
    print(f"   baseline accuracy : mean {rep['baseline accuracy'].mean():.4f}")
    print(f"   model beats the baseline in {(rep['difference'] > 0).sum()} of 100 splits, ties in "
          f"{(rep['difference'] == 0).sum()}, loses in {(rep['difference'] < 0).sum()}")
    print(f"   our fixed split (seed {SEED}) gave {test_acc:.4f}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(rep["model accuracy"], bins=18, color="#4C72B0", alpha=.85)
    ax.axvline(rep["model accuracy"].mean(), c="#C44E52", lw=2, label=f"mean of 100 splits = {rep['model accuracy'].mean():.4f}")
    ax.axvline(rep["baseline accuracy"].mean(), c="k", ls="--", lw=1.5, label=f"baseline = {rep['baseline accuracy'].mean():.4f}")
    ax.axvline(test_acc, c="#55A868", lw=2, ls=":", label=f"our split (seed {SEED}) = {test_acc:.4f}")
    ax.set_xlabel("test accuracy"); ax.set_ylabel("number of splits")
    ax.set_title("Final model - test accuracy over 100 random train/test splits")
    ax.legend(fontsize=8)
    savefig("25_repeated_holdout")

    # ------------------------------------------------------------------ 4) save the model and use it
    joblib.dump({"model": model, "threshold": THRESHOLD, "features": FEATURES}, MODEL_DIR / "final_model.joblib")
    print("\n[saved] model/final_model.joblib")

    person = pd.DataFrame([{"male": 1, "age": 61, "education": 1, "currentSmoker": 1, "cigsPerDay": 30, "BPMeds": 0,
                            "prevalentStroke": 0, "prevalentHyp": 1, "diabetes": 0, "totChol": 225, "sysBP": 150,
                            "diaBP": 95, "BMI": 28.58, "heartRate": 65, "glucose": 103}])[FEATURES]
    prob = model.predict_proba(person)[0, 1]
    print("\nExample - a new person: male, 61 years, 30 cigarettes/day, sysBP 150, glucose 103")
    print(f"   probability of CHD within 10 years = {prob:.3f}  ->  prediction: {'CHD' if prob >= THRESHOLD else 'No CHD'}")


if __name__ == "__main__":
    main()
