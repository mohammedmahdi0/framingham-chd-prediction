"""
run_all.py - run every experiment script in order (no plot windows) and save the console output of
each one to logs/<script>.txt. Takes about 18 minutes.

Run:  python run_all.py
Then: python make_screenshots.py     (turns logs/ into images for the report)
"""
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
SCRIPTS = [
    "01_visualization", "02_missing_data", "03_distribution", "04_feature_scaling", "04b_onehot_encoding",
    "05a_regularization_ridge_lasso_elasticnet", "05b_pca", "05c_stepwise_aic_bic",
    "05d_other_feature_selection", "05e_compare_reduction_methods",
    "06a_models_default", "06b_hyperparameter_tuning", "06c_threshold_tuning", "06d_ensembles",
    "06e_tree_pruning", "06f_feature_engineering", "06g_optimizers_sgd_newton",
    "06h_test_evaluation_all_models", "07_final_model",
]


def main():
    (BASE / "logs").mkdir(exist_ok=True)
    env = dict(os.environ, NO_SHOW="1", MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    failed = []
    for name in SCRIPTS:
        t = time.time()
        with open(BASE / "logs" / f"{name}.txt", "w", encoding="utf-8") as f:
            code = subprocess.run([sys.executable, f"{name}.py"], cwd=BASE, env=env, stdout=f,
                                  stderr=subprocess.STDOUT).returncode
        print(f"{'OK  ' if code == 0 else 'FAIL'} {name:45s} {time.time() - t:5.0f}s")
        if code != 0:
            failed.append(name)
    print("\nAll scripts finished successfully." if not failed else f"\nFAILED: {failed}")


if __name__ == "__main__":
    main()
