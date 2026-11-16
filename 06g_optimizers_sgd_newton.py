"""
STEP 6g - HOW THE MODEL IS TRAINED: GRADIENT DESCENT, SGD AND NEWTON'S METHOD
(methods from "Lecture 4 - Searching Methodologies" and "Lab 2 - SGD / Newton's Method")

Training a logistic regression means searching for the weights that minimize a cost function
(the log-loss). The lecture presents several ways of searching for that minimum:

    Gradient Descent (GD)  : uses the whole data set for every step
    Stochastic GD (SGD)    : uses a small batch for every step - cheap steps, but many of them
    Newton's Method        : also uses the curvature (Hessian) - very few steps, each one heavier

Part A implements the three methods by hand (as in the lab, but for logistic regression instead of
linear regression) and shows how fast each one reaches the minimum.
Part B trains the same model with scikit-learn's solvers (SGD, Newton, quasi-Newton, SAGA) and
compares accuracy, number of iterations and time.

The test set is not used. No class balancing is used.

Run:  python 06g_optimizers_sgd_newton.py      (about 1 minute)
"""
import time
from common import *
from sklearn.linear_model import SGDClassifier

C = 1.0     # same regularization for every method, so all of them solve exactly the same problem


# ------------------------------------------------------------------ Part A: hand-written optimizers
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(w, X, y):
    """Mean log-loss + L2 penalty (the intercept w[0] is not penalized)."""
    p = np.clip(sigmoid(X @ w), 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) + (w[1:] @ w[1:]) / (2 * C * len(y))


def gradient(w, X, y, n_total):
    g = X.T @ (sigmoid(X @ w) - y) / len(y)
    g[1:] += w[1:] / (C * n_total)
    return g


def hessian(w, X, y):
    p = sigmoid(X @ w)
    H = (X * (p * (1 - p))[:, None]).T @ X / len(y)
    H[1:, 1:] += np.eye(X.shape[1] - 1) / (C * len(y))
    return H


def gradient_descent(X, y, lr=0.5, steps=300):
    w, hist = np.zeros(X.shape[1]), []
    for _ in range(steps):
        hist.append(cost(w, X, y))
        w -= lr * gradient(w, X, y, len(y))
    return w, hist + [cost(w, X, y)]


def sgd(X, y, lr=0.05, epochs=300, batch_size=32):
    rng = np.random.default_rng(SEED)
    w, hist = np.zeros(X.shape[1]), []
    for _ in range(epochs):
        hist.append(cost(w, X, y))
        idx = rng.permutation(len(y))
        for i in range(0, len(y), batch_size):
            b = idx[i:i + batch_size]
            w -= lr * gradient(w, X[b], y[b], len(y))
    return w, hist + [cost(w, X, y)]


def newtons_method(X, y, max_iterations=25, tolerance=1e-8):
    w, hist = np.zeros(X.shape[1]), []
    for _ in range(max_iterations):
        hist.append(cost(w, X, y))
        g = gradient(w, X, y, len(y))
        if np.linalg.norm(g) < tolerance:
            break
        w -= np.linalg.solve(hessian(w, X, y), g)
    return w, hist


def main():
    header("STEP 6g - GRADIENT DESCENT, SGD AND NEWTON'S METHOD")
    Xp = PRE().fit_transform(X_train)
    Xb = np.c_[np.ones(len(Xp)), Xp.values]
    yv = y_train.values.astype(float)

    ref = LR(C=C, tol=1e-10).fit(Xp, y_train)             # scikit-learn's answer, used as the reference
    w_ref = np.r_[ref.intercept_, ref.coef_[0]]
    best_cost = cost(w_ref, Xb, yv)

    rows, hists = [], {}
    for name, fn in {"Gradient Descent (GD)": gradient_descent, "Stochastic GD (SGD)": sgd,
                     "Newton's Method": newtons_method}.items():
        t = time.time()
        w, hist = fn(Xb, yv)
        sec = time.time() - t
        hists[name] = np.array(hist)
        reach = np.where(hists[name] - best_cost < 1e-4)[0]
        rows.append({"Method (hand-written)": name, "final cost": hist[-1], "passes over the data": len(hist) - 1,
                     "passes to get within 0.0001 of the minimum": int(reach[0]) if len(reach) else np.nan,
                     "max |weight - reference weight|": np.abs(w - w_ref).max(),
                     "training accuracy": np.mean((sigmoid(Xb @ w) >= .5) == yv), "seconds": sec})
    part_a = pd.DataFrame(rows)
    print(f"Minimum of the cost function (scikit-learn reference): {best_cost:.6f}")
    show(part_a, "Part A - the three optimizers written by hand, on the training set")
    save_table(part_a, "step6g_handwritten_optimizers")

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
    for name, c in zip(hists, ["#4C72B0", "#55A868", "#C44E52"]):
        ax[0].plot(hists[name][:60], label=name, c=c, lw=1.8, marker="o" if "Newton" in name else None, ms=4)
        ax[1].semilogy(np.maximum(hists[name] - best_cost, 1e-12), label=name, c=c, lw=1.8)
    ax[0].axhline(best_cost, ls="--", c="k", lw=1, label="minimum")
    ax[0].set_xlabel("passes over the data"); ax[0].set_ylabel("cost (log-loss)")
    ax[0].set_title("Cost during training (first 60 passes)"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("passes over the data"); ax[1].set_ylabel("distance from the minimum (log scale)")
    ax[1].set_title("How fast each method reaches the minimum"); ax[1].legend(fontsize=8)
    savefig("21b_optimizers")

    # ------------------------------------------------------------------ Part B: scikit-learn solvers
    n = len(X_train) * 0.8      # size of a training fold; SGDClassifier's alpha plays the role of 1 / (C * n)
    solvers = {
        "lbfgs (quasi-Newton, default)": LR(C=C, solver="lbfgs"),
        "newton-cg (Newton)": LR(C=C, solver="newton-cg"),
        "newton-cholesky (Newton)": LR(C=C, solver="newton-cholesky"),
        "sag (stochastic average gradient)": LR(C=C, solver="sag"),
        "saga (stochastic)": LR(C=C, solver="saga"),
        "SGDClassifier (SGD, log-loss)": SGDClassifier(loss="log_loss", alpha=1 / (C * n), max_iter=1000, tol=1e-3,
                                                      random_state=SEED),
    }
    rows = []
    for name, est in solvers.items():
        r = cv_eval(Pipeline([("pre", PRE()), ("clf", clone(est))]))
        t = time.time()
        m = clone(est).fit(Xp, y_train)
        sec = time.time() - t
        w = np.r_[m.intercept_, m.coef_[0]]
        rows.append({"Solver (scikit-learn)": name, **r, "iterations": int(np.max(m.n_iter_)),
                     "max |weight - reference weight|": np.abs(w - w_ref).max(), "fit seconds": sec})
        print(f"  {name:36s} CV acc = {r['CV accuracy']:.4f}  iterations = {rows[-1]['iterations']}")
    part_b = pd.DataFrame(rows)
    save_table(part_b, "step6g_sklearn_solvers")
    show(part_b, "Part B - the same logistic regression trained by different scikit-learn solvers")

    a = part_a.set_index("Method (hand-written)")["passes to get within 0.0001 of the minimum"]
    exact = part_b.iloc[:-1]
    print(f"""
CONCLUSIONS
- Passes over the data needed to get within 0.0001 of the minimum cost:
      Newton's Method {a["Newton's Method"]:.0f}  |  SGD {a['Stochastic GD (SGD)']:.0f}  |  Gradient Descent {a['Gradient Descent (GD)']:.0f}
  This is the lecture's point: Newton uses the curvature, so it needs far fewer steps.
- SGD takes cheap, noisy steps: it gets near the minimum quickly but keeps wobbling around it
  (its final weights are the furthest from the reference).
- All scikit-learn solvers except SGDClassifier end at practically the same weights and the same accuracy
  ({exact['CV accuracy'].min():.4f} to {exact['CV accuracy'].max():.4f}). The optimizer changes HOW FAST the model is trained,
  not how good the trained model is.
- SGDClassifier gives {part_b['CV accuracy'].iloc[-1]:.4f}: its noisy steps stop near the minimum, not at it.""")


if __name__ == "__main__":
    main()
