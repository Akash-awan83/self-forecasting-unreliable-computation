"""
MULTI-SEED VERSION -- kya forecast-error result consistent hai?
==================================================================
Same architecture jo pehle bani thi, ab 5 different random seeds pe
chalate hain. Agar result har seed mein same direction dikhaye
(forecast error OOD pe zyada), tabhi hum kehenge ye "real" pattern hai,
sirf fluke nahi.
"""

import numpy as np
from sklearn.datasets import load_digits
from scipy.ndimage import rotate

digits = load_digits()
X_all = digits.images / 16.0
y_all = digits.target

input_dim, hidden_dim, n_classes = 64, 64, 10
n_selfmodel_outputs = hidden_dim
AW = 1.0
n_epochs = 350
adam_lr = 0.01
beta1, beta2, eps_adam = 0.9, 0.999, 1e-8


def relu(x):
    return np.where(x > 0, x, 0.01 * x)


def relu_deriv(x):
    return np.where(x > 0, 1.0, 0.01)


def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def run_one_seed(seed):
    rng = np.random.RandomState(seed)

    idx = rng.permutation(len(X_all))
    n_train = int(0.8 * len(X_all))
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    X_train = X_all[train_idx].reshape(n_train, -1)
    y_train = y_all[train_idx]
    X_test_imgs = X_all[test_idx]
    y_test = y_all[test_idx]
    X_test = X_test_imgs.reshape(len(X_test_imgs), -1)

    X_rot_imgs = np.array([rotate(im, angle=45, reshape=False) for im in X_test_imgs])
    X_rot = np.clip(X_rot_imgs.reshape(len(X_rot_imgs), -1), 0, 1)

    def train(use_sm):
        W1 = rng.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        b1 = np.zeros(hidden_dim)
        Wc = rng.randn(hidden_dim, n_classes) * np.sqrt(2.0 / hidden_dim)
        bc = np.zeros(n_classes)
        Ws = rng.randn(hidden_dim, n_selfmodel_outputs) * np.sqrt(2.0 / hidden_dim)
        bs = np.zeros(n_selfmodel_outputs)

        params = {"W1": W1, "b1": b1, "Wc": Wc, "bc": bc, "Ws": Ws, "bs": bs}
        m = {k: np.zeros_like(v) for k, v in params.items()}
        v = {k: np.zeros_like(v) for k, v in params.items()}

        def step(name, grad, t):
            m[name][:] = beta1 * m[name] + (1 - beta1) * grad
            v[name][:] = beta2 * v[name] + (1 - beta2) * (grad ** 2)
            mh = m[name] / (1 - beta1 ** t)
            vh = v[name] / (1 - beta2 ** t)
            params[name] -= adam_lr * mh / (np.sqrt(vh) + eps_adam)

        n = X_train.shape[0]
        y_oh = np.zeros((n, n_classes))
        y_oh[np.arange(n), y_train] = 1

        for epoch in range(n_epochs):
            t = epoch + 1
            caw = AW * min(1.0, epoch / 100.0) if use_sm else 0.0
            W1, b1, Wc, bc, Ws, bs = params["W1"], params["b1"], params["Wc"], params["bc"], params["Ws"], params["bs"]

            z1 = X_train @ W1 + b1
            a = relu(z1)
            logits = a @ Wc + bc
            probs = softmax(logits)
            a_hat = a @ Ws + bs

            dlogits = (probs - y_oh) / n
            dWc = a.T @ dlogits
            dbc = dlogits.sum(0)
            da = dlogits @ Wc.T

            if use_sm:
                dah = 2 * (a_hat - a) / n * caw
                dWs = a.T @ dah
                dbs = dah.sum(0)
                da = da + dah @ Ws.T - dah
            else:
                dWs, dbs = np.zeros_like(Ws), np.zeros_like(bs)

            dz1 = da * relu_deriv(z1)
            dW1 = X_train.T @ dz1
            db1 = dz1.sum(0)

            step("W1", dW1, t); step("b1", db1, t)
            step("Wc", dWc, t); step("bc", dbc, t)
            if use_sm:
                step("Ws", dWs, t); step("bs", dbs, t)

        return params

    base_p = train(False)
    sm_p = train(True)

    def evaluate(p, X, has_sm):
        z1 = X @ p["W1"] + p["b1"]
        a = relu(z1)
        probs = softmax(a @ p["Wc"] + p["bc"])
        pred = probs.argmax(1)
        conf = probs.max(1)
        ferr = np.linalg.norm(a @ p["Ws"] + p["bs"] - a, axis=1) if has_sm else None
        return pred, conf, ferr

    pred_n, conf_n, ferr_n = evaluate(sm_p, X_test, True)
    pred_r, conf_r, ferr_r = evaluate(sm_p, X_rot, True)
    bpred_n, bconf_n, _ = evaluate(base_p, X_test, False)
    bpred_r, bconf_r, _ = evaluate(base_p, X_rot, False)

    return {
        "seed": seed,
        "sm_acc_normal": (pred_n == y_test).mean(),
        "sm_acc_rotated": (pred_r == y_test).mean(),
        "base_acc_normal": (bpred_n == y_test).mean(),
        "base_acc_rotated": (bpred_r == y_test).mean(),
        "ferr_normal": ferr_n.mean(),
        "ferr_rotated": ferr_r.mean(),
        "ferr_ratio": ferr_r.mean() / (ferr_n.mean() + 1e-9),
        "base_w_std": base_p["Wc"].std(),
        "sm_w_std": sm_p["Wc"].std(),
    }


print("Running 5 seeds... (thoda time lagega)")
results = []
for seed in range(5):
    r = run_one_seed(seed)
    results.append(r)
    print(f"Seed {seed}: ferr_ratio={r['ferr_ratio']:.2f}x | "
          f"normal_ferr={r['ferr_normal']:.4f} rotated_ferr={r['ferr_rotated']:.4f} | "
          f"sm_acc(normal/rot)={r['sm_acc_normal']:.2f}/{r['sm_acc_rotated']:.2f}")

print("\n" + "=" * 70)
print("SUMMARY ACROSS 5 SEEDS")
print("=" * 70)
ratios = [r["ferr_ratio"] for r in results]
print(f"Forecast error ratio (rotated/normal): mean={np.mean(ratios):.2f}x, "
      f"std={np.std(ratios):.2f}, min={np.min(ratios):.2f}, max={np.max(ratios):.2f}")
print(f"(Agar sab seeds mein ratio >> 1, to result CONSISTENT/robust hai)")

base_std = [r["base_w_std"] for r in results]
sm_std = [r["sm_w_std"] for r in results]
print(f"\nBaseline weight-std: mean={np.mean(base_std):.4f}")
print(f"Self-model weight-std: mean={np.mean(sm_std):.4f}")
print(f"(Paper's prediction: self-model < baseline. Kya ye consistently hua?)")
