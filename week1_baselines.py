"""
Week 1: Does self-model forecast error track ERRORS, or just NOVELTY?

Setup (reconstruction, pure NumPy):
  - sklearn digits (8x8), scaled to [0,1], 70/30 stratified split
  - MLP: 64 -> 64 (ReLU) -> 64 (ReLU) -> [10 logits | 64 self-model outputs]
    The self-model head (from h2) predicts the network's own first hidden layer h1.
    Loss = CE + lam * MSE(self_pred, h1)   (target NOT detached, as in the self-modeling paper)
  - Distribution shift: rotation by 15/30/45/60/90 degrees

Uncertainty scores compared (higher = "less trustworthy"):
  same net   : 1-MSP, entropy, energy, FORECAST ERROR (ours)
  plain nets : 1-MSP (single plain net), deep ensemble (5 nets) entropy, MC dropout entropy
  novelty    : autoencoder reconstruction error, Mahalanobis distance (penultimate features)

Tests:
  A. OOD detection AUROC      : clean test vs rotated test            (novelty)
  B. Error detection AUROC    : correct vs wrong, WITHIN each angle    (errors, novelty held ~fixed)
  C. Selective prediction     : AURC on pooled clean+all-angle data    (does abstaining by score reduce risk?)
"""
import numpy as np
from scipy.ndimage import rotate
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import json, sys

ANGLES = [15, 30, 45, 60, 90]
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 300
LAM = 1.0


# ---------------------------------------------------------------- data
def load_data(seed):
    d = load_digits()
    X = d.data.astype(np.float64) / 16.0
    y = d.target
    return train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)


def rotate_set(X, angle):
    imgs = X.reshape(-1, 8, 8)
    out = np.stack([rotate(im, angle, reshape=False, order=1, mode="constant") for im in imgs])
    return np.clip(out, 0, 1).reshape(-1, 64)


# ---------------------------------------------------------------- utils
def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def logsumexp(z):
    m = z.max(1, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(1, keepdims=True))).ravel()


def entropy(p):
    return -(p * np.log(p + 1e-12)).sum(1)


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999):
        self.p, self.lr, self.b1, self.b2, self.t = params, lr, b1, b2, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, g):
        self.t += 1
        for k in self.p:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


# ---------------------------------------------------------------- MLP (+ optional self-model head, dropout)
class MLP:
    def __init__(self, seed, H=64, self_model=False, lam=LAM, dropout=0.0):
        r = np.random.default_rng(seed)
        he = lambda i, o: r.normal(0, np.sqrt(2 / i), (i, o))
        self.P = {"W1": he(64, H), "b1": np.zeros(H),
                  "W2": he(H, H), "b2": np.zeros(H),
                  "W3": he(H, 10) * 0.5, "b3": np.zeros(10)}
        if self_model:
            self.P["W4"] = he(H, H) * 0.5
            self.P["b4"] = np.zeros(H)
        self.sm, self.lam, self.dp, self.rng = self_model, lam, dropout, r

    def forward(self, X, train=False, mc=False):
        P = self.P
        z1 = X @ P["W1"] + P["b1"]; h1 = np.maximum(z1, 0)
        m1 = m2 = None
        h1d = h1
        if self.dp > 0 and (train or mc):
            m1 = (self.rng.random(h1.shape) > self.dp) / (1 - self.dp); h1d = h1 * m1
        z2 = h1d @ P["W2"] + P["b2"]; h2 = np.maximum(z2, 0)
        h2d = h2
        if self.dp > 0 and (train or mc):
            m2 = (self.rng.random(h2.shape) > self.dp) / (1 - self.dp); h2d = h2 * m2
        logits = h2d @ P["W3"] + P["b3"]
        s = h2d @ P["W4"] + P["b4"] if self.sm else None
        return dict(X=X, z1=z1, h1=h1, h1d=h1d, m1=m1, z2=z2, h2=h2, h2d=h2d, m2=m2, logits=logits, s=s)

    def backward(self, c, y):
        P, B = self.P, len(y)
        g = {}
        p = softmax(c["logits"]); p[np.arange(B), y] -= 1
        dlog = p / B
        g["W3"] = c["h2d"].T @ dlog; g["b3"] = dlog.sum(0)
        dh2d = dlog @ P["W3"].T
        dh1_direct = 0
        if self.sm:
            D = c["h1"].shape[1]
            ds = self.lam * 2 * (c["s"] - c["h1"]) / (B * D)
            g["W4"] = c["h2d"].T @ ds; g["b4"] = ds.sum(0)
            dh2d = dh2d + ds @ P["W4"].T
            dh1_direct = -ds                       # target not detached
        dh2 = dh2d * c["m2"] if c["m2"] is not None else dh2d
        dz2 = dh2 * (c["z2"] > 0)
        g["W2"] = c["h1d"].T @ dz2; g["b2"] = dz2.sum(0)
        dh1d = dz2 @ P["W2"].T
        dh1 = (dh1d * c["m1"] if c["m1"] is not None else dh1d) + dh1_direct
        dz1 = dh1 * (c["z1"] > 0)
        g["W1"] = c["X"].T @ dz1; g["b1"] = dz1.sum(0)
        return g

    def fit(self, X, y, epochs=EPOCHS, bs=64):
        opt = Adam(self.P)
        for _ in range(epochs):
            idx = self.rng.permutation(len(X))
            for i in range(0, len(X), bs):
                b = idx[i:i + bs]
                c = self.forward(X[b], train=True)
                opt.step(self.backward(c, y[b]))
        return self


# ---------------------------------------------------------------- autoencoder baseline (novelty only)
class AE:
    def __init__(self, seed, H=32):
        r = np.random.default_rng(seed)
        self.P = {"W1": r.normal(0, np.sqrt(2 / 64), (64, H)), "b1": np.zeros(H),
                  "W2": r.normal(0, np.sqrt(1 / H), (H, 64)), "b2": np.zeros(64)}
        self.rng = r

    def recon(self, X):
        z = X @ self.P["W1"] + self.P["b1"]; h = np.maximum(z, 0)
        return z, h, h @ self.P["W2"] + self.P["b2"]

    def fit(self, X, epochs=EPOCHS, bs=64):
        opt = Adam(self.P)
        for _ in range(epochs):
            idx = self.rng.permutation(len(X))
            for i in range(0, len(X), bs):
                xb = X[idx[i:i + bs]]
                z, h, o = self.recon(xb)
                do = 2 * (o - xb) / xb.size
                g = {"W2": h.T @ do, "b2": do.sum(0)}
                dz = (do @ self.P["W2"].T) * (z > 0)
                g["W1"] = xb.T @ dz; g["b1"] = dz.sum(0)
                opt.step(g)
        return self

    def score(self, X):
        return ((self.recon(X)[2] - X) ** 2).mean(1)


# ---------------------------------------------------------------- Mahalanobis (Lee et al. 2018), on penultimate features
class Mahalanobis:
    def __init__(self, feats, y):
        self.mu = np.stack([feats[y == k].mean(0) for k in range(10)])
        cen = feats - self.mu[y]
        cov = cen.T @ cen / len(feats) + 1e-3 * np.eye(feats.shape[1])
        self.prec = np.linalg.inv(cov)

    def score(self, f):
        d = f[:, None, :] - self.mu[None]
        return np.einsum("nkd,de,nke->nk", d, self.prec, d).min(1)


# ---------------------------------------------------------------- metrics
def aurc(score, correct):
    """Area under risk-coverage curve. Accept most-confident (lowest score) first. Lower = better."""
    order = np.argsort(score, kind="stable")
    err = (~correct[order]).astype(float)
    risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    return risk.mean()


def safe_auc(labels, score):
    return roc_auc_score(labels, score) if 0 < labels.sum() < len(labels) else np.nan


# ---------------------------------------------------------------- one seed
def run_seed(seed):
    Xtr, Xte, ytr, yte = load_data(seed)
    sets = {0: Xte, **{a: rotate_set(Xte, a) for a in ANGLES}}

    selfnet = MLP(seed, self_model=True).fit(Xtr, ytr)
    plains = [MLP(seed * 100 + k).fit(Xtr, ytr) for k in range(5)]
    dropnet = MLP(seed * 100 + 99, dropout=0.2).fit(Xtr, ytr)
    ae = AE(seed).fit(Xtr)
    maha = Mahalanobis(selfnet.forward(Xtr)["h2"], ytr)

    out = {}
    for a, X in sets.items():
        c = selfnet.forward(X)
        p = softmax(c["logits"])
        pred = p.argmax(1)
        correct = pred == yte

        ens_p = np.mean([softmax(m.forward(X)["logits"]) for m in plains], 0)
        mc_p = np.mean([softmax(dropnet.forward(X, mc=True)["logits"]) for _ in range(30)], 0)

        scores = {
            "forecast_err (ours)": ((c["s"] - c["h1"]) ** 2).mean(1),
            "1-MSP (same net)": 1 - p.max(1),
            "entropy (same net)": entropy(p),
            "energy (same net)": -logsumexp(c["logits"]),
            "1-MSP (plain net)": 1 - softmax(plains[0].forward(X)["logits"]).max(1),
            "ensemble entropy": entropy(ens_p),
            "MC dropout entropy": entropy(mc_p),
            "AE recon err": ae.score(X),
            "Mahalanobis": maha.score(c["h2"]),
        }
        out[a] = dict(scores=scores, correct=correct,
                      acc=correct.mean(),
                      acc_plain=(softmax(plains[0].forward(X)["logits"]).argmax(1) == yte).mean(),
                      acc_ens=(ens_p.argmax(1) == yte).mean())
    return out


def main():
    results = {}
    for s in SEEDS:
        print(f"seed {s} ...", flush=True)
        results[s] = run_seed(s)
    methods = list(results[SEEDS[0]][0]["scores"].keys())

    def agg(fn):
        vals = np.array([fn(results[s]) for s in SEEDS])
        return np.nanmean(vals), np.nanstd(vals)

    report = {"accuracy": {}, "forecast_ratio": {}, "A_ood_auroc": {}, "B_error_auroc_within_angle": {},
              "C_aurc_pooled": {}, "C_error_auroc_pooled": {}}

    print("\n=== Accuracy (self-model net / plain net / ensemble) ===")
    for a in [0] + ANGLES:
        m1, s1 = agg(lambda r: r[a]["acc"]); m2, _ = agg(lambda r: r[a]["acc_plain"]); m3, _ = agg(lambda r: r[a]["acc_ens"])
        report["accuracy"][a] = [m1, s1, m2, m3]
        print(f"  angle {a:>3}: self {m1:.3f}±{s1:.3f} | plain {m2:.3f} | ens {m3:.3f}")

    print("\n=== Forecast-error ratio (mean shifted / mean clean) ===")
    for a in ANGLES:
        m, sd = agg(lambda r: r[a]["scores"]["forecast_err (ours)"].mean() / r[0]["scores"]["forecast_err (ours)"].mean())
        report["forecast_ratio"][a] = [m, sd]
        print(f"  angle {a:>3}: {m:.2f}x ± {sd:.2f}")

    print("\n=== A. OOD detection AUROC (clean vs rotated) — NOVELTY ===")
    hdr = "  " + f"{'method':<22}" + "".join(f"{a:>12}" for a in ANGLES)
    print(hdr)
    for meth in methods:
        row = []
        for a in ANGLES:
            def f(r, a=a):
                sc = np.concatenate([r[0]["scores"][meth], r[a]["scores"][meth]])
                lab = np.r_[np.zeros(len(r[0]["correct"])), np.ones(len(r[a]["correct"]))]
                return safe_auc(lab, sc)
            row.append(agg(f))
        report["A_ood_auroc"][meth] = {a: v for a, v in zip(ANGLES, row)}
        print("  " + f"{meth:<22}" + "".join(f"{m:>7.3f}±{s:.2f}" for m, s in row))

    print("\n=== B. Error detection AUROC, WITHIN each angle (wrong vs correct) — ERRORS ===")
    print(hdr)
    for meth in methods:
        row = []
        for a in ANGLES:
            row.append(agg(lambda r, a=a: safe_auc((~r[a]["correct"]).astype(int), r[a]["scores"][meth])))
        report["B_error_auroc_within_angle"][meth] = {a: v for a, v in zip(ANGLES, row)}
        print("  " + f"{meth:<22}" + "".join(f"{m:>7.3f}±{s:.2f}" for m, s in row))

    print("\n=== C. Selective prediction on POOLED clean + all angles (self-model net's predictions) ===")
    print(f"  {'method':<22}{'AURC (lower better)':>22}{'err-AUROC':>14}")
    def pooled(r, meth):
        sc = np.concatenate([r[a]["scores"][meth] for a in [0] + ANGLES])
        cor = np.concatenate([r[a]["correct"] for a in [0] + ANGLES])
        return sc, cor
    for meth in methods:
        m, sd = agg(lambda r: aurc(*pooled(r, meth)))
        m2, sd2 = agg(lambda r: safe_auc((~pooled(r, meth)[1]).astype(int), pooled(r, meth)[0]))
        report["C_aurc_pooled"][meth] = [m, sd]; report["C_error_auroc_pooled"][meth] = [m2, sd2]
        print(f"  {meth:<22}{m:>14.4f}±{sd:.4f}{m2:>10.3f}±{sd2:.2f}")
    # reference lines
    m_or, _ = agg(lambda r: aurc((~pooled(r, methods[0])[1]).astype(float), pooled(r, methods[0])[1]))
    m_rand, _ = agg(lambda r: 1 - pooled(r, methods[0])[1].mean())
    print(f"  {'[oracle]':<22}{m_or:>14.4f}\n  {'[random]':<22}{m_rand:>14.4f}")
    report["C_reference"] = {"oracle": m_or, "random": m_rand}

    # risk-coverage curves for plotting (seed-averaged)
    covs = np.linspace(0.05, 1, 20)
    curves = {}
    for meth in methods + ["[oracle]"]:
        rs = []
        for s in SEEDS:
            if meth == "[oracle]":
                _, cor = pooled(results[s], methods[0]); sc = (~cor).astype(float)
            else:
                sc, cor = pooled(results[s], meth)
            order = np.argsort(sc, kind="stable"); err = (~cor[order]).astype(float)
            rs.append([err[: max(1, int(c * len(err)))].mean() for c in covs])
        curves[meth] = np.mean(rs, 0).tolist()
    report["risk_coverage"] = {"coverage": covs.tolist(), "curves": curves}

    with open("week1_results.json", "w") as fh:
        json.dump(report, fh, indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))


if __name__ == "__main__":
    main()
