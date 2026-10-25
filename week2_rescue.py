"""
Week 2 (rescue): Is there ANY self-model variant whose forecast error carries
error-specific information beyond MSP + autoencoder (novelty)?

Variants (all: 64 -> 64 -> 64 -> 10 MLP, NumPy, Adam, 300 epochs):
  A  h2 -> h1, target NOT detached, lam=1     (week-1 version)
  A' h2 -> h1, target NOT detached, lam=0.1
  B  h2 -> h1, target detached               (pure "observer", no simplification pressure)
  C  h1 -> h2, detached                      (forward self-forecast: predict own NEXT layer)
  D  h1 -> logits, detached                  (forecast own OUTPUT; score = MSE and KL(actual||forecast))

Mild shifts (where the model is partly right, so "which ones are wrong" is a real question):
  rotation 5/10/15/20/30 deg, gaussian noise sd 0.1/0.2/0.3, blur sigma 0.5/0.8/1.0

Main metric: within-shift error-detection AUROC (wrong vs correct) on each net's OWN predictions.
Key number: GAIN = AUROC(MSP+AE+forecast) - AUROC(MSP+AE). If ~0 or negative -> no rescue.
"""
import numpy as np, json
from scipy.ndimage import rotate, gaussian_filter
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from week1_baselines import load_data, softmax, Adam, AE, EPOCHS

SEEDS = [0, 1, 2, 3, 4]
SHIFTS = [("rot", 5), ("rot", 10), ("rot", 15), ("rot", 20), ("rot", 30),
          ("noise", 0.1), ("noise", 0.2), ("noise", 0.3),
          ("blur", 0.5), ("blur", 0.8), ("blur", 1.0)]
VARIANTS = {
    "A  h2->h1 nodetach lam1":   dict(src="h2", tgt="h1", detach=False, lam=1.0),
    "A' h2->h1 nodetach lam.1":  dict(src="h2", tgt="h1", detach=False, lam=0.1),
    "B  h2->h1 detach":          dict(src="h2", tgt="h1", detach=True, lam=1.0),
    "C  h1->h2 detach":          dict(src="h1", tgt="h2", detach=True, lam=1.0),
    "D  h1->logits detach":      dict(src="h1", tgt="logits", detach=True, lam=1.0),
}


def shift(X, kind, v, seed):
    imgs = X.reshape(-1, 8, 8)
    if kind == "rot":
        out = np.stack([rotate(im, v, reshape=False, order=1) for im in imgs])
    elif kind == "noise":
        out = imgs + np.random.default_rng(1000 + seed).normal(0, v, imgs.shape)
    else:
        out = np.stack([gaussian_filter(im, v) for im in imgs])
    return np.clip(out, 0, 1).reshape(-1, 64)


class SelfNet:
    def __init__(self, seed, src, tgt, detach, lam, H=64):
        r = np.random.default_rng(seed)
        he = lambda i, o: r.normal(0, np.sqrt(2 / i), (i, o))
        D = 10 if tgt == "logits" else H
        self.P = {"W1": he(64, H), "b1": np.zeros(H), "W2": he(H, H), "b2": np.zeros(H),
                  "W3": he(H, 10) * 0.5, "b3": np.zeros(10), "W4": he(H, D) * 0.5, "b4": np.zeros(D)}
        self.src, self.tgt, self.detach, self.lam, self.rng = src, tgt, detach, lam, r

    def forward(self, X):
        P = self.P
        c = {"X": X}
        c["z1"] = X @ P["W1"] + P["b1"]; c["h1"] = np.maximum(c["z1"], 0)
        c["z2"] = c["h1"] @ P["W2"] + P["b2"]; c["h2"] = np.maximum(c["z2"], 0)
        c["logits"] = c["h2"] @ P["W3"] + P["b3"]
        c["s"] = c[self.src] @ P["W4"] + P["b4"]
        return c

    def backward(self, c, y):
        P, B = self.P, len(y)
        T = c[self.tgt]
        ds = self.lam * 2 * (c["s"] - T) / (B * T.shape[1])
        extra = {"h1": 0, "h2": 0, "logits": 0}
        extra[self.src] = extra[self.src] + ds @ P["W4"].T
        if not self.detach:
            extra[self.tgt] = extra[self.tgt] - ds
        g = {"W4": c[self.src].T @ ds, "b4": ds.sum(0)}
        p = softmax(c["logits"]); p[np.arange(B), y] -= 1
        dlog = p / B + extra["logits"]
        g["W3"] = c["h2"].T @ dlog; g["b3"] = dlog.sum(0)
        dz2 = (dlog @ P["W3"].T + extra["h2"]) * (c["z2"] > 0)
        g["W2"] = c["h1"].T @ dz2; g["b2"] = dz2.sum(0)
        dz1 = (dz2 @ P["W2"].T + extra["h1"]) * (c["z1"] > 0)
        g["W1"] = c["X"].T @ dz1; g["b1"] = dz1.sum(0)
        return g

    def fit(self, X, y, epochs=EPOCHS, bs=64):
        opt = Adam(self.P)
        for _ in range(epochs):
            idx = self.rng.permutation(len(X))
            for i in range(0, len(X), bs):
                b = idx[i:i + bs]
                opt.step(self.backward(self.forward(X[b]), y[b]))
        return self


def rk(x): return rankdata(x) / len(x)
def auc(wrong, s): return roc_auc_score(wrong, s) if 0 < wrong.sum() < len(wrong) else np.nan


def scores_for(net, X):
    c = net.forward(X)
    p = softmax(c["logits"])
    out = {"forecast": ((c["s"] - c[net.tgt]) ** 2).mean(1)}
    if net.tgt == "logits":
        q = softmax(c["s"])
        out["forecast_KL"] = (p * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum(1)
    return p, out


def main():
    rows = []   # (variant, score_name, shift, seed, acc, auc_F, auc_MSP, auc_AE, auc_MSPAE, auc_all)
    for seed in SEEDS:
        print(f"seed {seed} ...", flush=True)
        Xtr, Xte, ytr, yte = load_data(seed)
        ae = AE(seed).fit(Xtr)
        shifted = {sh: shift(Xte, *sh, seed) for sh in SHIFTS}
        ae_sc = {sh: ae.score(X) for sh, X in shifted.items()}
        for vname, cfg in VARIANTS.items():
            net = SelfNet(seed, **cfg).fit(Xtr, ytr)
            clean_acc = (net.forward(Xte)["logits"].argmax(1) == yte).mean()
            for sh, X in shifted.items():
                p, fsc = scores_for(net, X)
                wrong = (p.argmax(1) != yte).astype(int)
                msp = 1 - p.max(1)
                for sname, F in fsc.items():
                    rows.append(dict(variant=vname, score=sname, shift=f"{sh[0]}{sh[1]}", seed=seed,
                                     clean_acc=clean_acc, acc=1 - wrong.mean(),
                                     F=auc(wrong, F), MSP=auc(wrong, msp), AE=auc(wrong, ae_sc[sh]),
                                     MSP_AE=auc(wrong, rk(msp) + rk(ae_sc[sh])),
                                     MSP_F=auc(wrong, rk(msp) + rk(F)),
                                     ALL=auc(wrong, rk(msp) + rk(ae_sc[sh]) + rk(F))))
    json.dump(rows, open("week2_rows.json", "w"), default=float)

    import collections
    keys = sorted({(r["variant"], r["score"]) for r in rows}, key=lambda k: list(VARIANTS).index(k[0]))
    shifts = [f"{a}{b}" for a, b in SHIFTS]

    print("\n=== Clean accuracy per variant (mean over seeds) ===")
    for v in VARIANTS:
        print(f"  {v:<28}{np.mean([r['clean_acc'] for r in rows if r['variant']==v]):.3f}")

    print("\n=== Accuracy under each shift (variant A) — mild regime check ===")
    print("  " + "  ".join(f"{s:>8}" for s in shifts))
    print("  " + "  ".join(f"{np.mean([r['acc'] for r in rows if r['shift']==s and r['variant']==list(VARIANTS)[0]]):>8.3f}" for s in shifts))

    print("\n=== Within-shift error-AUROC, averaged over all 11 mild shifts x 5 seeds ===")
    print(f"  {'variant / score':<40}{'F alone':>9}{'MSP':>7}{'AE':>7}{'MSP+AE':>8}{'MSP+F':>8}{'ALL':>7}{'GAIN':>16}{'shifts GAIN>0':>15}")
    summary = {}
    for v, s in keys:
        R = [r for r in rows if r["variant"] == v and r["score"] == s]
        m = lambda k: np.nanmean([r[k] for r in R])
        # gain per (seed, shift); then average per seed for std
        gain_seed = [np.nanmean([r["ALL"] - r["MSP_AE"] for r in R if r["seed"] == sd]) for sd in SEEDS]
        per_shift_gain = [np.nanmean([r["ALL"] - r["MSP_AE"] for r in R if r["shift"] == sh]) for sh in shifts]
        npos = sum(g > 0 for g in per_shift_gain)
        summary[f"{v} | {s}"] = dict(F=m("F"), MSP=m("MSP"), AE=m("AE"), MSP_AE=m("MSP_AE"), MSP_F=m("MSP_F"),
                                    ALL=m("ALL"), gain=np.mean(gain_seed), gain_std=np.std(gain_seed),
                                    per_shift_gain=dict(zip(shifts, per_shift_gain)))
        print(f"  {v+' / '+s:<40}{m('F'):>9.3f}{m('MSP'):>7.3f}{m('AE'):>7.3f}{m('MSP_AE'):>8.3f}{m('MSP_F'):>8.3f}{m('ALL'):>7.3f}"
              f"{np.mean(gain_seed):>+10.3f}±{np.std(gain_seed):.3f}{npos:>8}/{len(shifts)}")

    print("\n=== Forecast-alone error-AUROC per shift (vs MSP of same net) ===")
    print(f"  {'variant / score':<40}" + "".join(f"{s:>9}" for s in shifts))
    for v, s in keys:
        R = [r for r in rows if r["variant"] == v and r["score"] == s]
        print(f"  {v+' / '+s:<40}" + "".join(f"{np.nanmean([r['F'] for r in R if r['shift']==sh]):>9.3f}" for sh in shifts))
    R = [r for r in rows if r["variant"] == list(VARIANTS)[0] and r["score"] == "forecast"]
    print(f"  {'[MSP, net A]':<40}" + "".join(f"{np.nanmean([r['MSP'] for r in R if r['shift']==sh]):>9.3f}" for sh in shifts))

    json.dump(summary, open("week2_summary.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
