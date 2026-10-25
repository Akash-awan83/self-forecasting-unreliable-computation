import numpy as np, week1_baselines as W
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
def rk(x): return rankdata(x)/len(x)
R={s:W.run_seed(s) for s in W.SEEDS}
combos={"MSP alone":lambda S:S["1-MSP (same net)"],
 "forecast alone":lambda S:S["forecast_err (ours)"],
 "AE alone":lambda S:S["AE recon err"],
 "MSP+forecast":lambda S:rk(S["1-MSP (same net)"])+rk(S["forecast_err (ours)"]),
 "MSP+AE":lambda S:rk(S["1-MSP (same net)"])+rk(S["AE recon err"]),
 "MSP+AE+forecast":lambda S:rk(S["1-MSP (same net)"])+rk(S["AE recon err"])+rk(S["forecast_err (ours)"])}
print(f"{'combo':<18}{'err-AUROC@15':>14}{'err-AUROC@30':>14}{'pooled AURC':>14}")
for n,f in combos.items():
    a15=[roc_auc_score(~R[s][15]["correct"],f(R[s][15]["scores"])) for s in W.SEEDS]
    a30=[roc_auc_score(~R[s][30]["correct"],f(R[s][30]["scores"])) for s in W.SEEDS]
    au=[]
    for s in W.SEEDS:
        keys=[0]+W.ANGLES
        S={k:np.concatenate([R[s][a]["scores"][k] for a in keys]) for k in R[s][0]["scores"]}
        cor=np.concatenate([R[s][a]["correct"] for a in keys])
        au.append(W.aurc(f(S),cor))
    print(f"{n:<18}{np.mean(a15):>10.3f}±{np.std(a15):.2f}{np.mean(a30):>10.3f}±{np.std(a30):.2f}{np.mean(au):>10.4f}±{np.std(au):.3f}")
