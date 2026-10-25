# Self-Model Forecast Error: A Shift Detector, Not an Error Detector

**Summary.** I tested whether a network's error in predicting its own hidden
activations can flag *wrong predictions* better than ordinary output confidence.
Across 5 self-model variants, 8 baselines, 16 distribution shifts and 5 seeds,
it cannot. Maximum softmax probability (MSP) beat every variant. The signal is a
strong **distribution-shift** detector, but it does not identify *which answers
are wrong*. This repository contains the code, the full results, and a
measurement error I made and corrected.

---

## 1. The question

Premakumar et al. (2024, [arXiv:2407.10188](https://arxiv.org/abs/2407.10188))
train networks with an auxiliary task: predict their own internal state. I asked
a follow-on question:

> If a network is trained to forecast its own activations, does the error in that
> forecast tell us when the network's prediction is wrong?

This matters because output confidence is known to stay high on inputs the model
gets wrong, especially under distribution shift. An *internal* signal that did
better would be useful.

## 2. What I built

<!-- TODO: fill in dataset, architecture sizes, training budget -->

- **Dataset:** <digits dataset, size, split>
- **Architecture:** small MLP, <layer sizes>, implemented from scratch in NumPy
  (manual backpropagation, Adam). No deep-learning framework.
- **Seeds:** 5

**Five self-model variants:**

| ID | Variant | What the self-model predicts |
|----|---------|------------------------------|
| A  | Original | layer 2 → layer 1 activations |
| A' | Original, low weight | same, with a smaller self-model loss weight |
| B  | Detached | same as A, but the self-model does not change the main network |
| C  | Reversed forecast | layer 1 → the next layer's activations |
| D  | Output forecast | layer 1 → the network's own final output |

**Shifts:** 16 total. The headline comparison uses the 11 *mild* shifts (small
rotation, added noise, blur), where accuracy ranges from 36% to 96%. This is the
regime where "which answer is wrong?" is a real question; under heavy shift
almost everything is wrong and the task is trivial.

**Baselines:** 8 <!-- TODO: list them, e.g. MSP, max logit, entropy, margin,
energy, MC dropout, deep ensemble, autoencoder reconstruction error -->

## 3. Result

Detecting **wrong answers** (AUROC, higher is better):

| Method | AUROC |
|--------|-------|
| MSP (softmax confidence) | **0.837** |
| D — output forecast (KL score) | 0.814 |
| MSP + D combined | 0.834 |
| A, A', B, C — activation forecast | 0.55 – 0.57 |

- All four activation-forecast variants are close to random (0.5).
- Detaching the self-model (B) changed nothing, so the effect is not caused by the
  self-model interfering with training.
- D looked promising, but it sits *below* MSP, and combining it with MSP does not
  beat MSP alone. That is expected: D forecasts the model's own output, so it is
  largely another view of softmax rather than a new signal.

## 4. A measurement error I made

My earlier run reported D as a **+0.039** gain. That number was scored against an
**MSP + autoencoder** baseline. Under mild shift the autoencoder term actively
hurts, so that composite was a weak comparison point.

Scored against the *best* baseline, D is **−0.010**.

The lesson I took: fix the strongest baseline before measuring a gain, not after.

## 5. What this does and does not show

**Does show.** Under these conditions, self-model forecast error does not identify
wrong predictions. Under large shift it mostly behaves like an autoencoder
reconstruction error, i.e. it detects *new data*, not *bad computation*. The
result is consistent across 5 variants, 16 shifts and 5 seeds, so it is not a
single unlucky configuration.

**Does not show.** This is a tiny MLP on digits. A larger model with real
representational capacity may use a self-model differently. The honest next
question is whether a probe on a small open LLM's hidden states can beat
logit-based confidence at flagging wrong answers — with the baselines fixed from
day one.

## 6. Reproducing

```bash
<!-- TODO: exact commands -->
python week2_rescue.py
python best_baseline_check.py
```

<!-- TODO: point to the raw result files -->

## 7. References

- Premakumar et al., *Unexpected benefits of self-modeling in neural systems*, 2024. arXiv:2407.10188
- <!-- TODO: add the baseline papers you used, e.g. Hendrycks & Gimpel 2017 for MSP -->
