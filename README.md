# Self-Model Forecast Error: A Shift Detector, Not an Error Detector

**Summary.** I tested whether a network's error in predicting its own hidden
activations can flag *wrong predictions* better than ordinary output confidence.
Across 5 self-model variants, 8 baselines, 16 distribution shifts and 5 seeds, it
cannot. Maximum softmax probability (MSP) beat every variant. The signal turns
out to be a strong **distribution-shift** detector that does not identify *which
answers are wrong*. This repository contains the code, the full results, and a
measurement error I made and then corrected.

---

## 1. The question

Premakumar et al. (2024, [arXiv:2407.10188](https://arxiv.org/abs/2407.10188))
train networks with an auxiliary task: predict their own internal state. I asked
a follow-on question:

> If a network is trained to forecast its own activations, does the error in that
> forecast tell us when the network's prediction is wrong?

This matters because output confidence is known to stay high on inputs a model
gets wrong, especially under distribution shift. An *internal* signal that did
better would be genuinely useful.

## 2. Setup

Small MLPs on handwritten-digit classification, implemented from scratch in NumPy
with manual backpropagation and Adam. No deep-learning framework. Every number
below is averaged over **5 seeds**.

**Five self-model variants:**

| ID | Variant | What the self-model predicts |
|----|---------|------------------------------|
| A  | Original | layer 2 activations to layer 1 activations |
| A' | Original, low weight | same as A, with a smaller self-model loss weight |
| B  | Detached | same as A, but the self-model does not change the main network |
| C  | Reversed forecast | layer 1 to the next layer's activations |
| D  | Output forecast | layer 1 to the network's own final output |

**Shifts.** 16 in total. The headline comparison uses the 11 *mild* shifts (small
rotation, added noise, blur), where accuracy ranges from 36% to 96%. That is the
regime where "which answer is wrong?" is a real question. Under heavy shift
almost everything is wrong and the task becomes trivial.

**Baselines.** 8, including maximum softmax probability (MSP) and autoencoder
reconstruction error.

## 3. Result

Detecting **wrong answers** (AUROC, higher is better):

| Method | AUROC |
|--------|-------|
| MSP (softmax confidence) | **0.837** |
| D — output forecast (KL score) | 0.814 |
| MSP + D combined | 0.834 |
| A, A', B, C — activation forecast | 0.55 – 0.57 |

- All four activation-forecast variants sit close to random (0.5).
- Detaching the self-model (B) changed nothing, so the result is not caused by the
  self-model interfering with the main network's training.
- D looked promising at first, but it sits *below* MSP, and combining it with MSP
  does not beat MSP alone. That is what you would expect: D forecasts the model's
  own output, so it is largely another view of softmax rather than a new signal.

## 4. A measurement error I made

An earlier run reported D as a **+0.039** gain. That number was scored against an
**MSP + autoencoder** composite baseline. Under mild shift the autoencoder term
actively hurts, so that composite was a weak comparison point and the gain was an
artifact of choosing it.

Scored against the *best* baseline, D is **−0.010**.

The lesson I took from this: fix the strongest baseline before measuring a gain,
not after.

## 5. What this does and does not show

**It does show.** Under these conditions, self-model forecast error does not
identify wrong predictions. Under large shift it behaves much like an autoencoder
reconstruction error — it detects *new data*, not *bad computation*. The result
holds across 5 variants, 16 shifts and 5 seeds, so it is not one unlucky
configuration.

**It does not show.** This is a small MLP on digits. A larger model with real
representational capacity may use a self-model differently, and this experiment
cannot rule that out.

## 6. Where I would go next

Whether a probe on a small open LLM's hidden states can beat logit-based
confidence at flagging wrong answers — with the baselines fixed from day one, and
on a real model rather than a toy.

## 7. Files in this repository

| File | What it is |
|------|------------|
| `week1_baselines.py` | baseline methods |
| `week1_output.txt`, `week1_results.json` | baseline results |
| `week2_rescue.py` | the five self-model variants |
| `week2_output.txt` | results for the five variants |
| `combo_check.py`, `combo_output.txt` | combining the self-model score with MSP |
| `best_baseline_check.txt` | the corrected comparison described in section 4 |
| `self_modeling_mnist.py`, `multi_seed_test.py`, `self_forecast_toy.py` | earlier implementations |

## 8. Reference

Premakumar et al., *Unexpected benefits of self-modeling in neural systems*, 2024.
[arXiv:2407.10188](https://arxiv.org/abs/2407.10188)
