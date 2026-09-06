# Self-Forecasting as a Signal of Unreliable Computation

**Status:** Preliminary toy experiment (not a peer-reviewed result) — built as a research artifact exploring whether self-modeling can improve model calibration under distribution shift.

## Motivation

[Premakumar et al. 2024](https://arxiv.org/abs/2407.10188) showed that when a neural network is trained to predict its own hidden-layer activations (an auxiliary "self-modeling" task), the network becomes structurally simpler and more parameter-efficient. Their work measured this effect via network complexity (RLCT, weight-distribution width) — it did not test whether self-modeling affects **calibration under distribution shift**.

This repo asks a different question:

> **Can the error in a network's self-prediction (its "forecast error") serve as a more honest signal of unreliable computation than softmax confidence — especially when the model is fed unfamiliar, out-of-distribution inputs?**

Standard softmax confidence is known to stay high even when a model is wrong on out-of-distribution (OOD) inputs ("overconfidence under distribution shift"). The hypothesis here is that forecast error — the mismatch between what a network predicts its own internal state will be, and what that state actually is — might reveal unreliability that the output layer hides.

## Architecture

Same core setup as the original paper (10 classification outputs + N self-model outputs from the same hidden layer), extended with a distribution-shift evaluation:

```
Input → MLP → Hidden Layer [a] → Final Layer
                                    ├── 10 classification logits
                                    └── N self-model predictions (â) of [a]

Loss = classification_loss + AW × (â − a)²
```

## Files

| File | What it does |
|---|---|
| `self_forecast_toy.py` | Simplest possible version — 2D synthetic data, hand-written manual example, sanity-checks the forecast-error concept before scaling up. |
| `self_modeling_mnist.py` | Paper-matching architecture on the `sklearn` Digits dataset (8×8 handwritten digits, closest available dataset to MNIST without external downloads). Tests baseline vs. self-modeling network under 45° rotation as an OOD shift. |
| `multi_seed_test.py` | Repeats the MNIST-style experiment across 5 random seeds to check whether the result is consistent or a single-run fluke. |

## Results (5 seeds)

| Metric | Normal (in-distribution) | Rotated (OOD) |
|---|---|---|
| Softmax confidence | ~0.97 | ~0.94 *(barely drops — overconfident)* |
| Accuracy | ~97% | ~13% *(collapses)* |
| Forecast error | ~0.09 | ~0.86 *(**~9.7x higher**, consistent across all 5 seeds)* |

**Finding:** Forecast error increased consistently (6–12x, mean 9.68x) on OOD data across all 5 seeds, while softmax confidence remained misleadingly high. This supports the hypothesis that self-forecasting error is a more honest — if imperfect — signal of unreliable computation than output-layer confidence.

**What did *not* replicate:** the original paper's finding that self-modeling narrows the weight distribution (reduces complexity) did not hold in this setup — self-modeling networks had *wider* final-layer weights than baseline across all 5 seeds. Possible reasons: small hidden layer (64), single Auxiliary Weight value tested, Adam optimizer vs. the original paper's setup, small dataset. This is reported honestly as an open discrepancy, not smoothed over.

## Limitations (this is a toy experiment, not a finished result)

- Uses the small 8×8 `sklearn` Digits dataset, not full MNIST (no external dataset download available in the dev environment)
- Only one type of distribution shift tested (rotation) — no noise, no adversarial perturbation
- No AUROC or other proper discrimination metric — only mean forecast-error ratios
- No baseline comparison against existing OOD-detection methods (e.g., MC-Dropout, deep ensembles)
- 5 random seeds — a stronger claim would need more

## Next steps

- Scale to full MNIST/CIFAR-10
- Test additional shift types (Gaussian noise, adversarial examples)
- Compute AUROC for "forecast error predicts wrong prediction"
- Compare against softmax-based and ensemble-based OOD detection baselines
- Investigate why the complexity-reduction result didn't replicate

## Background reading

- Premakumar et al., ["Unexpected Benefits of Self-Modeling in Neural Systems"](https://arxiv.org/abs/2407.10188), arXiv:2407.10188 (2024)
