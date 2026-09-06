"""
SIMPLEST POSSIBLE self-forecasting experiment
================================================
Koi PyTorch nahi, koi MNIST nahi. Sirf numpy + manual backprop
taake har step samajh mein aaye.

STRUCTURE:
  input (2D point) -> h1 (hidden layer 1) -> h2 (hidden layer 2) -> output (2 classes)

  Normal network: sirf classification seekhta hai (input -> output sahi).
  Self-modeling network: EXTRA task bhi hai -> "FORECASTER" jo h1 dekh kar
                          h2 ko predict karne ki koshish karta hai.

  Forecast error = |predicted_h2 - actual_h2|
  Hypothesis: familiar (in-distribution) input pe forecast error LOW,
              unusual (OOD/shifted) input pe forecast error HIGH.
"""

import numpy as np

np.random.seed(42)

# ----------------------------------------------------------------------
# STEP 1: Bohat simple synthetic dataset banao
# ----------------------------------------------------------------------
# Class 0: points around (-2, -2)
# Class 1: points around (2, 2)
# Ye "normal" / training distribution hai.

def make_normal_data(n_per_class=100):
    class0 = np.random.randn(n_per_class, 2) * 0.5 + np.array([-2, -2])
    class1 = np.random.randn(n_per_class, 2) * 0.5 + np.array([2, 2])
    X = np.vstack([class0, class1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y

def make_shifted_data(n=200):
    # Ye "unusual" input hai -- bilkul alag region se, jo model ne
    # kabhi training mein nahi dekha (jaise "rotated digit" ka analogy).
    # Zyada points banate hain taake kuch correct aur kuch wrong dono milein.
    X = np.random.randn(n, 2) * 1.5 + np.array([6, -6])
    # "Pseudo ground truth": asal duniya mein bhi humein pata hota hai ke
    # ye point CONCEPTUALLY kis class ka hona chahiye tha (jaise ek
    # rotated "7" ka TRUE answer phir bhi 7 hi hota hai). Yahan hum use
    # karte hain: jo training-cluster-center (-2,-2) ya (2,2) ke zyada
    # qareeb hai, wahi "true" class hai.
    center0 = np.array([-2, -2])
    center1 = np.array([2, 2])
    dist0 = np.linalg.norm(X - center0, axis=1)
    dist1 = np.linalg.norm(X - center1, axis=1)
    y_true_pseudo = (dist1 < dist0).astype(int)
    return X, y_true_pseudo

X_train, y_train = make_normal_data(100)
X_test_normal, y_test_normal = make_normal_data(20)   # normal test
X_test_shifted, y_test_shifted_pseudo = make_shifted_data(200)   # OOD test, more points

print(f"Training data shape: {X_train.shape}")
print(f"Normal test shape:   {X_test_normal.shape}")
print(f"Shifted/OOD shape:   {X_test_shifted.shape}")

# ----------------------------------------------------------------------
# STEP 2: Network weights initialize karo
# ----------------------------------------------------------------------
# Architecture: 2 (input) -> 4 (h1) -> 4 (h2) -> 2 (output classes)
# Forecaster:   4 (h1) -> 4 (predicted h2)

input_dim, h1_dim, h2_dim, out_dim = 2, 4, 4, 2

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

# Main classifier weights
W1 = np.random.randn(input_dim, h1_dim) * 0.5
b1 = np.zeros(h1_dim)
W2 = np.random.randn(h1_dim, h2_dim) * 0.5
b2 = np.zeros(h2_dim)
W3 = np.random.randn(h2_dim, out_dim) * 0.5
b3 = np.zeros(out_dim)

# Forecaster weights (h1 -> predicted h2)
Wf = np.random.randn(h1_dim, h2_dim) * 0.5
bf = np.zeros(h2_dim)

lr = 0.05
lambda_forecast = 0.3  # kitna weight forecast-loss ko dena hai

# ----------------------------------------------------------------------
# STEP 3: Training loop (manual forward + backward pass)
# ----------------------------------------------------------------------
n_epochs = 2000
n_samples = X_train.shape[0]

for epoch in range(n_epochs):
    # ---- FORWARD PASS ----
    z1 = X_train @ W1 + b1
    h1 = relu(z1)                    # hidden state 1

    z2 = h1 @ W2 + b2
    h2 = relu(z2)                    # hidden state 2 (ACTUAL)

    logits = h2 @ W3 + b3
    probs = softmax(logits)

    # forecaster predicts h2 FROM h1 (before h2 is even computed "for real")
    predicted_h2 = relu(h1 @ Wf + bf)

    # ---- LOSSES ----
    y_onehot = np.zeros((n_samples, out_dim))
    y_onehot[np.arange(n_samples), y_train] = 1
    classification_loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-9), axis=1))
    forecast_loss = np.mean((predicted_h2 - h2) ** 2)
    total_loss = classification_loss + lambda_forecast * forecast_loss

    # ---- BACKWARD PASS (classification branch) ----
    dlogits = (probs - y_onehot) / n_samples
    dW3 = h2.T @ dlogits
    db3 = np.sum(dlogits, axis=0)

    # forecast_loss = mean((predicted_h2 - h2)^2) depends on h2 too!
    # d(forecast_loss)/d(h2) = -2*(predicted_h2 - h2)/n * lambda = -dpred
    # (dpred is computed a bit further below, so compute the h2-side term here directly)
    dh2_from_forecast = -2 * (predicted_h2 - h2) / n_samples * lambda_forecast

    dh2 = dlogits @ W3.T + dh2_from_forecast
    dz2 = dh2 * relu_deriv(z2)
    dW2 = h1.T @ dz2
    db2 = np.sum(dz2, axis=0)

    dh1_from_class = dz2 @ W2.T
    dz1_from_class = dh1_from_class * relu_deriv(z1)

    # ---- BACKWARD PASS (forecaster branch) ----
    dpred = 2 * (predicted_h2 - h2) / n_samples * lambda_forecast
    dz_f = dpred * relu_deriv(h1 @ Wf + bf)
    dWf = h1.T @ dz_f
    dbf = np.sum(dz_f, axis=0)

    dh1_from_forecast = dz_f @ Wf.T
    dz1_from_forecast = dh1_from_forecast * relu_deriv(z1)

    # h1 gets gradient from BOTH branches
    dz1 = dz1_from_class + dz1_from_forecast
    dW1 = X_train.T @ dz1
    db1 = np.sum(dz1, axis=0)

    # ---- UPDATE WEIGHTS ----
    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * dW2; b2 -= lr * db2
    W3 -= lr * dW3; b3 -= lr * db3
    Wf -= lr * dWf; bf -= lr * dbf

    if epoch % 500 == 0:
        acc = np.mean(np.argmax(probs, axis=1) == y_train)
        print(f"Epoch {epoch:4d} | class_loss={classification_loss:.4f} "
              f"forecast_loss={forecast_loss:.4f} train_acc={acc:.2f}")

print("\nTraining done.\n")

# ----------------------------------------------------------------------
# STEP 4: Forward pass function (no training, just inference)
# ----------------------------------------------------------------------
def forward(X):
    z1 = X @ W1 + b1
    h1 = relu(z1)
    z2 = h1 @ W2 + b2
    h2 = relu(z2)
    logits = h2 @ W3 + b3
    probs = softmax(logits)
    predicted_h2 = relu(h1 @ Wf + bf)
    forecast_error = np.linalg.norm(predicted_h2 - h2, axis=1)  # per-sample
    confidence = np.max(probs, axis=1)
    pred_class = np.argmax(probs, axis=1)
    return pred_class, confidence, forecast_error

# ----------------------------------------------------------------------
# STEP 5: MAIN RESULT -- compare normal vs shifted/OOD
# ----------------------------------------------------------------------
pred_n, conf_n, ferr_n = forward(X_test_normal)
pred_s, conf_s, ferr_s = forward(X_test_shifted)

acc_normal = np.mean(pred_n == y_test_normal)

# Split shifted/OOD predictions into CORRECT vs WRONG using pseudo ground-truth
shifted_correct_mask = (pred_s == y_test_shifted_pseudo)
shifted_wrong_mask = ~shifted_correct_mask

print("=" * 60)
print("RESULTS -- 3-way comparison")
print("=" * 60)
print(f"Normal test accuracy:  {acc_normal:.2f}")
print(f"Shifted/OOD 'accuracy' (vs pseudo ground-truth): {shifted_correct_mask.mean():.2f}")
print(f"  ({shifted_wrong_mask.sum()} out of {len(pred_s)} shifted points were WRONG)")
print()

print(f"1) NORMAL input (in-distribution, correct):")
print(f"   avg confidence:      {conf_n.mean():.4f}")
print(f"   avg forecast error:  {ferr_n.mean():.4f}")
print()

if shifted_correct_mask.sum() > 0:
    print(f"2) SHIFTED input, but model happened to be CORRECT:")
    print(f"   avg confidence:      {conf_s[shifted_correct_mask].mean():.4f}")
    print(f"   avg forecast error:  {ferr_s[shifted_correct_mask].mean():.4f}")
    print()
else:
    print("2) SHIFTED input, correct: (no samples in this bucket)\n")

if shifted_wrong_mask.sum() > 0:
    print(f"3) SHIFTED input, model was WRONG:")
    print(f"   avg confidence:      {conf_s[shifted_wrong_mask].mean():.4f}  <- still high? overconfident!")
    print(f"   avg forecast error:  {ferr_s[shifted_wrong_mask].mean():.4f}  <- highest of all?")
    print()
else:
    print("3) SHIFTED input, wrong: (no samples in this bucket -- try wider shift/noise)\n")

print("=" * 60)
print("HYPOTHESIS CHECK: forecast error should increase as: ")
print("   Normal < Shifted-but-correct < Shifted-and-WRONG")
print("=" * 60)
