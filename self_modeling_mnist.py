"""
SELF-MODELING EXPERIMENT -- MNIST-style (paper ki exact architecture)
======================================================================
Paper: "Unexpected Benefits of Self-Modeling in Neural Systems"
       (Premakumar et al. 2024, arXiv:2407.10188)

ARCHITECTURE (paper se seedha copy):
  Image -> MLP -> Hidden Layer [64 neurons] -> Final Layer [74 outputs]
                                                  = 10 classification logits
                                                  + 64 self-model predictions

  Loss = wc * Classification_Loss + ws * (a_hat - a)^2
  (wc, ws = paper mein "Auxiliary Weight" AW kehte hain)

NOTE: Asal MNIST (28x28, 60k images) is container mein download nahi ho
saka (no internet access for dataset download). Iski jagah scikit-learn
ka built-in "Digits" dataset use kar rahe hain -- 8x8 handwritten
digits, 10 classes, same concept, bina internet ke available hota hai.
Concept/architecture bilkul same hai, sirf image resolution chhoti hai.

HUMARA EXTRA TEST (paper mein nahi tha): rotation se distribution-shift
banate hain, aur dekhte hain ke self-modeling se calibration/OOD
behavior better hota hai ya nahi -- yehi tumhara hypothesis hai.
"""

import numpy as np
from sklearn.datasets import load_digits
from scipy.ndimage import rotate

np.random.seed(0)

# ----------------------------------------------------------------------
# STEP 1: Data load karo
# ----------------------------------------------------------------------
digits = load_digits()
X_all = digits.images  # shape (1797, 8, 8), values 0-16
y_all = digits.target  # 0-9

# Normalize pixel values to 0-1
X_all = X_all / 16.0

# Train/test split (manual, no sklearn helper -- taake sab dikhe)
n_total = len(X_all)
idx = np.random.permutation(n_total)
n_train = int(0.8 * n_total)
train_idx, test_idx = idx[:n_train], idx[n_train:]

X_train_imgs = X_all[train_idx]
y_train = y_all[train_idx]
X_test_imgs = X_all[test_idx]
y_test = y_all[test_idx]

# Flatten images (8x8 -> 64) for MLP input
X_train = X_train_imgs.reshape(len(X_train_imgs), -1)
X_test = X_test_imgs.reshape(len(X_test_imgs), -1)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Distribution-shift test set: rotate test images by 45 degrees
X_test_rotated_imgs = np.array([rotate(img, angle=45, reshape=False) for img in X_test_imgs])
X_test_rotated = X_test_rotated_imgs.reshape(len(X_test_rotated_imgs), -1)
X_test_rotated = np.clip(X_test_rotated, 0, 1)

# ----------------------------------------------------------------------
# STEP 2: Architecture setup (paper ke mutabiq)
# ----------------------------------------------------------------------
input_dim = 64          # 8x8 flattened
hidden_dim = 64          # paper ka smallest tested hidden size
n_classes = 10
n_selfmodel_outputs = hidden_dim  # self-model predicts ALL hidden activations
# Total final-layer outputs = 10 + 64 = 74  <-- yehi tumhara "74" hai!

AW = 1.0
lr = 0.05
n_epochs = 400

def relu(x):
    # Leaky ReLU: negative side ko bilkul zero nahi karte (chhota slope
    # rakhte hain), taake self-modeling ka extra gradient neurons ko
    # PERMANENTLY na maar sake (standard ReLU se ye collapse ho raha tha)
    return np.where(x > 0, x, 0.01 * x)

def relu_deriv(x):
    return np.where(x > 0, 1.0, 0.01)

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

# Layer 1: input -> hidden
W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
b1 = np.zeros(hidden_dim)

# Final layer: hidden -> 74 outputs (10 class logits + 64 self-model preds)
# Hum inko alag rakhte hain code mein clarity ke liye, lekin "same final
# layer" ka matlab hai dono hidden activations 'a' se seedha nikalte hain.
W_class = np.random.randn(hidden_dim, n_classes) * np.sqrt(2.0 / hidden_dim)
b_class = np.zeros(n_classes)

W_selfmodel = np.random.randn(hidden_dim, n_selfmodel_outputs) * np.sqrt(2.0 / hidden_dim)
b_selfmodel = np.zeros(n_selfmodel_outputs)

# ----------------------------------------------------------------------
# STEP 3: Training (SELF-MODELING model)
# ----------------------------------------------------------------------
def train_model(use_selfmodel, X_tr, y_tr, epochs=n_epochs):
    """use_selfmodel=False -> baseline (paper ke 'control' network jaisa)
    Adam optimizer use kar rahe hain -- plain SGD is joint (2-task) loss
    ke saath unstable nikla (dead-unit collapse), Adam har parameter ke
    liye apna adaptive step-size rakhta hai, isliye zyada robust hai."""
    W1_ = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
    b1_ = np.zeros(hidden_dim)
    Wc_ = np.random.randn(hidden_dim, n_classes) * np.sqrt(2.0 / hidden_dim)
    bc_ = np.zeros(n_classes)
    Ws_ = np.random.randn(hidden_dim, n_selfmodel_outputs) * np.sqrt(2.0 / hidden_dim)
    bs_ = np.zeros(n_selfmodel_outputs)

    # Adam optimizer state (m = 1st moment, v = 2nd moment) har parameter ke liye
    params = {"W1": W1_, "b1": b1_, "Wc": Wc_, "bc": bc_, "Ws": Ws_, "bs": bs_}
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(val) for k, val in params.items()}
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    adam_lr = 0.01

    def adam_step(name, grad, t):
        m[name][:] = beta1 * m[name] + (1 - beta1) * grad
        v[name][:] = beta2 * v[name] + (1 - beta2) * (grad ** 2)
        m_hat = m[name] / (1 - beta1 ** t)
        v_hat = v[name] / (1 - beta2 ** t)
        params[name] -= adam_lr * m_hat / (np.sqrt(v_hat) + eps_adam)

    n = X_tr.shape[0]
    y_onehot = np.zeros((n, n_classes))
    y_onehot[np.arange(n), y_tr] = 1

    for epoch in range(epochs):
        t = epoch + 1  # Adam bias-correction ke liye time-step 1 se shuru
        # AW warm-up: pehle 100 epochs mein 0 se target AW tak gradually
        current_aw = AW * min(1.0, epoch / 100.0) if use_selfmodel else 0.0

        W1_, b1_, Wc_, bc_, Ws_, bs_ = params["W1"], params["b1"], params["Wc"], params["bc"], params["Ws"], params["bs"]

        # ---- FORWARD ----
        z1 = X_tr @ W1_ + b1_
        a = relu(z1)  # hidden activations -- self-modeling TARGET

        logits = a @ Wc_ + bc_
        probs = softmax(logits)

        a_hat = a @ Ws_ + bs_  # self-model prediction of a (from itself -- via same hidden layer)

        # ---- LOSSES ----
        class_loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-9), axis=1))
        selfmodel_loss = np.mean((a_hat - a) ** 2) if use_selfmodel else 0.0

        # ---- BACKWARD: classification branch ----
        dlogits = (probs - y_onehot) / n
        dWc = a.T @ dlogits
        dbc = np.sum(dlogits, axis=0)
        da_from_class = dlogits @ Wc_.T

        # ---- BACKWARD: self-model branch ----
        if use_selfmodel:
            da_hat = 2 * (a_hat - a) / n * current_aw
            dWs = a.T @ da_hat
            dbs = np.sum(da_hat, axis=0)
            da_from_selfmodel_pred = da_hat @ Ws_.T
            da_from_selfmodel_target = -da_hat  # a bhi target hai, gradient wapas aata hai
        else:
            dWs = np.zeros_like(Ws_)
            dbs = np.zeros_like(bs_)
            da_from_selfmodel_pred = 0.0
            da_from_selfmodel_target = 0.0

        da = da_from_class + da_from_selfmodel_pred + da_from_selfmodel_target
        dz1 = da * relu_deriv(z1)
        dW1 = X_tr.T @ dz1
        db1 = np.sum(dz1, axis=0)

        # ---- UPDATE (Adam) ----
        adam_step("W1", dW1, t); adam_step("b1", db1, t)
        adam_step("Wc", dWc, t); adam_step("bc", dbc, t)
        if use_selfmodel:
            adam_step("Ws", dWs, t); adam_step("bs", dbs, t)

        if epoch % 50 == 0:
            acc = np.mean(np.argmax(probs, axis=1) == y_tr)
            tag = "SELF-MODEL" if use_selfmodel else "BASELINE  "
            print(f"[{tag}] Epoch {epoch:3d} | class_loss={class_loss:.4f} "
                  f"selfmodel_loss={selfmodel_loss:.4f} train_acc={acc:.3f}")

    return params["W1"], params["b1"], params["Wc"], params["bc"], params["Ws"], params["bs"]

print("\n=== Training BASELINE (no self-modeling) ===")
base_weights = train_model(use_selfmodel=False, X_tr=X_train, y_tr=y_train)

print("\n=== Training SELF-MODELING network ===")
sm_weights = train_model(use_selfmodel=True, X_tr=X_train, y_tr=y_train)

# ----------------------------------------------------------------------
# STEP 4: Evaluate -- complexity, accuracy, AND our OOD/calibration test
# ----------------------------------------------------------------------
def evaluate(weights, X_ev, y_ev=None):
    W1_, b1_, Wc_, bc_, Ws_, bs_ = weights
    z1 = X_ev @ W1_ + b1_
    a = relu(z1)
    logits = a @ Wc_ + bc_
    probs = softmax(logits)
    pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    a_hat = a @ Ws_ + bs_
    forecast_err = np.linalg.norm(a_hat - a, axis=1)
    acc = np.mean(pred == y_ev) if y_ev is not None else None
    return pred, conf, forecast_err, acc

print("\n" + "=" * 70)
print("PAPER-STYLE METRIC: network complexity (weight distribution width)")
print("=" * 70)
print(f"Baseline final-layer weight std:     {base_weights[2].std():.4f}")
print(f"Self-model final-layer weight std:   {sm_weights[2].std():.4f}")
print("(Paper's finding: self-modeling should make this NARROWER)")

print("\n" + "=" * 70)
print("OUR EXTRA TEST: calibration under distribution shift (rotation)")
print("=" * 70)

for name, weights in [("BASELINE", base_weights), ("SELF-MODEL", sm_weights)]:
    pred_n, conf_n, ferr_n, acc_n = evaluate(weights, X_test, y_test)
    pred_r, conf_r, ferr_r, acc_r = evaluate(weights, X_test_rotated, y_test)

    print(f"\n--- {name} ---")
    print(f"Normal test accuracy:   {acc_n:.3f} | avg confidence: {conf_n.mean():.3f}")
    print(f"Rotated test accuracy:  {acc_r:.3f} | avg confidence: {conf_r.mean():.3f}")
    print(f"  (confidence drop when wrong: is model 'honest' about being worse?)")
    if name == "SELF-MODEL":
        print(f"Forecast error -- normal:  {ferr_n.mean():.4f}")
        print(f"Forecast error -- rotated: {ferr_r.mean():.4f}")
        ratio = ferr_r.mean() / (ferr_n.mean() + 1e-9)
        print(f"Forecast error ratio (rotated/normal): {ratio:.2f}x")
