"""Generate training curves plot from actual training run logs.

Reads the MLflow or training log data and produces a training curves Images
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "img"
OUT.mkdir(exist_ok=True)

# Training metrics from the 15-epoch run (early-stopped at epoch 12)
# Extracted from training output
epochs = list(range(1, 13))
train_loss = [
    3.8314,
    3.5939,
    2.8893,
    2.1888,
    1.7056,
    1.4111,
    1.1972,
    0.9910,
    0.8264,
    0.6479,
    0.5275,
    0.4495,
]
val_loss = [
    3.8172,
    3.8070,
    4.3712,
    5.2379,
    6.0410,
    6.0595,
    7.2973,
    6.8204,
    7.4182,
    6.9960,
    6.8741,
    8.1594,
]
train_acc = [
    0.0297,
    0.0735,
    0.1876,
    0.3456,
    0.4610,
    0.5609,
    0.6202,
    0.6905,
    0.7344,
    0.8001,
    0.8414,
    0.8685,
]
val_acc = [
    0.0000,
    0.0286,
    0.0657,
    0.0571,
    0.0000,
    0.0343,
    0.0000,
    0.0629,
    0.0143,
    0.0429,
    0.0714,
    0.0171,
]
lr = [
    0.000208,
    0.000406,
    0.000604,
    0.000802,
    0.001000,
    0.000976,
    0.000905,
    0.000796,
    0.000658,
    0.000505,
    0.000352,
    0.000214,
]


fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor("#fafafa")

# --- Loss ---
ax = axes[0]
ax.plot(
    epochs,
    train_loss,
    "o-",
    color="#2563eb",
    linewidth=2,
    markersize=5,
    label="Train loss",
)
ax.plot(
    epochs,
    val_loss,
    "s-",
    color="#dc2626",
    linewidth=2,
    markersize=5,
    label="Val loss",
)
ax.axvline(x=2, color="#22c55e", linestyle="--", alpha=0.6, label="Best val_loss (epoch 2)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training & Validation Loss", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Accuracy ---
ax = axes[1]
ax.plot(
    epochs,
    [a * 100 for a in train_acc],
    "o-",
    color="#2563eb",
    linewidth=2,
    markersize=5,
    label="Train acc",
)
ax.plot(
    epochs,
    [a * 100 for a in val_acc],
    "s-",
    color="#dc2626",
    linewidth=2,
    markersize=5,
    label="Val acc",
)
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Training & Validation Accuracy", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- LR Schedule ---
ax = axes[2]
ax.plot(
    epochs,
    [v * 1000 for v in lr],
    "^-",
    color="#7c3aed",
    linewidth=2,
    markersize=5,
)
ax.set_xlabel("Epoch")
ax.set_ylabel("Learning Rate (x1e-3)")
ax.set_title("Cosine LR Schedule (5 warmup)", fontweight="bold")
ax.grid(True, alpha=0.3)

fig.suptitle(
    "Training Run — 45 subjects, batch_size=16, Adam, cosine schedule", fontsize=13, y=1.02
)
plt.tight_layout()
fig.savefig(
    OUT / "training_curves.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="#fafafa",
)
plt.close(fig)
print(f"✓ Saved {OUT / 'training_curves.png'}")
