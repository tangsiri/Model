# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots (projection='3d')

# ============================================================
# Select linear or nonlinear model
# ============================================================
choice = input(
    "Run sensitivity analysis on linear or nonlinear model? (1=linear / 0=nonlinear): "
).strip()
is_linear = (choice == "1")

base_dir = os.path.dirname(os.path.abspath(__file__))
root_model_dir = os.path.join(
    base_dir,
    "Progress_of_LSTM_linear" if is_linear else "Progress_of_LSTM_nonlinear"
)

print(f"\n📌 Running sensitivity analysis for: {os.path.basename(root_model_dir)}")
print(f"📂 Model directory: {root_model_dir}")
print("-" * 70)

# ============================================================
# Read progress.npy for all scenarios
# ============================================================
records = []

for scen_name in os.listdir(root_model_dir):
    scen_dir = os.path.join(root_model_dir, scen_name)
    progress_file = os.path.join(scen_dir, "progress.npy")

    if not os.path.isfile(progress_file):
        print(f"\n🔍 Scenario: {scen_name}")
        print("   ⚠️ progress.npy not found (scenario may not be fully trained).")
        continue

    data = np.load(progress_file, allow_pickle=True).item()

    # Parse parameters from folder name: epXX_Aa.bb_Tc.dd
    try:
        parts = scen_name.split("_")
        epochs_cfg = int(parts[0].replace("ep", ""))
        alpha_val = float(parts[1].replace("A", ""))
        thresh_val = float(parts[2].replace("T", ""))
    except Exception:
        epochs_cfg = alpha_val = thresh_val = None

    print(f"\n🔍 Scenario: {scen_name}")
    print(f"   ✔ Best Val Loss    : {data['best_val_loss']:.6e}")
    print(f"   ✔ Epochs Trained   : {data['epochs_trained']}")
    print(f"   ✔ Final Val Loss   : {data['val_loss'][-1]:.6e}")

    records.append({
        "Scenario": scen_name,
        "Epochs_cfg": epochs_cfg,
        "ALPHA": alpha_val,
        "THRESH": thresh_val,
        "Best_Val_Loss": data["best_val_loss"],
        "Final_Train_Loss": data["train_loss"][-1],
        "Final_Val_Loss": data["val_loss"][-1],
        "Epochs_Trained": data["epochs_trained"],
    })

# ============================================================
# Create DataFrame
# ============================================================
df = pd.DataFrame(records)
print("\n====================== SUMMARY TABLE ======================")
print(df)
print("==========================================================")

if df.empty:
    raise RuntimeError("No valid scenarios found (DataFrame is empty).")

# ============================================================
# Find best scenario based on Best_Val_Loss
# ============================================================
best_idx = df["Best_Val_Loss"].idxmin()
best_row = df.loc[best_idx]

print("\n🏆 Best scenario (minimum Best_Val_Loss):")
print(best_row)

# ============================================================
# 1) Effect of Epochs
# ============================================================
plt.figure(figsize=(6, 4))
group_ep = df.groupby("Epochs_cfg")["Best_Val_Loss"].mean()
plt.plot(group_ep.index, group_ep.values, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Mean Best Val Loss")
plt.title("Effect of Epoch Count on Model Performance")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 2) Effect of ALPHA
# ============================================================
plt.figure(figsize=(6, 4))
group_alpha = df.groupby("ALPHA")["Best_Val_Loss"].mean()
plt.plot(group_alpha.index, group_alpha.values, marker="o")
plt.xlabel("Alpha")
plt.ylabel("Mean Best Val Loss")
plt.title("Effect of Alpha on Model Performance")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 3) Effect of THRESH
# ============================================================
plt.figure(figsize=(6, 4))
group_thresh = df.groupby("THRESH")["Best_Val_Loss"].mean()
plt.plot(group_thresh.index, group_thresh.values, marker="o")
plt.xlabel("Threshold")
plt.ylabel("Mean Best Val Loss")
plt.title("Effect of Threshold on Model Performance")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 4) 2D Heatmap (ALPHA vs THRESH)
# ============================================================
print("\n📡 Building 2D heatmap ...")

heatmap_table = df.pivot_table(
    index="ALPHA",
    columns="THRESH",
    values="Best_Val_Loss",
    aggfunc="mean"
)

plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_table, annot=True, fmt=".2e", cmap="viridis")
plt.title("Sensitivity Heatmap (Best_Val_Loss)")
plt.xlabel("THRESH")
plt.ylabel("ALPHA")
plt.tight_layout()
plt.show()

# ============================================================
# 5) 3D Surface Plot: Best_Val_Loss vs (ALPHA, THRESH) per Epochs_cfg
# ============================================================
print("\n📡 Building 3D surface plots ...")

unique_epochs = sorted(df["Epochs_cfg"].dropna().unique())

for E in unique_epochs:
    sub = df[df["Epochs_cfg"] == E].copy()
    if sub.empty:
        continue

    pivot_3d = sub.pivot_table(
        index="ALPHA",
        columns="THRESH",
        values="Best_Val_Loss",
        aggfunc="mean"
    )

    # Need at least a 2x2 grid for a proper surface
    if pivot_3d.shape[0] < 2 or pivot_3d.shape[1] < 2:
        print(f"⚠️ Not enough (ALPHA, THRESH) combinations for 3D plot at Epochs={E}.")
        continue

    X_vals = pivot_3d.columns.values   # THRESH
    Y_vals = pivot_3d.index.values     # ALPHA
    X, Y = np.meshgrid(X_vals, Y_vals)
    Z = pivot_3d.values

    # Mask NaNs (missing combinations)
    Z_masked = np.ma.masked_invalid(Z)
    if np.all(Z_masked.mask):
        print(f"⚠️ All values are NaN for Epochs={E}, skipping 3D plot.")
        continue

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z_masked, cmap="viridis", edgecolor="none")

    ax.set_xlabel("THRESH")
    ax.set_ylabel("ALPHA")
    ax.set_zlabel("Best_Val_Loss")
    ax.set_title(f"3D Surface of Best_Val_Loss (Epochs={E})")

    fig.colorbar(surf, shrink=0.5, aspect=10, label="Best_Val_Loss")
    plt.tight_layout()
    plt.show()

print("\n🎉 Sensitivity analysis completed successfully!")
