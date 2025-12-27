# -*- coding: utf-8 -*-
"""
File name      : step3_fixed_files-v01.py
Author         : pc22
Created on     : Wed Dec 24 09:30:18 2025
Last modified  : Wed Dec 24 09:30:18 2025

------------------------------------------------------------
Purpose:
------------------------------------------------------------
To perform global clustering and balancing of ground-motion (GM) records
for LSTM-based seismic response modeling, and then apply the same selected
GM record IDs consistently across ALL structural heights.

------------------------------------------------------------
Description:
------------------------------------------------------------
This script assumes that the ground-motion record set is identical for all
heights (H2, H3, H4, ...). Therefore, it uses the first available height
folder as a REFERENCE height to cluster GM records, select an equal number
of records per cluster (balanced subset), and then applies those selected
record IDs to build balanced datasets for every height.

Main steps:
1) Detect available height folders (H*) in the fixed GM/THA train outputs.
2) Choose the first height folder as the reference height (same GM set for all).
3) Ask the user for the number of clusters K.
4) For the reference height:
   - Load X_data_H*.npy and Y_data_H*.npy dictionaries.
   - Extract GM features per record: PGA, RMS, Duration_ratio, Frequency_index.
   - Standardize features (StandardScaler).
   - Run KMeans clustering.
   - Print cluster statistics (mean/std/center in original scale).
   - Save cluster analysis figures (bar + scatter plots) to cluster_analysis/.
5) Create a balanced subset by selecting the same number of records from each cluster
   (target = minimum cluster size).
6) For ALL heights:
   - Load X_data and Y_data.
   - Intersect selected record IDs with available keys in that height.
   - Save balanced dictionaries to cluster_balanced_global/ without modifying
     the original fixed outputs.

------------------------------------------------------------
Inputs:
------------------------------------------------------------
- Fixed datasets generated previously (TRAIN only):

If is_linear = True:
- Output/3_GM_Fixed_train_linear/H*/X_data_H*.npy
- Output/3_THA_Fixed_train_linear/H*/Y_data_H*.npy

If is_linear = False:
- Output/3_GM_Fixed_train_nonlinear/H*/X_data_H*.npy
- Output/3_THA_Fixed_train_nonlinear/H*/Y_data_H*.npy

- User input:
  * Number of clusters K (integer >= 1)

------------------------------------------------------------
Outputs:
------------------------------------------------------------
(1) Cluster analysis figures:
- Output/3_GM_Fixed_train_{linear|nonlinear}/cluster_analysis/
  * cluster_counts_bar.png
  * PGA_RMS_Clusters_with_Centers.png
  * Duration_Freq_Clusters_with_Centers.png
  * PGA_RMS_Cluster_<c>.png  (for each cluster c)

(2) Balanced datasets for each height (new folders; original fixed files unchanged):
- Output/3_GM_Fixed_train_{linear|nonlinear}/H*/cluster_balanced_global/
    X_data_cluster_balanced_global_H*.npy
- Output/3_THA_Fixed_train_{linear|nonlinear}/H*/cluster_balanced_global/
    Y_data_cluster_balanced_global_H*.npy

------------------------------------------------------------
Changes since previous version:
------------------------------------------------------------
- 

------------------------------------------------------------
Impact of changes:
------------------------------------------------------------
- 

------------------------------------------------------------
Status:
------------------------------------------------------------
- Stable

------------------------------------------------------------
Notes:
------------------------------------------------------------
- The clustering is performed ONLY on the reference height because the GM record
  set is assumed identical across heights; selected record IDs are then reused
  for all heights to ensure consistent balancing.
- Height itself is NOT used as a clustering feature.
- Selection is reproducible via fixed SEED=1234 (random + numpy).
- This script does NOT delete or overwrite original X_data_H*.npy / Y_data_H*.npy;
  it only creates additional balanced outputs in cluster_balanced_global/.
"""
# (Source: user uploaded code) :contentReference[oaicite:0]{index=0}

import os
import numpy as np
import random
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ============================================================
# Basic settings & seeds
# ============================================================
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(base_dir, os.pardir)

# ------------------------------------------------------------
# Choose linear or nonlinear dataset
# ------------------------------------------------------------
is_linear = True  # set to False if you want to use nonlinear THA data

if is_linear:
    gm_root_dir = os.path.join(root_dir, "Output", "3_GM_Fixed_train_linear")
    tha_root_dir = os.path.join(root_dir, "Output", "3_THA_Fixed_train_linear")
else:
    gm_root_dir = os.path.join(root_dir, "Output", "3_GM_Fixed_train_nonlinear")
    tha_root_dir = os.path.join(root_dir, "Output", "3_THA_Fixed_train_nonlinear")

print("📂 GM root dir :", gm_root_dir)
print("📂 THA root dir:", tha_root_dir)

if not os.path.isdir(gm_root_dir):
    raise FileNotFoundError(f"❌ GM root dir not found: {gm_root_dir}")
if not os.path.isdir(tha_root_dir):
    raise FileNotFoundError(f"❌ THA root dir not found: {tha_root_dir}")

# ============================================================
# Find available height folders: H2, H3, H4, ...
# ============================================================
height_folders = sorted([
    d for d in os.listdir(gm_root_dir)
    if d.lower().startswith("h") and os.path.isdir(os.path.join(gm_root_dir, d))
])

if not height_folders:
    raise RuntimeError("❌ No height folders (H*) found inside GM root dir.")

print("\n📏 Available heights:")
for h in height_folders:
    print("   →", h)

# Use first available height as reference (GM records are the same for all)
ref_height = height_folders[0]
print(f"\n📌 Reference height for clustering GM records: {ref_height}")

# ============================================================
# Ask user for number of clusters K
# ============================================================
while True:
    try:
        k_input = input("\nPlease enter the number of clusters K (e.g., 4): ").strip()
        K = int(k_input)
        if K < 1:
            print("⚠️  K must be at least 1.")
            continue
        break
    except ValueError:
        print("⚠️  Please enter an integer value for K.")

print(f"✅ Using K = {K} clusters.\n")

# ============================================================
# Feature extraction function
# ============================================================
def extract_features_from_record(x):
    """
    Extract scalar features from a 1D ground motion record x(t).

    Returns:
        [ PGA,
          RMS,
          Duration_ratio,
          Frequency_index ]
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        x = x.reshape(-1)

    # 1) Peak Ground Acceleration (PGA)
    pga = float(np.max(np.abs(x)) + 1e-12)

    # 2) Root Mean Square (RMS) of acceleration
    rms = float(np.sqrt(np.mean(x**2)))

    # 3) Duration ratio: fraction of samples with |a(t)| >= 0.05 * PGA
    thr = 0.05 * pga
    duration_ratio = float(np.sum(np.abs(x) >= thr) / len(x))

    # 4) Frequency index: zero-crossing rate (normalized by length)
    signs = np.sign(x)
    signs[signs == 0] = 1
    freq_index = float(np.sum(np.diff(signs) != 0) / len(x))

    return [pga, rms, duration_ratio, freq_index]

# ============================================================
# STEP 1: Clustering using reference height
# ============================================================
ref_x_file = os.path.join(gm_root_dir, ref_height, f"X_data_{ref_height}.npy")
ref_y_file = os.path.join(tha_root_dir, ref_height, f"Y_data_{ref_height}.npy")

print("=== STEP 1: Clustering on reference height ===")
print("📂 X (ref):", ref_x_file)
print("📂 Y (ref):", ref_y_file)

if not os.path.exists(ref_x_file) or not os.path.exists(ref_y_file):
    raise FileNotFoundError("❌ Reference X or Y file not found.")

X_ref = np.load(ref_x_file, allow_pickle=True).item()
Y_ref = np.load(ref_y_file, allow_pickle=True).item()

ref_keys = sorted(set(X_ref.keys()) & set(Y_ref.keys()))
print(f"📊 Number of common GM–THA records in reference height ({ref_height}): {len(ref_keys)}")

if len(ref_keys) < 5:
    raise RuntimeError("❌ Not enough records for clustering in reference height.")

# Extract features for all records at reference height
feature_list = []
valid_keys = []

for k in ref_keys:
    try:
        feats = extract_features_from_record(X_ref[k])
        feature_list.append(feats)
        valid_keys.append(k)
    except Exception as e:
        print(f"⚠️ Feature extraction failed for record {k}: {e}")

feature_arr = np.array(feature_list, dtype=float)
print("📊 Feature matrix shape (reference):", feature_arr.shape)

if feature_arr.shape[0] < K:
    print(f"⚠️ Number of records ({feature_arr.shape[0]}) is smaller than K.")
    K = max(1, feature_arr.shape[0] // 2)
    print(f"   → K reduced to {K}.")

# Standardize features before clustering
scaler = StandardScaler()
feature_scaled = scaler.fit_transform(feature_arr)

print(f"\n🚀 Running K-Means with K = {K} on reference height {ref_height} ...")
kmeans = KMeans(n_clusters=K, random_state=SEED, n_init=10)
labels = kmeans.fit_predict(feature_scaled)

# Group record IDs by cluster
cluster_members = defaultdict(list)
for k, lab in zip(valid_keys, labels):
    cluster_members[lab].append(k)

print("\n📦 Record distribution among clusters (reference height):")
for c in range(K):
    n_c = np.sum(labels == c)
    print(f"   Cluster {c}: {n_c} records")

# ============================================================
# STEP 2: Numerical cluster summary (for reporting)
# ============================================================
print("\n📊 Statistical summary of features per cluster (original scale):")
cluster_centers_orig = scaler.inverse_transform(kmeans.cluster_centers_)
feature_names = ["PGA", "RMS", "Duration_ratio", "Frequency_index"]

for c in range(K):
    idx_c = (labels == c)
    feats_c = feature_arr[idx_c]
    n_c = feats_c.shape[0]
    print(f"\n----- Cluster {c} (N = {n_c}) -----")
    for j, fname in enumerate(feature_names):
        mean_val = np.mean(feats_c[:, j])
        std_val  = np.std(feats_c[:, j])
        center   = cluster_centers_orig[c, j]
        print(f"{fname}: mean = {mean_val:.4f} | std = {std_val:.4f} | center = {center:.4f}")

# ============================================================
# STEP 3: Visualization – bar chart & scatter plots
# ============================================================
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

analysis_dir = os.path.join(gm_root_dir, "cluster_analysis")
os.makedirs(analysis_dir, exist_ok=True)

# --- 3.1) Bar chart: number of records in each cluster ---
cluster_counts = [np.sum(labels == c) for c in range(K)]

plt.figure(figsize=(9, 6), dpi=200)
plt.bar(range(K), cluster_counts, color="#0072B2", edgecolor="black", linewidth=1.2)
plt.xlabel("Cluster Index", fontsize=14)
plt.ylabel("Number of Records", fontsize=14)
plt.title("Record Distribution over GM Clusters", fontsize=16, weight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.xticks(range(K), [f"C{c}" for c in range(K)], fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

bar_path = os.path.join(analysis_dir, "cluster_counts_bar.png")
plt.savefig(bar_path, dpi=300)
plt.close()
print("\n📁 Bar chart saved to:")
print("   ", bar_path)

# --- colors for clusters (fixed & discrete) ---
colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#999999"]
# if K > len(colors), they will be reused cyclically
colors = (colors * ((K // len(colors)) + 1))[:K]

# ------------------------------------------------------------
# 3.2) Scatter: PGA–RMS with cluster centers and legend
# ------------------------------------------------------------
centers = scaler.inverse_transform(kmeans.cluster_centers_)

plt.figure(figsize=(9, 6), dpi=200)
for c in range(K):
    idx = (labels == c)
    plt.scatter(
        feature_arr[idx, 0], feature_arr[idx, 1],
        s=8,
        alpha=0.6,
        color=colors[c],
        label=f"Cluster {c}"
    )

# cluster centers
plt.scatter(
    centers[:, 0], centers[:, 1],
    s=160,
    marker="X",
    color="black",
    edgecolors="white",
    linewidth=1.5,
    label="Cluster Center"
)

plt.xlabel("PGA (Peak Ground Acceleration)", fontsize=14)
plt.ylabel("RMS (Root Mean Square of Acceleration)", fontsize=14)
plt.title("PGA–RMS Feature Clustering", fontsize=16, weight="bold")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

scatter_pga_rms_path = os.path.join(analysis_dir, "PGA_RMS_Clusters_with_Centers.png")
plt.savefig(scatter_pga_rms_path, dpi=300)
plt.close()
print("📁 PGA–RMS scatter (with cluster centers) saved to:")
print("   ", scatter_pga_rms_path)

# ------------------------------------------------------------
# 3.3) Scatter: Duration_ratio–Frequency_index with centers
# ------------------------------------------------------------
plt.figure(figsize=(9, 6), dpi=200)
for c in range(K):
    idx = (labels == c)
    plt.scatter(
        feature_arr[idx, 2], feature_arr[idx, 3],
        s=8,
        alpha=0.6,
        color=colors[c],
        label=f"Cluster {c}"
    )

plt.scatter(
    centers[:, 2], centers[:, 3],
    s=160,
    marker="X",
    color="black",
    edgecolors="white",
    linewidth=1.5,
    label="Cluster Center"
)

plt.xlabel("Duration Ratio (|a(t)| ≥ 0.05·PGA)", fontsize=14)
plt.ylabel("Frequency Index (Zero-Crossing Rate)", fontsize=14)
plt.title("Duration–Frequency Feature Clustering", fontsize=16, weight="bold")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

scatter_dur_freq_path = os.path.join(analysis_dir, "Duration_Freq_Clusters_with_Centers.png")
plt.savefig(scatter_dur_freq_path, dpi=300)
plt.close()
print("📁 Duration–Frequency scatter (with cluster centers) saved to:")
print("   ", scatter_dur_freq_path)

# ------------------------------------------------------------
# 3.4) Per-cluster detailed PGA–RMS scatter figures
# ------------------------------------------------------------
for c in range(K):
    plt.figure(figsize=(7, 5), dpi=200)
    idx = (labels == c)
    plt.scatter(
        feature_arr[idx, 0], feature_arr[idx, 1],
        s=10,
        alpha=0.7,
        color=colors[c]
    )
    plt.scatter(
        centers[c, 0], centers[c, 1],
        color="black",
        marker="X",
        s=200,
        linewidth=2
    )
    plt.xlabel("PGA (Peak Ground Acceleration)", fontsize=12)
    plt.ylabel("RMS (Root Mean Square of Acceleration)", fontsize=12)
    plt.title(f"PGA–RMS Distribution for Cluster {c}", fontsize=14, weight="bold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = os.path.join(analysis_dir, f"PGA_RMS_Cluster_{c}.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"📁 Detailed PGA–RMS figure saved for Cluster {c}:")
    print("   ", out)

# ============================================================
# STEP 4: Select equal number of records from each cluster
# ============================================================
target_per_cluster = min(len(cluster_members[c]) for c in cluster_members)
print(f"\n🎯 Target number of records per cluster (balanced): {target_per_cluster}")

selected_keys = []
for c in cluster_members:
    lst = cluster_members[c][:]
    random.shuffle(lst)
    chosen = lst[:target_per_cluster]
    print(f"   → Cluster {c}: {len(chosen)} records selected.")
    selected_keys.extend(chosen)

selected_keys = sorted(selected_keys)
print(f"\n✅ Total number of selected records (used for all heights): {len(selected_keys)}")

# ============================================================
# STEP 5: Apply selected record IDs to ALL heights and save
# ============================================================
print("\n=== STEP 5: Building balanced datasets for all heights ===")

for height in height_folders:
    print("\n" + "-" * 70)
    print(f"📌 Height: {height}")

    x_file = os.path.join(gm_root_dir,  height, f"X_data_{height}.npy")
    y_file = os.path.join(tha_root_dir, height, f"Y_data_{height}.npy")

    if not os.path.exists(x_file):
        print(f"⚠️ X_data not found for {height}: {x_file} → skipped.")
        continue
    if not os.path.exists(y_file):
        print(f"⚠️ Y_data not found for {height}: {y_file} → skipped.")
        continue

    X_dict = np.load(x_file, allow_pickle=True).item()
    Y_dict = np.load(y_file, allow_pickle=True).item()

    keys_xy = set(X_dict.keys()) & set(Y_dict.keys())
    keys_final = sorted(set(selected_keys) & keys_xy)

    print(f"   Number of usable records for this height: {len(keys_final)}")

    if len(keys_final) == 0:
        print("   ⚠️ No overlapping keys between selected_keys and this height. Skipped.")
        continue

    X_bal = {k: X_dict[k] for k in keys_final}
    Y_bal = {k: Y_dict[k] for k in keys_final}

    out_gm_dir  = os.path.join(gm_root_dir,  height, "cluster_balanced_global")
    out_tha_dir = os.path.join(tha_root_dir, height, "cluster_balanced_global")
    os.makedirs(out_gm_dir,  exist_ok=True)
    os.makedirs(out_tha_dir, exist_ok=True)

    x_out = os.path.join(out_gm_dir,  f"X_data_cluster_balanced_global_{height}.npy")
    y_out = os.path.join(out_tha_dir, f"Y_data_cluster_balanced_global_{height}.npy")

    np.save(x_out, X_bal)
    np.save(y_out, Y_bal)

    print("   💾 Balanced files saved:")
    print("      ", x_out)
    print("      ", y_out)

print("\n🎉 Global clustering, balancing, and figure generation completed successfully.")
print("📂 All analysis figures are stored in:", analysis_dir)
