# -*- coding: utf-8 -*-
"""
File name      :
Author         : pc22

Created on     : Wed Dec 24 09:08:14 2025
Last modified  : Wed Dec 24 09:08:14 2025

------------------------------------------------------------
Purpose:
------------------------------------------------------------
To prepare fixed GM–THA datasets for LSTM-based seismic response
modeling and optionally apply global clustering and balancing
of ground motion records to reduce dataset bias across heights.

------------------------------------------------------------
Description:
------------------------------------------------------------
This script implements a unified data-preparation pipeline with
two main stages:

1) Fixed_Files (Code-1):
   - Reads ground motion (GM) and time-history analysis (THA)
     displacement outputs.
   - Removes the first column from disp.txt (time column).
   - Aligns GM and THA records to their minimum common length.
   - Saves per-height datasets as X_data_H*.npy and Y_data_H*.npy
     dictionaries.

2) Global_Clustering_Balancing (Code-2):
   - Extracts GM features (PGA, RMS, duration ratio, frequency index)
     from a reference height.
   - Applies K-Means clustering and selects an equal number of
     records per cluster.
   - Applies the selected record IDs consistently to all heights.
   - Saves balanced datasets in cluster_balanced_global folders
     without modifying original fixed outputs.

A controller section at the end allows the user to enable or
disable clustering while preserving all original paths and
file naming conventions.

------------------------------------------------------------
Inputs:
------------------------------------------------------------
- Output/1_IDA_Records_{train|predict}/zire ham/*.txt
- Output/2_THA_{train}_{linear|nonlinear}/H*/**/disp.txt
- User inputs:
    * mode (train / predict)
    * linear or nonlinear analysis
    * list of structural heights
    * number of clusters K (for clustering mode)

------------------------------------------------------------
Outputs:
------------------------------------------------------------
Fixed (always generated):
- Output/3_GM_Fixed_{mode}_{linear|nonlinear}/H*/X_data_H*.npy
- Output/3_THA_Fixed_{mode}_{linear|nonlinear}/H*/Y_data_H*.npy

Clustered (train mode only, optional):
- Output/3_GM_Fixed_train_{linear|nonlinear}/H*/cluster_balanced_global/
  X_data_cluster_balanced_global_H*.npy
- Output/3_THA_Fixed_train_{linear|nonlinear}/H*/cluster_balanced_global/
  Y_data_cluster_balanced_global_H*.npy

Additional outputs:
- Cluster analysis figures (bar charts and scatter plots)
  saved under:
  Output/3_GM_Fixed_train_{linear|nonlinear}/cluster_analysis/

------------------------------------------------------------
Changes since previous version:
------------------------------------------------------------
- Integrated Fixed_Files (Code-1) and global clustering (Code-2)
  into a single controlled pipeline.
- Added user-controlled option to enable/disable clustering.
- Preserved all original output paths and file names.

------------------------------------------------------------
Impact of changes:
------------------------------------------------------------
- Enables fair and balanced GM selection across all heights.
- Maintains full compatibility with existing LSTM training
  and prediction scripts.
- Improves reproducibility and traceability of dataset preparation.

------------------------------------------------------------
Status:
------------------------------------------------------------
- Development

------------------------------------------------------------
Notes:
------------------------------------------------------------
- Clustering is applied only in train mode to preserve original
  Code-2 directory structure.
- Structural height is NOT used as a clustering feature.
- Selected GM record IDs are enforced consistently across all heights.
"""



import os
import glob
import time
import numpy as np
import random
from tqdm import tqdm  # pip install tqdm

from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# ============================================================
# -------------------- Code-1: Fixed_Files -------------------
# (Same logic, same paths)
# ============================================================
def Fixed_Files(dataDir, mode, is_linear, heights=None, min_lines=None):
    """
    Prepare fixed datasets for LSTM by height:
    - Collect disp.txt files and remove the first column (e.g., time)
    - Copy GM files
    - Align X (GM) and Y (THA) by min length per record
    - Save X_data_H*.npy and Y_data_H*.npy as dict: {record_id.txt: np.array([...])}
    - Remove intermediate *.txt files so only *.npy remain
    """

    start_time = time.time()
    dataDir = os.path.abspath(dataDir)

    # ---------------------------
    # train/predict and linear/nonlinear
    # ---------------------------
    mode = mode.strip().lower()
    if mode not in ["train", "predict"]:
        raise ValueError("mode must be 'train' or 'predict'.")

    lin_str = "linear" if is_linear else "nonlinear"

    # IDA depends only on mode
    gm_input_dir = os.path.join(dataDir, 'Output', f'1_IDA_Records_{mode}', 'zire ham')

    # THA depends on mode and lin_str
    tha_root_dir = os.path.join(dataDir, 'Output', f'2_THA_{mode}_{lin_str}')

    # fixed outputs roots
    gm_fixed_root = os.path.join(dataDir, 'Output', f'3_GM_Fixed_{mode}_{lin_str}')
    tha_fixed_root = os.path.join(dataDir, 'Output', f'3_THA_Fixed_{mode}_{lin_str}')

    print("\n===============================")
    print(f"📌 mode      = {mode}")
    print(f"📌 is_linear = {is_linear}  ({lin_str})")
    print(f"📥 GM input  = {gm_input_dir}")
    print(f"📥 THA root  = {tha_root_dir}")
    print(f"📤 GM Fixed  = {gm_fixed_root}")
    print(f"📤 THA Fixed = {tha_fixed_root}")
    print("===============================\n")

    # ---------------------------
    # helpers
    # ---------------------------
    def clear_or_make(folder_path, pattern='*.txt'):
        """Delete matching files if folder exists; otherwise create folder."""
        if os.path.exists(folder_path):
            for f in glob.glob(os.path.join(folder_path, pattern)):
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"⚠️ Failed to delete {f}: {e}")
        else:
            os.makedirs(folder_path, exist_ok=True)

    # 0) build Merge_disp-files by copying disp.txt and removing first column
    def merge_disp_files(tha_dir, tha_merge_dir):
        os.makedirs(tha_merge_dir, exist_ok=True)
        disp_files = glob.glob(os.path.join(tha_dir, '**', 'disp.txt'), recursive=True)
        print(f"🔍 Found {len(disp_files)} disp.txt files in {tha_dir}. Merging...")

        for file_path in tqdm(disp_files, desc="📄 Merging disp-files (remove first column)"):
            try:
                folder_name = os.path.basename(os.path.dirname(file_path))
                output_file = os.path.join(tha_merge_dir, f"{folder_name}.txt")

                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                processed = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 1:
                        processed.append(' '.join(parts[1:]) + '\n')

                with open(output_file, 'w', encoding='utf-8') as out_f:
                    out_f.writelines(processed)

            except Exception as e:
                print(f"⚠️ Error processing {file_path}: {e}")

        print(f"✅ Merged disp outputs saved in {tha_merge_dir}\n")

    # 1) copy GM files without trimming
    def copy_gm_files(input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        files = glob.glob(os.path.join(input_dir, '*.txt'))
        print(f"🔍 Found {len(files)} GM files in {input_dir}. Copying...")

        for file in tqdm(files, desc="📂 Copy GM (no trimming)"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                file_name = os.path.basename(file).replace('_for_ML', '')
                with open(os.path.join(output_dir, file_name), 'w', encoding='utf-8') as f_out:
                    f_out.writelines(lines)
            except Exception as e:
                print(f"⚠️ Error copying {file}: {e}")

    # 2) align each GM–THA pair to minimum length
    def align_pairs_to_min_len(gm_fixed_dir, tha_merge_dir, tha_fixed_dir):
        os.makedirs(tha_fixed_dir, exist_ok=True)
        for gm_file in tqdm(glob.glob(os.path.join(gm_fixed_dir, '*.txt')),
                            desc="🔄 Aligning X/Y record lengths"):
            file_name = os.path.basename(gm_file)
            tha_file_path = os.path.join(tha_merge_dir, file_name)

            if not os.path.exists(tha_file_path):
                continue

            try:
                with open(gm_file, 'r', encoding='utf-8') as f:
                    gm_lines = f.readlines()
                with open(tha_file_path, 'r', encoding='utf-8') as f:
                    tha_lines = f.readlines()

                L = min(len(gm_lines), len(tha_lines))
                if L == 0:
                    try:
                        os.remove(gm_file)
                    except Exception:
                        pass
                    continue

                gm_trim = gm_lines[:L]
                tha_trim = tha_lines[:L]

                with open(gm_file, 'w', encoding='utf-8') as f:
                    f.writelines(gm_trim)
                with open(os.path.join(tha_fixed_dir, file_name), 'w', encoding='utf-8') as f:
                    f.writelines(tha_trim)

            except Exception as e:
                print(f"⚠️ Alignment error for {file_name}: {e}")

    # 3) remove empty text files
    def remove_empty_files(gm_fixed_dir, tha_fixed_dir):
        for folder in [gm_fixed_dir, tha_fixed_dir]:
            for file in tqdm(glob.glob(os.path.join(folder, '*.txt')),
                             desc=f"🗑 Removing empty files in {os.path.basename(folder)}"):
                try:
                    if os.stat(file).st_size == 0:
                        os.remove(file)
                except Exception as e:
                    print(f"⚠️ Failed to remove {file}: {e}")

    # 4) save dict as npy
    def save_numpy_data(input_dir, output_file):
        data_dict = {}
        for file in tqdm(glob.glob(os.path.join(input_dir, '*.txt')),
                         desc=f"📥 Building {os.path.basename(output_file)}"):
            file_name = os.path.basename(file)
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    arr = []
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        arr.append(float(parts[0]))  # single channel
                    data = np.array(arr, dtype=np.float32)
                if data.size > 0:
                    data_dict[file_name] = data
            except Exception as e:
                print(f"⚠️ Read error {file}: {e}")

        if not data_dict:
            print(f"⚠️ No data found to save in {output_file}.")
        else:
            np.save(output_file, data_dict)
            print(f"✅ Saved {output_file} ({len(data_dict)} records)\n")

    # auto-detect heights if not provided
    if not heights:
        heights = []
        if os.path.isdir(tha_root_dir):
            for name in sorted(os.listdir(tha_root_dir)):
                full = os.path.join(tha_root_dir, name)
                if os.path.isdir(full) and name.startswith("H"):
                    h_str = name[1:].replace('p', '.')
                    try:
                        heights.append(float(h_str))
                    except ValueError:
                        pass

    if not heights:
        heights = [3.0]

    print(f"📏 Heights to process: {', '.join(str(h) for h in heights)}\n")

    # run pipeline for each height
    for h in heights:
        if float(h).is_integer():
            h_tag = f"H{int(h)}"
        else:
            h_tag = "H" + str(h).replace('.', 'p')

        tha_dir = os.path.join(tha_root_dir, h_tag)
        if not os.path.isdir(tha_dir):
            print(f"⚠️ THA folder not found for height {h}: {tha_dir} → skipped.\n")
            continue

        tha_merge_dir = os.path.join(tha_dir, 'Merge_disp-files')
        gm_fixed_dir = os.path.join(gm_fixed_root, h_tag)
        tha_fixed_dir = os.path.join(tha_fixed_root, h_tag)

        print("--------------------------------------------------")
        print(f"🏗️ Processing height H = {h} m")
        print(f"📥 THA dir       = {tha_dir}")
        print(f"📥 THA merge dir = {tha_merge_dir}")
        print(f"📤 GM Fixed dir  = {gm_fixed_dir}")
        print(f"📤 THA Fixed dir = {tha_fixed_dir}")
        print("--------------------------------------------------\n")

        merge_disp_files(tha_dir, tha_merge_dir)
        clear_or_make(gm_fixed_dir)
        clear_or_make(tha_fixed_dir)
        copy_gm_files(gm_input_dir, gm_fixed_dir)
        align_pairs_to_min_len(gm_fixed_dir, tha_merge_dir, tha_fixed_dir)
        remove_empty_files(gm_fixed_dir, tha_fixed_dir)

        x_out = os.path.join(gm_fixed_dir, f'X_data_{h_tag}.npy')
        y_out = os.path.join(tha_fixed_dir, f'Y_data_{h_tag}.npy')

        save_numpy_data(gm_fixed_dir, x_out)
        save_numpy_data(tha_fixed_dir, y_out)

        clear_or_make(gm_fixed_dir, pattern='*.txt')
        clear_or_make(tha_fixed_dir, pattern='*.txt')

    total_time = round(time.time() - start_time, 2)
    print(f"\n✅ Fixed processing completed. Total time: {total_time} seconds\n")


# ============================================================
# -------------------- Code-2: Clustering --------------------
# (Same logic, same paths)
# ============================================================
def extract_features_from_record(x):
    """Return [PGA, RMS, Duration_ratio, Frequency_index] from a 1D GM record."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        x = x.reshape(-1)

    pga = float(np.max(np.abs(x)) + 1e-12)
    rms = float(np.sqrt(np.mean(x**2)))

    thr = 0.05 * pga
    duration_ratio = float(np.sum(np.abs(x) >= thr) / len(x))

    signs = np.sign(x)
    signs[signs == 0] = 1
    freq_index = float(np.sum(np.diff(signs) != 0) / len(x))

    return [pga, rms, duration_ratio, freq_index]


def Global_Clustering_Balancing(root_dir, is_linear=True, seed=1234):
    """
    Global clustering & balancing across all heights.
    NOTE: Exactly matches your original Code-2 behavior and paths (train-only).
    """
    random.seed(seed)
    np.random.seed(seed)

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

    height_folders = sorted([
        d for d in os.listdir(gm_root_dir)
        if d.lower().startswith("h") and os.path.isdir(os.path.join(gm_root_dir, d))
    ])

    if not height_folders:
        raise RuntimeError("❌ No height folders (H*) found inside GM root dir.")

    print("\n📏 Available heights:")
    for h in height_folders:
        print("   →", h)

    ref_height = height_folders[0]
    print(f"\n📌 Reference height for clustering GM records: {ref_height}")

    # Ask K
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

    feature_list, valid_keys = [], []
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

    scaler = StandardScaler()
    feature_scaled = scaler.fit_transform(feature_arr)

    print(f"\n🚀 Running K-Means with K = {K} on reference height {ref_height} ...")
    kmeans = KMeans(n_clusters=K, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(feature_scaled)

    cluster_members = defaultdict(list)
    for k, lab in zip(valid_keys, labels):
        cluster_members[lab].append(k)

    print("\n📦 Record distribution among clusters (reference height):")
    for c in range(K):
        n_c = np.sum(labels == c)
        print(f"   Cluster {c}: {n_c} records")

    # Figures (exact behavior preserved)
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.unicode_minus"] = False

    analysis_dir = os.path.join(gm_root_dir, "cluster_analysis")
    os.makedirs(analysis_dir, exist_ok=True)

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

    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#999999"]
    colors = (colors * ((K // len(colors)) + 1))[:K]

    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    plt.figure(figsize=(9, 6), dpi=200)
    for c in range(K):
        idx = (labels == c)
        plt.scatter(feature_arr[idx, 0], feature_arr[idx, 1], s=8, alpha=0.6, color=colors[c], label=f"Cluster {c}")
    plt.scatter(centers[:, 0], centers[:, 1], s=160, marker="X", color="black",
                edgecolors="white", linewidth=1.5, label="Cluster Center")
    plt.xlabel("PGA (Peak Ground Acceleration)", fontsize=14)
    plt.ylabel("RMS (Root Mean Square of Acceleration)", fontsize=14)
    plt.title("PGA–RMS Feature Clustering", fontsize=16, weight="bold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=12)
    plt.tight_layout()
    scatter_pga_rms_path = os.path.join(analysis_dir, "PGA_RMS_Clusters_with_Centers.png")
    plt.savefig(scatter_pga_rms_path, dpi=300)
    plt.close()
    print("📁 PGA–RMS scatter saved to:")
    print("   ", scatter_pga_rms_path)

    plt.figure(figsize=(9, 6), dpi=200)
    for c in range(K):
        idx = (labels == c)
        plt.scatter(feature_arr[idx, 2], feature_arr[idx, 3], s=8, alpha=0.6, color=colors[c], label=f"Cluster {c}")
    plt.scatter(centers[:, 2], centers[:, 3], s=160, marker="X", color="black",
                edgecolors="white", linewidth=1.5, label="Cluster Center")
    plt.xlabel("Duration Ratio (|a(t)| ≥ 0.05·PGA)", fontsize=14)
    plt.ylabel("Frequency Index (Zero-Crossing Rate)", fontsize=14)
    plt.title("Duration–Frequency Feature Clustering", fontsize=16, weight="bold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=12)
    plt.tight_layout()
    scatter_dur_freq_path = os.path.join(analysis_dir, "Duration_Freq_Clusters_with_Centers.png")
    plt.savefig(scatter_dur_freq_path, dpi=300)
    plt.close()
    print("📁 Duration–Frequency scatter saved to:")
    print("   ", scatter_dur_freq_path)

    for c in range(K):
        plt.figure(figsize=(7, 5), dpi=200)
        idx = (labels == c)
        plt.scatter(feature_arr[idx, 0], feature_arr[idx, 1], s=10, alpha=0.7, color=colors[c])
        plt.scatter(centers[c, 0], centers[c, 1], color="black", marker="X", s=200, linewidth=2)
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


# ============================================================
# ------------------------ MAIN (Controller) ------------------
# Only controls the run flow (no path changes)
# ============================================================
if __name__ == "__main__":
    # Same assumption as your Code-1: file is in Model/, project root is one level up
    data_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    print("=== Pipeline (Fixed + Optional Global Clustering) ===")

    # 0) cluster?
    c = input("آیا کلاستر می‌خواهی؟ (1=بله / 0=خیر): ").strip()
    use_cluster = (c == "1")

    # 1) train or predict
    choice = input("برای train عدد 0 و برای predict عدد 1 را وارد کن: ").strip()
    if choice == "0":
        mode = "train"
    elif choice == "1":
        mode = "predict"
    else:
        print("❌ فقط 0 یا 1 مجاز است.")
        raise SystemExit

    # 2) linear or nonlinear
    choice_lin = input("مدل خطی است یا غیرخطی؟ برای خطی 1 و برای غیرخطی 0 را وارد کن: ").strip()
    if choice_lin == "1":
        is_linear = True
    elif choice_lin == "0":
        is_linear = False
    else:
        print("❌ فقط 0 یا 1 مجاز است.")
        raise SystemExit

    # 3) heights
    heights_raw = input("ارتفاع ستون‌ها را وارد کن (مثلاً: 3 یا 3 4 5): ").strip()
    heights = []
    if heights_raw:
        for token in heights_raw.replace(',', ' ').split():
            try:
                heights.append(float(token))
            except ValueError:
                print(f"⚠️ مقدار «{token}» عدد معتبری نیست و نادیده گرفته می‌شود.")
    if not heights:
        print("⚠️ هیچ ارتفاع معتبری وارد نشد؛ ارتفاع پیش‌فرض 3 متر استفاده می‌شود.")
        heights = [3.0]

    # Always run Fixed_Files first (required for both paths)
    Fixed_Files(data_directory, mode=mode, is_linear=is_linear, heights=heights)

    # Run clustering only if selected AND mode is train (to preserve exact Code-2 paths)
    if use_cluster:
        if mode != "train":
            print("\n⚠️ کلاستر فقط برای train در کد دوم شما تعریف شده است (مسیرهای train_...).")
            print("✅ بنابراین برای حفظ دقیق مسیرها، در حالت predict کلاستر اجرا نشد.")
        else:
            Global_Clustering_Balancing(data_directory, is_linear=is_linear, seed=1234)
    else:
        print("\n✅ کلاستر انتخاب نشد؛ فقط خروجی‌های Fixed ساخته شدند.")
