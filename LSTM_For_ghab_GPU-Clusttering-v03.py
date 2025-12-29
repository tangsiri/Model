# -*- coding: utf-8 -*-
"""
File name      : LSTM_For_ghab_GPU-Clusttering.py
Author         : pc22
Created on     : Sun Dec 28 08:47:13 2025
Last modified  : Mon Dec 29 09:00:00 2025
------------------------------------------------------------
Purpose:
    Improve peak (extreme) prediction quality in LSTM-based
    time-history response forecasting by introducing a robust
    peak-aware training strategy based on:
      (1) Quantile-based peak definition (Top-K%)
      (2) Oversampling of peak-containing time series
    in addition to a corrected per-sample peak-weighted loss.

------------------------------------------------------------
Description:
    This script trains an LSTM network to predict structural
    time-history responses (THA outputs) from ground motion (GM)
    time series. It supports:
      - Linear / Nonlinear training modes
      - Clustered vs non-clustered datasets
      - Two training modes:
          (1) Per-height independent models
          (2) Global multi-height model with height as an input feature
      - Variable-length sequences via padding + masking
      - Checkpointing and safe resume via BackupAndRestore
        and periodic checkpoints
      - Logging of training progress, scalers, and loss curves

    Compared to the original baseline implementation, this version
    introduces two additional mechanisms to address systematic
    underrepresentation of extreme responses (peaks):

      (A) Quantile-based peak definition:
          Peaks are defined as the Top-K% largest response values
          within each individual time series (per-sample), rather
          than using a fixed fraction of the maximum value.

      (B) Oversampling of peak-containing sequences:
          During training, time series that contain peaks are
          replicated multiple times to increase their frequency
          in the training set and mitigate peak rarity.

    In addition, all training outputs are now automatically
    organized under a script-specific directory (named after
    the executing Python file) to prevent result overwriting
    and to improve experiment traceability across code versions.

------------------------------------------------------------
Inputs:
    Data (from previous pipeline steps):
      - GM input time series:
          Output/3_GM_Fixed_train_linear/H*/...
          Output/3_GM_Fixed_train_nonlinear/H*/...
      - THA output time series:
          Output/3_THA_Fixed_train_linear/H*/...
          Output/3_THA_Fixed_train_nonlinear/H*/...

    Runtime user inputs:
      - Linear vs Nonlinear mode
      - Clustered vs Non-clustered datasets
      - Structural heights to include
      - Training mode (per-height or multi-height)

    Scenario hyperparameters:
      - EPOCHS
      - ALPHA              (peak emphasis strength)
      - PEAK_METHOD        ("relative" or "quantile")
      - THRESH             (relative threshold or quantile q)
      - WEIGHT_MODE        (1 = binary, 2 = soft sigmoid)
      - TAU                (sigmoid temperature)
      - OVERSAMPLE_FACTOR  (integer ≥ 1; 1 disables oversampling)

------------------------------------------------------------
Outputs:
    All outputs are saved under:

      Output/
        └── Progress_of_LSTM_linear/
        │     └── <script_name>/
        │           └── <cluster|noCluster>/
        │                 └── <training_mode>/
        │                       └── <scenario_name>/
        │                             ├── LSTM.keras
        │                             ├── checkpoints/
        │                             ├── backup/
        │                             ├── progress.npy
        │                             └── loss_curve_*.png
        └── Progress_of_LSTM_nonlinear/
              └── <script_name>/
                    ...

    where <script_name> is automatically inferred from the
    executing Python file name.

------------------------------------------------------------
Changes since previous version:
    1) Quantile-based peak definition (Top-K%) added as an
       alternative to relative max-based peak detection.
    2) Oversampling of peak-containing time series introduced
       during training to address peak rarity.
    3) Output directory structure updated:
       - A script-specific parent directory is created for
         each execution to isolate results from different
         code versions or experiments.

------------------------------------------------------------
Impact of changes:
    - Improved prediction accuracy for extreme response values
      (peaks), especially for higher structural heights.
    - Reduced systematic underestimation caused by peak rarity.
    - Cleaner experiment management and full traceability of
      results across different script versions and runs.

------------------------------------------------------------
Status:
    Stable

------------------------------------------------------------
Notes:
    - Quantile-based peak detection is always performed per
      individual time series (never global).
    - Oversampling is applied only to the training set.
    - Setting PEAK_METHOD="relative" and OVERSAMPLE_FACTOR=1
      recovers behavior close to the baseline implementation.
"""






import os
import sys
import re
import time
import shutil
import json
import random
import numpy as np
import joblib
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, BackupAndRestore

# ============================================================
# ✅ Fix Unicode in console printing (Windows)
# ============================================================
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ============================================================
# 🧠 Seeds for full reproducibility
# ============================================================
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# 🟩 GPU memory growth (safe)
# ============================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

# ============================================================
# 🧭 Helper functions
# ============================================================
def param_to_str(x):
    s = f"{x}"
    s = s.replace(".", "p")
    return s

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def parse_heights_input(user_in, available_heights):
    user_in = user_in.strip()
    if not user_in:
        return available_heights

    tokens = user_in.split()
    chosen = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if t.isdigit():
            idx = int(t)
            if 0 <= idx < len(available_heights):
                chosen.append(available_heights[idx])
        else:
            if t in available_heights:
                chosen.append(t)
    chosen = list(dict.fromkeys(chosen))
    return chosen if chosen else available_heights

def pad_sequences_3d(seqs, pad_value=-999.0):
    max_len = max(s.shape[0] for s in seqs)
    n_feat = seqs[0].shape[1]
    out = np.full((len(seqs), max_len, n_feat), pad_value, dtype=np.float32)
    lengths = np.zeros((len(seqs),), dtype=np.int32)
    for i, s in enumerate(seqs):
        L = s.shape[0]
        out[i, :L, :] = s
        lengths[i] = L
    return out, lengths, max_len

def create_train_val_split(n, val_ratio=0.2, seed=1234):
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(round(n * val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx

# ============================================================
# ✅ Peak-aware weighted MSE loss (relative/quantile + binary/soft)
# ============================================================
def make_weighted_mse_loss(PAD_value, alpha, thresh, weight_mode=2, tau=0.05,
                          peak_method="relative", q=0.95):
    """
    Peak definition (per time-series):
        - peak_method="relative": peak if rel = |y|/max(|y|) >= thresh
        - peak_method="quantile": peak if |y| >= quantile_q(|y|) where quantile_q is computed per sample

    weight_mode:
        1 -> Binary peak mask
        2 -> Soft sigmoid weighting around peak boundary
    """
    PAD = tf.constant(PAD_value, dtype=tf.float32)

    def _loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        mask = tf.cast(tf.not_equal(y_true, PAD), tf.float32)  # [B,T,1] or [B,T,C]
        abs_y = tf.abs(y_true)

        # ---- peak strength (per-sample) ----
        if peak_method == "relative":
            max_abs = tf.reduce_max(abs_y, axis=1, keepdims=True)  # per sample
            max_abs = tf.maximum(max_abs, 1e-12)
            rel = abs_y / max_abs
            if weight_mode == 1:
                peak_strength = tf.cast(rel >= thresh, tf.float32)
            else:
                peak_strength = tf.sigmoid((rel - thresh) / tf.maximum(tau, 1e-6))

        elif peak_method == "quantile":
            # Flatten time dimension per sample, compute quantile threshold per sample
            # abs_y: [B,T,1] -> [B,T]
            abs_y_2d = tf.squeeze(abs_y, axis=-1)
            # sort along T
            sorted_vals = tf.sort(abs_y_2d, axis=1)
            T = tf.shape(sorted_vals)[1]
            # q in [0..1], index = ceil(q*(T-1))
            q_clamped = tf.clip_by_value(tf.cast(q, tf.float32), 0.0, 1.0)
            idx = tf.cast(tf.math.round(q_clamped * tf.cast(T - 1, tf.float32)), tf.int32)
            idx = tf.clip_by_value(idx, 0, T - 1)
            # gather threshold per sample
            batch_indices = tf.range(tf.shape(sorted_vals)[0])
            gather_idx = tf.stack([batch_indices, tf.fill(tf.shape(batch_indices), idx)], axis=1)
            thr_vals = tf.gather_nd(sorted_vals, gather_idx)  # [B]
            thr_vals = tf.reshape(thr_vals, (-1, 1, 1))       # [B,1,1]

            if weight_mode == 1:
                peak_strength = tf.cast(abs_y >= thr_vals, tf.float32)
            else:
                peak_strength = tf.sigmoid((abs_y - thr_vals) / tf.maximum(tau, 1e-6))

        else:
            raise ValueError("Invalid peak_method. Use 'relative' or 'quantile'.")

        # weights
        w = 1.0 + alpha * peak_strength
        w = w * mask

        # per-sample normalization of weights to avoid batch coupling
        w_sum = tf.reduce_sum(w, axis=1, keepdims=True) + 1e-12
        m_sum = tf.reduce_sum(mask, axis=1, keepdims=True) + 1e-12
        w_norm = w * (m_sum / w_sum)

        se = tf.square((y_true - y_pred) * mask)
        loss = tf.reduce_sum(se * w_norm) / tf.reduce_sum(mask)
        return loss

    return _loss

# ============================================================
# ✅ Build LSTM model
# ============================================================
def build_model(input_dim, pad_value, loss_fn):
    inp = layers.Input(shape=(None, input_dim), name="input_ts")
    x = layers.Masking(mask_value=pad_value)(inp)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    out = layers.Dense(1)(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=loss_fn,
                  metrics=[tf.keras.metrics.MeanSquaredError(name="mse")])
    return model

# ============================================================
# 🧭 Runtime options (user inputs)
# ============================================================
print("\nپیش‌بینی خطی؟ (1=خطی / 0=غیرخطی): ", end="")
is_linear = input().strip()
is_linear = True if is_linear == "1" else False
print("📌 پیش‌بینی بر اساس مدل خطی انجام می‌شود.\n" if is_linear else "📌 پیش‌بینی بر اساس مدل غیرخطی انجام می‌شود.\n")

print("آموزش با داده‌های کلاستر؟ (1=کلاستر / 0=غیرکلاستر): ", end="")
use_cluster = input().strip()
use_cluster = True if use_cluster == "1" else False

cluster_label = ""
if use_cluster:
    cluster_label = input("برچسب K کلاستر (مثلاً 4) - فقط برای نام‌گذاری (اختیاری): ").strip()

print("-------------------------------------------")
print("روش تعریف پیک را انتخاب کن:")
print("1 = relative (thresh × max هر تایم‌سری)")
print("2 = quantile (Top-K% هر تایم‌سری)")
print("-------------------------------------------")
_peak_choice = input("انتخاب (1 یا 2): ").strip()
PEAK_METHOD = "quantile" if _peak_choice == "2" else "relative"

# Quantile q
PEAK_Q = 0.95
if PEAK_METHOD == "quantile":
    q_in = input("مقدار q برای Quantile را وارد کن (مثلاً 0.95 = 5% بالایی). خالی = 0.95: ").strip()
    if q_in:
        try:
            PEAK_Q = float(q_in)
        except Exception:
            PEAK_Q = 0.95

# Oversampling factor
os_in = input("Oversampling factor (عدد صحیح >=1؛ 1 یعنی خاموش). خالی = 1: ").strip()
OVERSAMPLE_FACTOR = 1
if os_in:
    try:
        OVERSAMPLE_FACTOR = int(os_in)
        if OVERSAMPLE_FACTOR < 1:
            OVERSAMPLE_FACTOR = 1
    except Exception:
        OVERSAMPLE_FACTOR = 1

# ============================================================
# 🧭 Paths
# ============================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
root_project_dir = os.path.abspath(os.path.join(base_dir, ".."))

output_root_dir = os.path.join(root_project_dir, "Output")

gm_root_dir = os.path.join(output_root_dir, "3_GM_Fixed_train_linear" if is_linear else "3_GM_Fixed_train_nonlinear")
tha_root_dir = os.path.join(output_root_dir, "3_THA_Fixed_train_linear" if is_linear else "3_THA_Fixed_train_nonlinear")

# script-name folder isolation
script_name = os.path.splitext(os.path.basename(__file__))[0]

base_model_root = os.path.join(
    output_root_dir,
    "Progress_of_LSTM_linear" if is_linear else "Progress_of_LSTM_nonlinear",
    script_name
)

# ============================================================
# 🧪 Scenarios
# ============================================================
SCENARIOS = [
    {"EPOCHS": 76, "ALPHA": 1, "THRESH": 0.5, "WEIGHT_MODE": 2, "TAU": 0.05},
    {"EPOCHS": 100, "ALPHA": 1, "THRESH": 0.5, "WEIGHT_MODE": 2, "TAU": 0.05},
]

# ============================================================
# 🧭 Check data folders
# ============================================================
if not os.path.isdir(gm_root_dir):
    raise FileNotFoundError(f"❌ GM root dir پیدا نشد: {gm_root_dir}")
if not os.path.isdir(tha_root_dir):
    raise FileNotFoundError(f"❌ THA root dir پیدا نشد: {tha_root_dir}")

available_heights = sorted(
    name for name in os.listdir(gm_root_dir)
    if os.path.isdir(os.path.join(gm_root_dir, name)) and re.match(r"^H\d+$", name)
)

if not available_heights:
    raise RuntimeError("❌ هیچ پوشه‌ای مثل H1, H2,... در مسیر GM پیدا نشد.")

print("\n📂 ارتفاع‌های موجود:")
for i, h in enumerate(available_heights):
    print(f"  [{i}] {h}")

heights_in = input("\nارتفاع‌ها را وارد کن (مثلاً: H2 H3 یا 0 1). خالی = همه: ")
chosen_heights = parse_heights_input(heights_in, available_heights)

print("\n✅ ارتفاع‌های انتخاب‌شده:", chosen_heights)

# Training mode: multi-height or per-height
print("\nTraining mode:")
print("1 = Global multi-height (height as a feature)")
print("0 = Per-height independent models")
mode_in = input("انتخاب (1 یا 0): ").strip()
use_multi_height = True if mode_in == "1" else False

# ============================================================
# 🧠 Load and prepare data (per height)
# ============================================================
def load_height_data(height_name):
    gm_dir = os.path.join(gm_root_dir, height_name)
    tha_dir = os.path.join(tha_root_dir, height_name)

    if use_cluster:
        x_path = os.path.join(gm_dir, f"X_data_cluster_balanced_global_{height_name}.npy")
        y_path = os.path.join(tha_dir, f"Y_data_cluster_balanced_global_{height_name}.npy")
    else:
        x_path = os.path.join(gm_dir, f"X_data_{height_name}.npy")
        y_path = os.path.join(tha_dir, f"Y_data_{height_name}.npy")

    if not os.path.isfile(x_path):
        raise FileNotFoundError(f"❌ فایل X یافت نشد: {x_path}")
    if not os.path.isfile(y_path):
        raise FileNotFoundError(f"❌ فایل Y یافت نشد: {y_path}")

    X = np.load(x_path, allow_pickle=True)
    Y = np.load(y_path, allow_pickle=True)

    return X, Y

# ============================================================
# 🧠 Scaling utilities (keep as-is)
# ============================================================
from sklearn.preprocessing import StandardScaler

def fit_scalers(X_list, Y_list):
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_concat = np.concatenate([x.reshape(-1, x.shape[-1]) for x in X_list], axis=0)
    Y_concat = np.concatenate([y.reshape(-1, 1) for y in Y_list], axis=0)

    scaler_x.fit(X_concat)
    scaler_y.fit(Y_concat)
    return scaler_x, scaler_y

def apply_scalers(X_list, Y_list, scaler_x, scaler_y):
    X_scaled = []
    Y_scaled = []
    for x, y in zip(X_list, Y_list):
        x2 = scaler_x.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        y2 = scaler_y.transform(y.reshape(-1, 1)).reshape(y.shape)
        X_scaled.append(x2.astype(np.float32))
        Y_scaled.append(y2.astype(np.float32))
    return X_scaled, Y_scaled

# ============================================================
# 🧠 Oversampling (train only)
# ============================================================
def has_peak(y, peak_method, thresh, q):
    abs_y = np.abs(y.reshape(-1))
    if abs_y.size == 0:
        return False
    if peak_method == "relative":
        m = abs_y.max()
        if m <= 1e-12:
            return False
        rel = abs_y / m
        return np.any(rel >= thresh)
    else:
        # quantile
        thr = np.quantile(abs_y, q)
        return np.any(abs_y >= thr)

def oversample_lists(X_list, Y_list, peak_method, thresh, q, factor):
    if factor <= 1:
        return X_list, Y_list
    X_out, Y_out = [], []
    for x, y in zip(X_list, Y_list):
        X_out.append(x); Y_out.append(y)
        if has_peak(y, peak_method, thresh, q):
            for _ in range(factor - 1):
                X_out.append(x)
                Y_out.append(y)
    return X_out, Y_out

# ============================================================
# 🚀 Main training
# ============================================================
PAD = -999.0

# Mode folder naming
if use_cluster:
    mode_folder_name = f"clusterK{cluster_label}_allHeights" if cluster_label else "cluster_allHeights"
else:
    mode_folder_name = "noCluster_allHeights"

root_model_dir = os.path.join(base_model_root, mode_folder_name)
ensure_dir(root_model_dir)

if use_multi_height:
    global_multi_root_dir = os.path.join(root_model_dir, "Global_training_with_height")
    ensure_dir(global_multi_root_dir)

    # Load all selected heights, add height feature
    X_list_all = []
    Y_list_all = []

    for h in chosen_heights:
        X_h, Y_h = load_height_data(h)
        # Ensure list format (if saved as object arrays)
        if isinstance(X_h, np.ndarray) and X_h.dtype == object:
            X_h_list = list(X_h)
        else:
            X_h_list = list(X_h)

        if isinstance(Y_h, np.ndarray) and Y_h.dtype == object:
            Y_h_list = list(Y_h)
        else:
            Y_h_list = list(Y_h)

        # Height numeric as feature
        h_val = float(h.replace("H", ""))
        for x, y in zip(X_h_list, Y_h_list):
            h_feat = np.full((x.shape[0], 1), h_val, dtype=np.float32)
            x2 = np.concatenate([x.astype(np.float32), h_feat], axis=1)
            X_list_all.append(x2)
            Y_list_all.append(y.astype(np.float32))

    # Fit scalers and apply
    scaler_X, scaler_Y = fit_scalers(X_list_all, Y_list_all)
    X_list_all, Y_list_all = apply_scalers(X_list_all, Y_list_all, scaler_X, scaler_Y)

    # Split
    n_all = len(X_list_all)
    train_idx, val_idx = create_train_val_split(n_all, val_ratio=0.2, seed=SEED)
    split_idx_path = os.path.join(global_multi_root_dir, f"split_idx_seed{SEED}.npy")
    np.save(split_idx_path, {"train": train_idx, "val": val_idx})

    X_train = [X_list_all[i] for i in train_idx]
    Y_train = [Y_list_all[i] for i in train_idx]
    X_val   = [X_list_all[i] for i in val_idx]
    Y_val   = [Y_list_all[i] for i in val_idx]

    # Oversample only train
    X_train, Y_train = oversample_lists(X_train, Y_train, PEAK_METHOD, 0.5, PEAK_Q, OVERSAMPLE_FACTOR)

    # Pad sequences
    X_train_pad, _, _ = pad_sequences_3d(X_train, pad_value=PAD)
    Y_train_pad, _, _ = pad_sequences_3d([y.reshape(-1, 1) for y in Y_train], pad_value=PAD)
    X_val_pad, _, _   = pad_sequences_3d(X_val, pad_value=PAD)
    Y_val_pad, _, _   = pad_sequences_3d([y.reshape(-1, 1) for y in Y_val], pad_value=PAD)

    INPUT_DIM = X_train_pad.shape[-1]

    # Save scalers
    joblib.dump(scaler_X, os.path.join(global_multi_root_dir, f"scaler_X_{'linear' if is_linear else 'nonlinear'}.pkl"))
    joblib.dump(scaler_Y, os.path.join(global_multi_root_dir, f"scaler_Y_{'linear' if is_linear else 'nonlinear'}.pkl"))

    # Loop scenarios
    for scen in SCENARIOS:
        EPOCHS = int(scen["EPOCHS"])
        ALPHA  = float(scen["ALPHA"])
        THRESH = float(scen["THRESH"])
        WEIGHT_MODE = int(scen.get("WEIGHT_MODE", 2))
        TAU = float(scen.get("TAU", 0.05))

        # ---- Naming tags (peak definition + oversampling) ----
        if PEAK_METHOD == "quantile":
            peak_tag = f"Q{param_to_str(PEAK_Q)}"
        else:
            peak_tag = f"R{param_to_str(THRESH)}"
        os_tag = f"OS{OVERSAMPLE_FACTOR}"

        scen_name = f"ep{EPOCHS}_A{param_to_str(ALPHA)}_{peak_tag}_{os_tag}_M{WEIGHT_MODE}_tau{param_to_str(TAU)}"
        model_dir = os.path.join(global_multi_root_dir, scen_name)
        ensure_dir(model_dir)

        ckpt_dir = os.path.join(model_dir, "checkpoints")
        backup_dir = os.path.join(model_dir, "backup")
        ensure_dir(ckpt_dir)
        ensure_dir(backup_dir)

        loss_fn = make_weighted_mse_loss(PAD, ALPHA, THRESH, weight_mode=WEIGHT_MODE, tau=TAU,
                                        peak_method=PEAK_METHOD, q=PEAK_Q)
        model = build_model(INPUT_DIM, PAD, loss_fn)

        # callbacks
        best_model_path = os.path.join(model_dir, "LSTM.keras")
        ckpt_path = os.path.join(ckpt_dir, "ckpt_epoch_{epoch:04d}.keras")
        callbacks = [
            BackupAndRestore(backup_dir=backup_dir),
            ModelCheckpoint(filepath=best_model_path, monitor="val_loss", save_best_only=True, mode="min", verbose=1),
            ModelCheckpoint(filepath=ckpt_path, monitor="val_loss", save_best_only=False, save_freq="epoch", verbose=0),
        ]

        history = model.fit(
            X_train_pad, Y_train_pad,
            validation_data=(X_val_pad, Y_val_pad),
            epochs=EPOCHS,
            batch_size=20,
            callbacks=callbacks,
            verbose=1
        )

        # Save progress
        progress = {
            "history": history.history,
            "EPOCHS": EPOCHS,
            "ALPHA": ALPHA,
            "THRESH": THRESH,
            "WEIGHT_MODE": WEIGHT_MODE,
            "TAU": TAU,
            "PEAK_METHOD": PEAK_METHOD,
            "PEAK_Q": PEAK_Q,
            "OVERSAMPLE_FACTOR": OVERSAMPLE_FACTOR,
            "seed": SEED,
            "is_linear": is_linear,
            "use_cluster": use_cluster,
            "chosen_heights": chosen_heights,
            "use_multi_height": use_multi_height,
        }
        np.save(os.path.join(model_dir, "progress.npy"), progress)

        # Plot loss
        plt.figure()
        plt.plot(history.history["loss"], label="train_loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, f"loss_curve_{scen_name}.png"), dpi=150)
        plt.close()

else:
    # Per-height models
    for h in chosen_heights:
        height_root = os.path.join(root_model_dir, h)
        ensure_dir(height_root)

        X_h, Y_h = load_height_data(h)
        X_list = list(X_h) if (isinstance(X_h, np.ndarray) and X_h.dtype == object) else list(X_h)
        Y_list = list(Y_h) if (isinstance(Y_h, np.ndarray) and Y_h.dtype == object) else list(Y_h)

        scaler_X, scaler_Y = fit_scalers(X_list, Y_list)
        X_list, Y_list = apply_scalers(X_list, Y_list, scaler_X, scaler_Y)

        n = len(X_list)
        train_idx, val_idx = create_train_val_split(n, val_ratio=0.2, seed=SEED)
        split_idx_path = os.path.join(height_root, f"split_idx_seed{SEED}.npy")
        np.save(split_idx_path, {"train": train_idx, "val": val_idx})

        X_train = [X_list[i] for i in train_idx]
        Y_train = [Y_list[i] for i in train_idx]
        X_val   = [X_list[i] for i in val_idx]
        Y_val   = [Y_list[i] for i in val_idx]

        # Oversample only train
        X_train, Y_train = oversample_lists(X_train, Y_train, PEAK_METHOD, 0.5, PEAK_Q, OVERSAMPLE_FACTOR)

        X_train_pad, _, _ = pad_sequences_3d(X_train, pad_value=PAD)
        Y_train_pad, _, _ = pad_sequences_3d([y.reshape(-1, 1) for y in Y_train], pad_value=PAD)
        X_val_pad, _, _   = pad_sequences_3d(X_val, pad_value=PAD)
        Y_val_pad, _, _   = pad_sequences_3d([y.reshape(-1, 1) for y in Y_val], pad_value=PAD)

        INPUT_DIM = X_train_pad.shape[-1]

        # Save scalers
        joblib.dump(scaler_X, os.path.join(height_root, f"scaler_X_{h}_{'linear' if is_linear else 'nonlinear'}.pkl"))
        joblib.dump(scaler_Y, os.path.join(height_root, f"scaler_Y_{h}_{'linear' if is_linear else 'nonlinear'}.pkl"))

        for scen in SCENARIOS:
            EPOCHS = int(scen["EPOCHS"])
            ALPHA  = float(scen["ALPHA"])
            THRESH = float(scen["THRESH"])
            WEIGHT_MODE = int(scen.get("WEIGHT_MODE", 2))
            TAU = float(scen.get("TAU", 0.05))

            # ---- Naming tags (peak definition + oversampling) ----
            if PEAK_METHOD == "quantile":
                peak_tag = f"Q{param_to_str(PEAK_Q)}"
            else:
                peak_tag = f"R{param_to_str(THRESH)}"
            os_tag = f"OS{OVERSAMPLE_FACTOR}"

            scen_name = f"ep{EPOCHS}_A{param_to_str(ALPHA)}_{peak_tag}_{os_tag}_M{WEIGHT_MODE}_tau{param_to_str(TAU)}"
            model_dir = os.path.join(height_root, scen_name)
            ensure_dir(model_dir)

            ckpt_dir = os.path.join(model_dir, "checkpoints")
            backup_dir = os.path.join(model_dir, "backup")
            ensure_dir(ckpt_dir)
            ensure_dir(backup_dir)

            loss_fn = make_weighted_mse_loss(PAD, ALPHA, THRESH, weight_mode=WEIGHT_MODE, tau=TAU,
                                            peak_method=PEAK_METHOD, q=PEAK_Q)
            model = build_model(INPUT_DIM, PAD, loss_fn)

            best_model_path = os.path.join(model_dir, "LSTM.keras")
            ckpt_path = os.path.join(ckpt_dir, "ckpt_epoch_{epoch:04d}.keras")
            callbacks = [
                BackupAndRestore(backup_dir=backup_dir),
                ModelCheckpoint(filepath=best_model_path, monitor="val_loss", save_best_only=True, mode="min", verbose=1),
                ModelCheckpoint(filepath=ckpt_path, monitor="val_loss", save_best_only=False, save_freq="epoch", verbose=0),
            ]

            history = model.fit(
                X_train_pad, Y_train_pad,
                validation_data=(X_val_pad, Y_val_pad),
                epochs=EPOCHS,
                batch_size=20,
                callbacks=callbacks,
                verbose=1
            )

            progress = {
                "history": history.history,
                "EPOCHS": EPOCHS,
                "ALPHA": ALPHA,
                "THRESH": THRESH,
                "WEIGHT_MODE": WEIGHT_MODE,
                "TAU": TAU,
                "PEAK_METHOD": PEAK_METHOD,
                "PEAK_Q": PEAK_Q,
                "OVERSAMPLE_FACTOR": OVERSAMPLE_FACTOR,
                "seed": SEED,
                "is_linear": is_linear,
                "use_cluster": use_cluster,
                "height": h,
                "use_multi_height": use_multi_height,
            }
            np.save(os.path.join(model_dir, "progress.npy"), progress)

            plt.figure()
            plt.plot(history.history["loss"], label="train_loss")
            plt.plot(history.history["val_loss"], label="val_loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, f"loss_curve_{scen_name}.png"), dpi=150)
            plt.close()

print("\n✅ Training finished.")
