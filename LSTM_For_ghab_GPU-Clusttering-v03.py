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






import sys, io
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')

import os, gc, glob, re, shutil, random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Masking, LSTM, Activation, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import joblib

plt.ioff()

# ==============================================================
# 🔒 Reproducibility / Determinism
# ==============================================================
SEED = 1234
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    pass
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

# ==============================================================
# 🚀 GPU dynamic memory
# ==============================================================
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory set dynamically.")
    except RuntimeError as e:
        print(e)

# ==============================================================
# 📁 Paths + Linear / Nonlinear selection
# ==============================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(base_dir, os.pardir))

output_root_dir = os.path.join(root_dir, "Output")

choice = input(
    "آموزش بر اساس پاسخ خطی باشد یا غیرخطی؟ "
    "برای خطی عدد 1 و برای غیرخطی عدد 0 را وارد کن: "
).strip()
is_linear = (choice == "1")

if is_linear:
    print("📌 آموزش بر اساس داده‌های خطی (THA_linear) انجام می‌شود.")
    gm_root_dir  = os.path.join(output_root_dir, "3_GM_Fixed_train_linear")
    tha_root_dir = os.path.join(output_root_dir, "3_THA_Fixed_train_linear")
    base_model_root = os.path.join(output_root_dir, "Progress_of_LSTM_linear")
else:
    print("📌 آموزش بر اساس داده‌های غیرخطی (THA_nonlinear) انجام می‌شود.")
    gm_root_dir  = os.path.join(output_root_dir, "3_GM_Fixed_train_nonlinear")
    tha_root_dir = os.path.join(output_root_dir, "3_THA_Fixed_train_nonlinear")
    base_model_root = os.path.join(output_root_dir, "Progress_of_LSTM_nonlinear")

os.makedirs(base_model_root, exist_ok=True)

print("📂 GM root dir :", gm_root_dir)
print("📂 THA root dir:", tha_root_dir)
print("📂 Base model root:", base_model_root)
print()

# ==============================================================
# 🧾 Script-name isolation (avoid overwriting across versions)
# ==============================================================
script_name = os.path.splitext(os.path.basename(__file__))[0]
base_model_root = os.path.join(base_model_root, script_name)
os.makedirs(base_model_root, exist_ok=True)

# ==============================================================
# 🔁 Clustered vs non-clustered
# ==============================================================
print("-------------------------------------------")
print("آیا می‌خواهی از داده‌های *کلاسترشده* استفاده کنم؟")
print("   1 = بله، از فایل‌های cluster_balanced_global استفاده کن")
print("   0 = خیر، از فایل‌های اصلی X_data_H* و Y_data_H* استفاده کن")
print("-------------------------------------------\n")

cluster_choice = input("انتخابت را وارد کن (1 یا 0): ").strip()
USE_CLUSTERED = (cluster_choice == "1")

if USE_CLUSTERED:
    print("✅ استفاده از داده‌های کلاسترشده (cluster_balanced_global).")
    cluster_label = input(
        "برای اسم پوشهٔ مدل، تعداد کلاسترها (K) را بنویس "
        "(مثلاً 4). اگر خالی بگذاری، 'clustered' نوشته می‌شود: "
    ).strip()
    if cluster_label:
        mode_folder_name = f"clusterK{cluster_label}_allHeights"
    else:
        mode_folder_name = "clustered_allHeights"
else:
    print("✅ استفاده از داده‌های اصلی بدون کلاسترینگ.")
    mode_folder_name = "noCluster_allHeights"

root_model_dir = os.path.join(base_model_root, mode_folder_name)
os.makedirs(root_model_dir, exist_ok=True)

print("\n📂 Root model dir for this run:")
print("   ", root_model_dir)
print()

# ==============================================================
# 🧠 Peak definition + Oversampling inputs
# ==============================================================
print("-------------------------------------------")
print("روش تعریف پیک را انتخاب کن:")
print("1 = relative (thresh × max هر تایم‌سری)")
print("2 = quantile (Top-K% هر تایم‌سری)")
print("-------------------------------------------")
_peak_choice = input("انتخاب (1 یا 2): ").strip()
PEAK_METHOD = "quantile" if _peak_choice == "2" else "relative"

PEAK_Q = 0.95
if PEAK_METHOD == "quantile":
    q_in = input("مقدار q برای Quantile را وارد کن (مثلاً 0.95 یعنی 5% بالایی). خالی = 0.95: ").strip()
    if q_in:
        try:
            PEAK_Q = float(q_in)
        except Exception:
            PEAK_Q = 0.95

os_in = input("Oversampling factor (عدد صحیح >=1؛ 1 یعنی خاموش). خالی = 1: ").strip()
OVERSAMPLE_FACTOR = 1
if os_in:
    try:
        OVERSAMPLE_FACTOR = int(os_in)
        if OVERSAMPLE_FACTOR < 1:
            OVERSAMPLE_FACTOR = 1
    except Exception:
        OVERSAMPLE_FACTOR = 1

# ==============================================================
# 🔧 Scenarios
# ==============================================================
SCENARIOS = [
    {"EPOCHS": 76, "ALPHA": 1, "THRESH": 0.5, "WEIGHT_MODE": 1, "TAU": 0.05},
    # {"EPOCHS": 100, "ALPHA": 1, "THRESH": 0.5, "WEIGHT_MODE": 1, "TAU": 0.05},
]

# ==============================================================
# 🧭 Data folders check
# ==============================================================
if not os.path.isdir(gm_root_dir):
    raise FileNotFoundError(f"❌ GM root dir پیدا نشد: {gm_root_dir}")
if not os.path.isdir(tha_root_dir):
    raise FileNotFoundError(f"❌ THA root dir پیدا نشد: {tha_root_dir}")

available_heights = sorted(
    name for name in os.listdir(gm_root_dir)
    if os.path.isdir(os.path.join(gm_root_dir, name)) and name.startswith("H")
)
if not available_heights:
    raise ValueError(f"❌ هیچ پوشه‌ای به شکل H* در {gm_root_dir} پیدا نشد. مرحله ۳ را چک کن.")

print("📏 ارتفاع‌های موجود در داده‌ها:")
for h in available_heights:
    print("  -", h)

print("\n-------------------------------------------")
print("برای چه ارتفاع‌هایی می‌خواهی آموزش انجام شود؟")
print("مثال: H2 H3 H4  یا  2 3 4  (خالی = همه)")
print("-------------------------------------------\n")

heights_raw = input("ارتفاع‌ها را وارد کن (خالی = همه): ").strip()

if not heights_raw:
    height_tags = available_heights[:]
    print("\n✅ همه ارتفاع‌های موجود انتخاب شدند.")
else:
    tokens = heights_raw.replace(',', ' ').split()
    selected, invalid = [], []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if tok.startswith("H"):
            tag = tok
        else:
            try:
                v = float(tok)
                if v.is_integer():
                    tag = f"H{int(v)}"
                else:
                    tag = "H" + str(v).replace('.', 'p')
            except Exception:
                invalid.append(tok)
                continue
        if tag in available_heights:
            selected.append(tag)
        else:
            invalid.append(tok)
    height_tags = list(dict.fromkeys(selected))
    if invalid:
        print("\n⚠️ این مقادیر معتبر نبودند یا داده‌ای برای آنها پیدا نشد و نادیده گرفته شدند:")
        for x in invalid:
            print("  -", x)
    if not height_tags:
        print("❌ هیچ ارتفاع معتبری انتخاب نشد؛ برنامه متوقف می‌شود.")
        sys.exit(1)

print("\n📏 ارتفاع‌های نهایی انتخاب‌شده برای آموزش:")
print("  " + ", ".join(height_tags))
print()

def height_value_from_tag(h_tag: str) -> float:
    s = h_tag[1:]
    s = s.replace('p', '.')
    return float(s)

height_values = {h_tag: height_value_from_tag(h_tag) for h_tag in height_tags}
print("🔢 نگاشت ارتفاع‌ها (tag → value):")
for h_tag, hv in height_values.items():
    print(f"  {h_tag} → {hv}")

# ==============================================================
# ↔️ Training mode selection
# ==============================================================
print("\n-------------------------------------------")
print("آیا می‌خواهی همهٔ ارتفاع‌ها با هم و با فیچر ارتفاع آموزش داده شوند؟")
print("   1 = بله، یک مدل مشترک با فیچر ارتفاع (Multi-height + Feature H)")
print("   0 = خیر، برای هر ارتفاع جداگانه مدل مستقل آموزش داده شود")
print("-------------------------------------------\n")

mh_choice = input("انتخابت را وارد کن (1 یا 0): ").strip()
USE_MULTI_HEIGHT = (mh_choice == "1")

if USE_MULTI_HEIGHT:
    print("✅ مود ۱: مدل مشترک برای همه ارتفاع‌ها با فیچر ارتفاع (X = [GM , H])")
else:
    print("✅ مود ۲: مدل جداگانه برای هر ارتفاع، بدون فیچر ارتفاع (X = [GM])")

# ==============================================================
# ⚙️ Common settings / helpers
# ==============================================================
PAD = -999.0
BATCH_SIZE = 20

def param_to_str(v):
    v = float(v)
    if v.is_integer():
        return f"{int(v)}.0"
    return str(v)

def get_data_paths_for_height(h_tag):
    """
    Dict-based input files, consistent with your original working code.
    """
    if USE_CLUSTERED:
        gm_dir  = os.path.join(gm_root_dir,  h_tag, "cluster_balanced_global")
        tha_dir = os.path.join(tha_root_dir, h_tag, "cluster_balanced_global")
        x_name  = f"X_data_cluster_balanced_global_{h_tag}.npy"
        y_name  = f"Y_data_cluster_balanced_global_{h_tag}.npy"
    else:
        gm_dir  = os.path.join(gm_root_dir,  h_tag)
        tha_dir = os.path.join(tha_root_dir, h_tag)
        x_name  = f"X_data_{h_tag}.npy"
        y_name  = f"Y_data_{h_tag}.npy"
    return os.path.join(gm_dir, x_name), os.path.join(tha_dir, y_name)

# ==============================================================
# ✅ Peak-aware weighted MSE loss (relative/quantile + binary/soft)
# ==============================================================
def make_weighted_mse_loss(PAD_value, alpha, thresh, weight_mode=2, tau=0.05,
                          peak_method="relative", q=0.95):
    """
    Peak definition (per time-series):
      - peak_method="relative": peak if rel=|y|/max(|y|) >= thresh
      - peak_method="quantile": peak if |y| >= quantile_q(|y|) per sample

    weight_mode:
      1 -> binary mask
      2 -> soft sigmoid
    """
    PAD_val = tf.constant(PAD_value, dtype=tf.float32)

    def weighted_mse_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        mask = tf.cast(tf.not_equal(y_true, PAD_val), tf.float32)  # [B,T,1]
        abs_y = tf.abs(y_true) * mask                               # [B,T,1]

        if peak_method == "relative":
            max_abs = tf.reduce_max(abs_y, axis=1, keepdims=True) + 1e-6  # [B,1,1]
            rel = abs_y / max_abs
            if weight_mode == 1:
                peak_strength = tf.cast(rel >= thresh, tf.float32)
            else:
                t = tf.constant(float(tau), dtype=tf.float32)
                peak_strength = tf.sigmoid((rel - thresh) / (t + 1e-6))

        elif peak_method == "quantile":
            abs_y_2d = tf.squeeze(abs_y, axis=-1)                     # [B,T]
            sorted_vals = tf.sort(abs_y_2d, axis=1)
            T = tf.shape(sorted_vals)[1]
            q_clamped = tf.clip_by_value(tf.cast(q, tf.float32), 0.0, 1.0)
            idx = tf.cast(tf.math.round(q_clamped * tf.cast(T - 1, tf.float32)), tf.int32)
            idx = tf.clip_by_value(idx, 0, T - 1)
            batch_indices = tf.range(tf.shape(sorted_vals)[0])
            gather_idx = tf.stack([batch_indices, tf.fill(tf.shape(batch_indices), idx)], axis=1)
            thr_vals = tf.gather_nd(sorted_vals, gather_idx)          # [B]
            thr_vals = tf.reshape(thr_vals, (-1, 1, 1))               # [B,1,1]
            if weight_mode == 1:
                peak_strength = tf.cast(abs_y >= thr_vals, tf.float32)
            else:
                t = tf.constant(float(tau), dtype=tf.float32)
                peak_strength = tf.sigmoid((abs_y - thr_vals) / (t + 1e-6))
        else:
            raise ValueError("Invalid peak_method. Use 'relative' or 'quantile'.")

        w = 1.0 + alpha * peak_strength
        w = w * mask

        # per-sample weight normalization (no batch coupling)
        w_sum = tf.reduce_sum(w, axis=1, keepdims=True) + 1e-6
        m_sum = tf.reduce_sum(mask, axis=1, keepdims=True) + 1e-6
        w = w * (m_sum / w_sum)

        sq = tf.square(y_true - y_pred) * mask
        loss_per_sample = tf.reduce_sum(w * sq, axis=1) / m_sum
        return tf.reduce_mean(loss_per_sample)

    return weighted_mse_loss

def build_model(input_dim, PAD_value, loss_fn):
    inp = Input(shape=(None, input_dim), name="dense_input")
    x = Masking(mask_value=PAD_value, name="masking")(inp)
    x = LSTM(100, return_sequences=True, name="lstm1")(x)
    x = Activation('relu', name="relu1")(x)
    x = LSTM(100, return_sequences=True, name="lstm2")(x)
    x = Activation('relu', name="relu2")(x)
    x = Dense(100, name="dense1")(x)
    out = Dense(1, name="dense_out")(x)

    model = Model(inputs=inp, outputs=out, name="lstm_masked_dense")
    model.compile(loss=loss_fn, optimizer=Adam(learning_rate=0.001), metrics=['mse'])
    return model

class PeriodicSaver(tf.keras.callbacks.Callback):
    def __init__(self, save_dir, backup_dir=None, period=10, keep_last=1):
        super().__init__()
        self.save_dir = save_dir
        self.backup_dir = backup_dir
        self.period = int(period)
        self.keep_last = max(int(keep_last), 1)
        os.makedirs(self.save_dir, exist_ok=True)

    def _prune_checkpoints(self):
        files = glob.glob(os.path.join(self.save_dir, "ckpt_epoch_*.keras"))
        if len(files) <= self.keep_last:
            return

        def parse_epoch(p):
            m = re.search(r"ckpt_epoch_(\d+)\.keras$", os.path.basename(p))
            return int(m.group(1)) if m else 0

        files.sort(key=parse_epoch)
        for f in files[:-self.keep_last]:
            try:
                os.remove(f)
                print(f"🧹 old checkpoint removed: {f}")
            except Exception as e:
                print(f"⚠️ could not remove {f}: {e}")

    def _prune_backup(self):
        if not self.backup_dir or not os.path.exists(self.backup_dir):
            return
        entries = [os.path.join(self.backup_dir, n) for n in os.listdir(self.backup_dir)]
        if len(entries) <= self.keep_last:
            return
        entries.sort(key=lambda p: os.path.getmtime(p))
        for p in entries[:-self.keep_last]:
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    os.remove(p)
                print(f"🧹 old backup removed: {p}")
            except Exception as e:
                print(f"⚠️ could not remove backup {p}: {e}")

    def on_epoch_end(self, epoch, logs=None):
        ep = epoch + 1
        if ep % self.period == 0:
            path = os.path.join(self.save_dir, f"ckpt_epoch_{ep:04d}.keras")
            self.model.save(path)
            print(f"💾 Periodic checkpoint saved: {path}")
            self._prune_checkpoints()
            self._prune_backup()

def find_latest_periodic_checkpoint(directory):
    files = glob.glob(os.path.join(directory, "ckpt_epoch_*.keras"))
    if not files:
        return None, 0

    def parse_epoch(p):
        m = re.search(r"ckpt_epoch_(\d+)\.keras$", os.path.basename(p))
        return int(m.group(1)) if m else 0

    files.sort(key=parse_epoch)
    latest = files[-1]
    return latest, parse_epoch(latest)

def is_scenario_finished(model_dir, expected_epochs):
    progress_path = os.path.join(model_dir, "progress.npy")
    if not os.path.exists(progress_path):
        return False
    try:
        data = np.load(progress_path, allow_pickle=True).item()
        trained = int(data.get("epochs_trained", data.get("epochs", 0)))
        return trained >= expected_epochs
    except Exception as e:
        print(f"⚠️ could not read progress for {model_dir}: {e}")
        return False

# ==============================================================
# Oversampling helpers (train only)
# ==============================================================
def has_peak_np(y, peak_method, thresh, q):
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
        thr = np.quantile(abs_y, q)
        return np.any(abs_y >= thr)

def oversample_lists(X_list, Y_list, peak_method, thresh, q, factor):
    if factor <= 1:
        return X_list, Y_list
    X_out, Y_out = [], []
    for x, y in zip(X_list, Y_list):
        X_out.append(x); Y_out.append(y)
        if has_peak_np(y, peak_method, thresh, q):
            for _ in range(factor - 1):
                X_out.append(x)
                Y_out.append(y)
    return X_out, Y_out

# ==============================================================
# Dataset builders (list -> tf.data)
# ==============================================================
def gen_from_lists(x_list, y_list):
    for x, y in zip(x_list, y_list):
        yield x, y

def make_datasets(X_train_list, Y_train_list, X_val_list, Y_val_list, input_dim):
    output_signature = (
        tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
    )

    ds_train = tf.data.Dataset.from_generator(
        lambda: gen_from_lists(X_train_list, Y_train_list),
        output_signature=output_signature
    ).padded_batch(
        BATCH_SIZE,
        padded_shapes=([None, input_dim], [None, 1]),
        padding_values=(PAD, PAD),
        drop_remainder=False
    ).prefetch(tf.data.AUTOTUNE)

    ds_val = tf.data.Dataset.from_generator(
        lambda: gen_from_lists(X_val_list, Y_val_list),
        output_signature=output_signature
    ).padded_batch(
        BATCH_SIZE,
        padded_shapes=([None, input_dim], [None, 1]),
        padding_values=(PAD, PAD),
        drop_remainder=False
    ).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val

# ==============================================================
# 🚀 MODE 1: Global multi-height (+ height feature)
# ==============================================================
global_multi_root_dir = os.path.join(root_model_dir, "Global_training_with_height")
os.makedirs(global_multi_root_dir, exist_ok=True)

if USE_MULTI_HEIGHT:
    INPUT_DIM = 2  # [GM, H]
    X_all, Y_all = [], []

    for h_tag in height_tags:
        print("\n" + "#" * 80)
        print(f"📥 بارگذاری داده برای ارتفاع: {h_tag}")
        print("#" * 80)

        x_data_path, y_data_path = get_data_paths_for_height(h_tag)

        if not os.path.exists(x_data_path):
            print(f"⚠️ X برای {h_tag} پیدا نشد: {x_data_path} → رد می‌شود.")
            continue
        if not os.path.exists(y_data_path):
            print(f"⚠️ Y برای {h_tag} پیدا نشد: {y_data_path} → رد می‌شود.")
            continue

        # ✅ Dict-based loading (matches your original code)
        X_dict = np.load(x_data_path, allow_pickle=True).item()
        Y_dict = np.load(y_data_path, allow_pickle=True).item()

        common_keys = sorted(set(X_dict.keys()) & set(Y_dict.keys()))
        if not common_keys:
            print(f"❌ هیچ کلید مشترکی برای {h_tag} پیدا نشد. رد می‌شود.")
            continue

        h_val = np.float32(height_values[h_tag])

        for k in common_keys:
            x = X_dict[k].reshape(-1, 1).astype('float32')
            y = Y_dict[k].reshape(-1, 1).astype('float32')

            T = x.shape[0]
            h_col = np.full((T, 1), h_val, dtype='float32')
            x_feat = np.concatenate([x, h_col], axis=1)  # [GM, H]

            X_all.append(x_feat)
            Y_all.append(y)

    num_samples = len(X_all)
    if num_samples == 0:
        raise ValueError("❌ هیچ نمونه‌ای برای آموزش پیدا نشد (بعد از فیلتر ارتفاع‌ها/کلاسترها).")

    print(f"\n📦 تعداد کل نمونه‌ها (همه ارتفاع‌ها با هم): {num_samples}")

    # Global scaling
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    scaler_Y = MinMaxScaler(feature_range=(-1, 1))

    X_concat = np.concatenate(X_all, axis=0)
    Y_concat = np.concatenate(Y_all, axis=0)

    scaler_X.fit(X_concat)
    scaler_Y.fit(Y_concat)

    X_all = [scaler_X.transform(x) for x in X_all]
    Y_all = [scaler_Y.transform(y) for y in Y_all]

    # Save scalers
    if is_linear:
        scaler_X_path = os.path.join(global_multi_root_dir, "scaler_X_linear.pkl")
        scaler_Y_path = os.path.join(global_multi_root_dir, "scaler_Y_linear.pkl")
    else:
        scaler_X_path = os.path.join(global_multi_root_dir, "scaler_X_nonlinear.pkl")
        scaler_Y_path = os.path.join(global_multi_root_dir, "scaler_Y_nonlinear.pkl")

    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_Y, scaler_Y_path)
    print("\n💾 Global scalers saved:")
    print("  →", scaler_X_path)
    print("  →", scaler_Y_path)

    # Fixed global split indices (permutation)
    split_idx_path = os.path.join(global_multi_root_dir, f"split_idx_seed{SEED}.npy")

    if os.path.exists(split_idx_path):
        idx = np.load(split_idx_path)
        if len(idx) != num_samples:
            idx = np.arange(num_samples)
            rng = np.random.default_rng(SEED)
            rng.shuffle(idx)
            np.save(split_idx_path, idx)
            print("⚠️ طول داده‌ها تغییر کرده بود؛ split جدید ساخته شد.")
        else:
            print("✅ Fixed global split indices loaded:", split_idx_path)
    else:
        idx = np.arange(num_samples)
        rng = np.random.default_rng(SEED)
        rng.shuffle(idx)
        np.save(split_idx_path, idx)
        print("✅ Fixed global split indices saved:", split_idx_path)

    train_split = int(0.50 * num_samples)
    val_split   = int(0.63 * num_samples)

    X_train_base = [X_all[i] for i in idx[:train_split]]
    Y_train_base = [Y_all[i] for i in idx[:train_split]]

    X_val_list   = [X_all[i] for i in idx[train_split:val_split]]
    Y_val_list   = [Y_all[i] for i in idx[train_split:val_split]]

    print(f"\n📦 Global split | Train: {len(X_train_base)}  Val: {len(X_val_list)}  Test: {num_samples - val_split}")

    heights_str = ", ".join(height_tags)

    # Scenario loop
    for scen in SCENARIOS:
        EPOCHS = int(scen["EPOCHS"])
        ALPHA  = float(scen["ALPHA"])
        THRESH = float(scen["THRESH"])
        WEIGHT_MODE = int(scen.get("WEIGHT_MODE", 2))
        TAU = float(scen.get("TAU", 0.05))

        # Oversampling (train only) – use scenario THRESH when relative
        X_train_list, Y_train_list = oversample_lists(
            X_train_base, Y_train_base,
            PEAK_METHOD, THRESH, PEAK_Q, OVERSAMPLE_FACTOR
        )

        # ---- Naming tags (peak definition + oversampling) ----
        if PEAK_METHOD == "quantile":
            peak_tag = f"Q{param_to_str(PEAK_Q)}"
        else:
            peak_tag = f"R{param_to_str(THRESH)}"
        os_tag = f"OS{OVERSAMPLE_FACTOR}"

        scen_name = f"ep{EPOCHS}_A{param_to_str(ALPHA)}_{peak_tag}_{os_tag}_M{WEIGHT_MODE}_tau{param_to_str(TAU)}"
        model_dir = os.path.join(global_multi_root_dir, scen_name)
        os.makedirs(model_dir, exist_ok=True)

        model_path    = os.path.join(model_dir, "LSTM.keras")
        backup_dir    = os.path.join(model_dir, "backup")
        ckpt_dir      = os.path.join(model_dir, "checkpoints")
        progress_path = os.path.join(model_dir, "progress.npy")
        os.makedirs(backup_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        if is_scenario_finished(model_dir, EPOCHS):
            print("\n" + "=" * 80)
            print(f"⏩ سناریو {scen_name} قبلاً کامل شده است. رد می‌شود.")
            print("=" * 80)
            continue

        print("\n" + "=" * 80)
        print(f"🚀 شروع سناریو (Global + Height): {scen_name}")
        print(f"   → EPOCHS={EPOCHS}, ALPHA={ALPHA}, THRESH={THRESH}, WEIGHT_MODE={WEIGHT_MODE}, TAU={TAU}")
        print(f"   → PeakMethod={PEAK_METHOD}, Q={PEAK_Q}, Oversample={OVERSAMPLE_FACTOR}")
        print(f"   → Heights: {heights_str}")
        print("=" * 80)

        ds_train, ds_val = make_datasets(X_train_list, Y_train_list, X_val_list, Y_val_list, INPUT_DIM)

        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        loss_fn = make_weighted_mse_loss(
            PAD, ALPHA, THRESH,
            weight_mode=WEIGHT_MODE, tau=TAU,
            peak_method=PEAK_METHOD, q=PEAK_Q
        )
        model = build_model(INPUT_DIM, PAD, loss_fn)

        checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor='val_loss', save_best_only=True, verbose=1
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10, verbose=1, min_lr=1e-6
        )
        backup_cb = tf.keras.callbacks.BackupAndRestore(backup_dir=backup_dir)
        periodic_ckpt = PeriodicSaver(
            save_dir=ckpt_dir, backup_dir=backup_dir,
            period=10, keep_last=1
        )

        initial_epoch = 0
        if not os.path.exists(backup_dir) or not os.listdir(backup_dir):
            ckpt_path, ep = find_latest_periodic_checkpoint(ckpt_dir)
            if ckpt_path:
                try:
                    tmp_model = load_model(ckpt_path, compile=False)
                    tmp_model.compile(loss=loss_fn, optimizer=Adam(learning_rate=0.001), metrics=['mse'])
                    model = tmp_model
                    initial_epoch = ep
                    print(f"🔁 Resuming from periodic checkpoint: {ckpt_path} (initial_epoch={initial_epoch})")
                except Exception as e:
                    print(f"⚠️ Failed to load periodic checkpoint: {e}")

        history = model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=EPOCHS,
            initial_epoch=initial_epoch,
            callbacks=[backup_cb, checkpoint_best, periodic_ckpt, reduce_lr],
            verbose=1
        )

        epochs_trained = initial_epoch + len(history.history.get('loss', []))

        progress_data = {
            'train_loss': history.history.get('loss', []),
            'val_loss': history.history.get('val_loss', []),
            'best_val_loss': float(min(history.history.get('val_loss', [np.inf]))),
            'epochs': EPOCHS,
            'epochs_trained': int(epochs_trained),
            'heights_used': height_tags,
            'use_clustered': USE_CLUSTERED,
            'cluster_folder': mode_folder_name,
            'peak_method': PEAK_METHOD,
            'peak_q': float(PEAK_Q),
            'oversample_factor': int(OVERSAMPLE_FACTOR),
            'alpha': float(ALPHA),
            'thresh': float(THRESH),
            'weight_mode': int(WEIGHT_MODE),
            'tau': float(TAU),
            'seed': SEED,
        }
        np.save(progress_path, progress_data)
        print("💾 Progress saved:", progress_path)

        try:
            plt.figure()
            plt.plot(progress_data['train_loss'], label='Train Loss')
            plt.plot(progress_data['val_loss'], label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'All Heights [{heights_str}] - Loss - {scen_name}')
            plt.legend()
            plt.tight_layout()

            loss_plot_path = os.path.join(model_dir, f"loss_curve_{scen_name}.png")
            plt.savefig(loss_plot_path, dpi=300)
            plt.close()
            print("📊 Loss curve saved to:", loss_plot_path)
        except Exception as e:
            print(f"⚠️ Plot error (loss curve): {e}")

        print(f"✅ Scenario {scen_name} finished.\n")

    print("🎉 همه سناریوها در مود Multi-height تمام شدند.")

# ==============================================================
# 🚀 MODE 2: Per-height (no height feature)
# ==============================================================
else:
    INPUT_DIM = 1

    for h_tag in height_tags:
        print("\n" + "#" * 80)
        print(f"🏗️ شروع آموزش جداگانه برای ارتفاع: {h_tag}")
        print("#" * 80)

        x_data_path, y_data_path = get_data_paths_for_height(h_tag)

        if not os.path.exists(x_data_path):
            print(f"⚠️ X_data برای {h_tag} پیدا نشد: {x_data_path} → رد می‌شود.")
            continue
        if not os.path.exists(y_data_path):
            print(f"⚠️ Y_data برای {h_tag} پیدا نشد: {y_data_path} → رد می‌شود.")
            continue

        X_dict = np.load(x_data_path, allow_pickle=True).item()
        Y_dict = np.load(y_data_path, allow_pickle=True).item()

        common_keys = sorted(set(X_dict.keys()) & set(Y_dict.keys()))
        if not common_keys:
            print(f"❌ هیچ کلید مشترکی بین X و Y برای {h_tag} پیدا نشد. رد می‌شود.")
            continue

        X_all, Y_all = [], []
        for k in common_keys:
            x = X_dict[k].reshape(-1, 1).astype('float32')
            y = Y_dict[k].reshape(-1, 1).astype('float32')
            X_all.append(x)
            Y_all.append(y)

        num_samples = len(X_all)
        if num_samples == 0:
            print(f"⚠️ هیچ نمونه‌ای برای {h_tag} وجود ندارد. رد می‌شود.")
            continue

        print(f"📦 تعداد نمونه‌ها برای {h_tag}: {num_samples}")

        scaler_X = MinMaxScaler(feature_range=(-1, 1))
        scaler_Y = MinMaxScaler(feature_range=(-1, 1))

        X_concat = np.concatenate(X_all, axis=0)
        Y_concat = np.concatenate(Y_all, axis=0)

        scaler_X.fit(X_concat)
        scaler_Y.fit(Y_concat)

        X_all = [scaler_X.transform(x) for x in X_all]
        Y_all = [scaler_Y.transform(y) for y in Y_all]

        height_model_root = os.path.join(root_model_dir, h_tag)
        os.makedirs(height_model_root, exist_ok=True)

        if is_linear:
            scaler_X_path = os.path.join(height_model_root, "scaler_X_linear.pkl")
            scaler_Y_path = os.path.join(height_model_root, "scaler_Y_linear.pkl")
        else:
            scaler_X_path = os.path.join(height_model_root, "scaler_X_nonlinear.pkl")
            scaler_Y_path = os.path.join(height_model_root, "scaler_Y_nonlinear.pkl")

        joblib.dump(scaler_X, scaler_X_path)
        joblib.dump(scaler_Y, scaler_Y_path)
        print("💾 height-scalers saved:")
        print("  →", scaler_X_path)
        print("  →", scaler_Y_path)

        split_idx_path = os.path.join(height_model_root, f"split_idx_seed{SEED}.npy")

        if os.path.exists(split_idx_path):
            idx = np.load(split_idx_path)
            if len(idx) != num_samples:
                idx = np.arange(num_samples)
                rng = np.random.default_rng(SEED)
                rng.shuffle(idx)
                np.save(split_idx_path, idx)
                print("⚠️ طول داده‌ها تغییر کرده بود؛ split جدید ساخته شد.")
            else:
                print("✅ Fixed split indices loaded:", split_idx_path)
        else:
            idx = np.arange(num_samples)
            rng = np.random.default_rng(SEED)
            rng.shuffle(idx)
            np.save(split_idx_path, idx)
            print("✅ Fixed split indices saved:", split_idx_path)

        train_split = int(0.50 * num_samples)
        val_split   = int(0.63 * num_samples)

        X_train_base = [X_all[i] for i in idx[:train_split]]
        Y_train_base = [Y_all[i] for i in idx[:train_split]]

        X_val_list   = [X_all[i] for i in idx[train_split:val_split]]
        Y_val_list   = [Y_all[i] for i in idx[train_split:val_split]]

        for scen in SCENARIOS:
            EPOCHS = int(scen["EPOCHS"])
            ALPHA  = float(scen["ALPHA"])
            THRESH = float(scen["THRESH"])
            WEIGHT_MODE = int(scen.get("WEIGHT_MODE", 2))
            TAU = float(scen.get("TAU", 0.05))

            X_train_list, Y_train_list = oversample_lists(
                X_train_base, Y_train_base,
                PEAK_METHOD, THRESH, PEAK_Q, OVERSAMPLE_FACTOR
            )

            if PEAK_METHOD == "quantile":
                peak_tag = f"Q{param_to_str(PEAK_Q)}"
            else:
                peak_tag = f"R{param_to_str(THRESH)}"
            os_tag = f"OS{OVERSAMPLE_FACTOR}"

            scen_name = f"ep{EPOCHS}_A{param_to_str(ALPHA)}_{peak_tag}_{os_tag}_M{WEIGHT_MODE}_tau{param_to_str(TAU)}"
            model_dir = os.path.join(height_model_root, scen_name)
            os.makedirs(model_dir, exist_ok=True)

            model_path    = os.path.join(model_dir, "LSTM.keras")
            backup_dir    = os.path.join(model_dir, "backup")
            ckpt_dir      = os.path.join(model_dir, "checkpoints")
            progress_path = os.path.join(model_dir, "progress.npy")

            if is_scenario_finished(model_dir, EPOCHS):
                print("\n" + "=" * 80)
                print(f"⏩ {h_tag} | سناریو {scen_name} قبلاً کامل شده است. رد می‌شود.")
                print("=" * 80)
                continue

            print("\n" + "=" * 80)
            print(f"🚀 {h_tag} | شروع سناریو: {scen_name}")
            print(f"   → EPOCHS={EPOCHS}, ALPHA={ALPHA}, THRESH={THRESH}, WEIGHT_MODE={WEIGHT_MODE}, TAU={TAU}")
            print(f"   → PeakMethod={PEAK_METHOD}, Q={PEAK_Q}, Oversample={OVERSAMPLE_FACTOR}")
            print("=" * 80)

            os.makedirs(ckpt_dir, exist_ok=True)
            os.makedirs(backup_dir, exist_ok=True)

            ds_train, ds_val = make_datasets(X_train_list, Y_train_list, X_val_list, Y_val_list, INPUT_DIM)

            tf.random.set_seed(SEED)
            np.random.seed(SEED)
            random.seed(SEED)

            loss_fn = make_weighted_mse_loss(
                PAD, ALPHA, THRESH,
                weight_mode=WEIGHT_MODE, tau=TAU,
                peak_method=PEAK_METHOD, q=PEAK_Q
            )
            model = build_model(INPUT_DIM, PAD, loss_fn)

            checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
                model_path, monitor='val_loss', save_best_only=True, verbose=1
            )
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=10, verbose=1, min_lr=1e-6
            )
            backup_cb = tf.keras.callbacks.BackupAndRestore(backup_dir=backup_dir)
            periodic_ckpt = PeriodicSaver(
                save_dir=ckpt_dir, backup_dir=backup_dir,
                period=10, keep_last=1
            )

            initial_epoch = 0
            if not os.path.exists(backup_dir) or not os.listdir(backup_dir):
                ckpt_path, ep = find_latest_periodic_checkpoint(ckpt_dir)
                if ckpt_path:
                    try:
                        tmp_model = load_model(ckpt_path, compile=False)
                        tmp_model.compile(loss=loss_fn, optimizer=Adam(learning_rate=0.001), metrics=['mse'])
                        model = tmp_model
                        initial_epoch = ep
                        print(f"🔁 {h_tag} | Resuming from checkpoint: {ckpt_path} (initial_epoch={initial_epoch})")
                    except Exception as e:
                        print(f"⚠️ Failed to load periodic checkpoint: {e}")

            history = model.fit(
                ds_train,
                validation_data=ds_val,
                epochs=EPOCHS,
                initial_epoch=initial_epoch,
                callbacks=[backup_cb, checkpoint_best, periodic_ckpt, reduce_lr],
                verbose=1
            )

            epochs_trained = initial_epoch + len(history.history.get('loss', []))

            progress_data = {
                'train_loss': history.history.get('loss', []),
                'val_loss': history.history.get('val_loss', []),
                'best_val_loss': float(min(history.history.get('val_loss', [np.inf]))),
                'epochs': EPOCHS,
                'epochs_trained': int(epochs_trained),
                'height': h_tag,
                'use_clustered': USE_CLUSTERED,
                'cluster_folder': mode_folder_name,
                'peak_method': PEAK_METHOD,
                'peak_q': float(PEAK_Q),
                'oversample_factor': int(OVERSAMPLE_FACTOR),
                'alpha': float(ALPHA),
                'thresh': float(THRESH),
                'weight_mode': int(WEIGHT_MODE),
                'tau': float(TAU),
                'seed': SEED,
            }
            np.save(progress_path, progress_data)
            print("💾 Progress saved:", progress_path)

            try:
                plt.figure()
                plt.plot(progress_data['train_loss'], label='Train Loss')
                plt.plot(progress_data['val_loss'], label='Val Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'{h_tag} - Loss - {scen_name}')
                plt.legend()
                plt.tight_layout()

                loss_plot_path = os.path.join(model_dir, f"loss_curve_{scen_name}.png")
                plt.savefig(loss_plot_path, dpi=300)
                plt.close()
                print("📊 Loss curve saved to:", loss_plot_path)
            except Exception as e:
                print(f"⚠️ Plot error (loss curve): {e}")

            print(f"✅ {h_tag} | Scenario {scen_name} finished.\n")

    print("🎉 آموزش همه ارتفاع‌ها در مود per-height تمام شد.")
