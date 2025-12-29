# -*- coding: utf-8 -*-
"""
File name      : LSTM_For_ghab_GPU-Clusttering.py
@Author: pc22
Created on     : Sun Dec 28 08:47:13 2025
Last modified  : Sun Dec 28 08:47:13 2025
------------------------------------------------------------
Purpose:
    Improve peak (extreme) prediction quality in LSTM-based
    time-history response forecasting by fixing peak-weighted
    loss behavior (batch-dependency) and introducing a smoother
    peak emphasis mechanism.

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
      - Checkpointing and safe resume via BackupAndRestore + periodic checkpoints
      - Logging of training progress, scalers, and loss curves

    The key modification vs the original version is the peak-weighted loss:
      - Peak detection and weighting is now computed per-sample (not per-batch)
      - Optional soft (sigmoid) weighting is introduced to avoid hard 0/1 jumps
      - Weight normalization is done per-sample to prevent samples in the same batch
        from influencing each other’s peak weights

------------------------------------------------------------
Inputs:
    Data (from previous pipeline steps):
      - GM inputs:
          Output/3_GM_Fixed_train_linear/H*/...
          Output/3_GM_Fixed_train_nonlinear/H*/...
        Files:
          - X_data_H*.npy
          - X_data_cluster_balanced_global_H*.npy  (if clustered)
      - THA outputs:
          Output/3_THA_Fixed_train_linear/H*/...
          Output/3_THA_Fixed_train_nonlinear/H*/...
        Files:
          - Y_data_H*.npy
          - Y_data_cluster_balanced_global_H*.npy  (if clustered)

    Runtime user inputs:
      - Linear vs Nonlinear (1/0)
      - Clustered vs Non-clustered (1/0)
      - Optional: cluster K label (for naming output folder only)
      - Heights to include (e.g., H2 H3 ... or numbers)
      - Training mode: Multi-height + height feature vs per-height

    Scenario hyperparameters (SCENARIOS list):
      - EPOCHS
      - ALPHA  (peak emphasis strength)
      - THRESH (relative peak threshold in [0..1] based on |y|/max(|y|))
      - WEIGHT_MODE (1=binary, 2=soft-sigmoid)
      - TAU (sigmoid temperature controlling softness)

------------------------------------------------------------
Outputs:
    Saved under:
      Output/Progress_of_LSTM_linear/<cluster|noCluster>/...
      Output/Progress_of_LSTM_nonlinear/<cluster|noCluster>/...

    For each scenario:
      - Best model checkpoint:
          LSTM.keras
      - Periodic checkpoints:
          checkpoints/ckpt_epoch_XXXX.keras
      - Backup state for safe resume:
          backup/...
      - Training log:
          progress.npy  (loss curves + metadata)
      - Loss plot:
          loss_curve_<scenario>.png
    Also:
      - Scalers (joblib):
          scaler_X_linear.pkl / scaler_Y_linear.pkl
          scaler_X_nonlinear.pkl / scaler_Y_nonlinear.pkl
      - Deterministic split indices:
          split_idx_seed1234.npy

------------------------------------------------------------
Changes since previous version:
    1) Peak-weighted loss fixed to be per-sample (NOT per-batch):
       - Original:
           max_abs = reduce_max(abs_y)  (computed over the whole batch)
           peak_mask = abs_y >= thresh * max_abs  (binary)
           w_sum/mask_sum normalization over whole batch
       - New:
           max_abs = reduce_max(abs_y, axis=1, keepdims=True)  (per-sample)
           peak_strength:
               WEIGHT_MODE=1 -> binary mask
               WEIGHT_MODE=2 -> sigmoid((rel - thresh)/tau)  (soft)
           w_sum/m_sum normalization computed per-sample

       ✅ Why this was done:
       - In the original code, if one sample in a batch has a very large peak,
         max_abs becomes large for the whole batch → other samples’ peaks may
         fail the threshold and receive no extra weight → systematic underprediction
         of peaks (especially for higher-height cases where peak behavior is rarer).

    2) Added two explicit hyperparameters to SCENARIOS:
       - WEIGHT_MODE (1/2) and TAU
       ✅ Why:
       - Gives controlled switch between original “hard peak mask” behavior
         and a smoother version that encourages learning the neighborhood
         around peaks (reduces peak flattening).

    3) Output root directory for training artifacts moved to Output/:
       - base_model_root now points to Output/Progress_of_LSTM_* instead of
         being stored next to the script.
       ✅ Why:
       - Prevents overwriting/fragmentation across code versions and keeps
         all experiment outputs centralized in Output for traceability.

------------------------------------------------------------
Impact of changes:
    - More reliable peak emphasis during training:
        * Peaks are detected relative to each individual time series (per sample),
          not distorted by other batch members.
        * Soft weighting (sigmoid) can improve learning of peak neighborhoods
          and reduce systematic peak underestimation.
    - Better comparability and reproducibility:
        * Deterministic splits preserved
        * Scenario naming includes weight mode & tau to avoid mixing results
    - Cleaner experiment management:
        * Outputs are organized under Output/Progress_of_LSTM_* consistently

------------------------------------------------------------
Status:
    Stable (modified loss behavior tested conceptually; further validation recommended)

------------------------------------------------------------
Notes:
    - THRESH in this version is still relative to per-sample max(|y_true|).
      (Quantile/Top-K% peak definition and oversampling are NOT yet included here.)
    - Suggested starting settings:
        WEIGHT_MODE=2, TAU=0.05, THRESH≈0.7~0.85, ALPHA≈3~8
    - For extreme class imbalance in peaks, consider adding:
        (A) Quantile/Top-K% peak definition
        (B) Oversampling of peak-containing sequences
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

# 🔒 ثابت کردن سیدها برای تکرارپذیری
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

# ============================================================== #
# 🚀 GPU dynamic memory
# ============================================================== #
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory set dynamically.")
    except RuntimeError as e:
        print(e)

# ============================================================== #
# 📁 Paths + انتخاب خطی / غیرخطی
#   ✅ فقط تغییر این بخش: خروجی Progress_of_LSTM_* به Output منتقل شد
# ============================================================== #
base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(base_dir, os.pardir))

# ✅ مسیر جدید برای ذخیره مدل‌ها داخل Output (اسم پوشه همان می‌ماند)
output_root_dir = os.path.join(root_dir, "Output")

choice = input(
    "آموزش بر اساس پاسخ خطی باشد یا غیرخطی؟ "
    "برای خطی عدد 1 و برای غیرخطی عدد 0 را وارد کن: "
).strip()
is_linear = (choice == "1")

if is_linear:
    print("📌 آموزش بر اساس داده‌های خطی (THA_linear) انجام می‌شود.")
    gm_root_dir  = os.path.join(root_dir, "Output", "3_GM_Fixed_train_linear")
    tha_root_dir = os.path.join(root_dir, "Output", "3_THA_Fixed_train_linear")

    # ✅ قبلاً: base_model_root = os.path.join(base_dir, "Progress_of_LSTM_linear")
    base_model_root = os.path.join(output_root_dir, "Progress_of_LSTM_linear")
else:
    print("📌 آموزش بر اساس داده‌های غیرخطی (THA_nonlinear) انجام می‌شود.")
    gm_root_dir  = os.path.join(root_dir, "Output", "3_GM_Fixed_train_nonlinear")
    tha_root_dir = os.path.join(root_dir, "Output", "3_THA_Fixed_train_nonlinear")

    # ✅ قبلاً: base_model_root = os.path.join(base_dir, "Progress_of_LSTM_nonlinear")
    base_model_root = os.path.join(output_root_dir, "Progress_of_LSTM_nonlinear")

os.makedirs(base_model_root, exist_ok=True)

print("📂 GM root dir :", gm_root_dir)
print("📂 THA root dir:", tha_root_dir)
print("📂 Base model root:", base_model_root)
print()

# ============================================================== #
# 🔁 انتخاب استفاده از داده‌های کلاسترشده یا اصلی
# ============================================================== #
print("-------------------------------------------")
print("آیا می‌خواهی از داده‌های *کلاسترشده* استفاده کنم؟")
print("   1 = بله، از فایل‌های cluster_balanced_global استفاده کن")
print("   0 = خیر، از فایل‌های اصلی X_data_H* و Y_data_H* استفاده کن")
print("-------------------------------------------\n")

cluster_choice = input("انتخابت را وارد کن (1 یا 0): ").strip()
USE_CLUSTERED = (cluster_choice == "1")

if USE_CLUSTERED:
    print("✅ استفاده از داده‌های کلاسترشده (cluster_balanced_global).")

    # ⚠️ فقط برای نام‌گذاری پوشهٔ خروجی؛ روی داده‌ها اثری ندارد
    cluster_label = input(
        "برای اسم پوشهٔ مدل، تعداد کلاسترها (K) را بنویس "
        "(مثلاً 4). اگر خالی بگذاری، به‌صورت کلی 'clustered' نوشته می‌شود: "
    ).strip()

    if cluster_label:
        mode_folder_name = f"clusterK{cluster_label}_allHeights"
    else:
        mode_folder_name = "clustered_allHeights"
else:
    print("✅ استفاده از داده‌های اصلی بدون کلاسترینگ.")
    mode_folder_name = "noCluster_allHeights"

# 📂 ریشهٔ واقعی خروجی مدل‌ها (وابسته به کلاستر / بدون کلاستر)
root_model_dir = os.path.join(base_model_root, mode_folder_name)
os.makedirs(root_model_dir, exist_ok=True)

print("\n📂 Root model dir for this run:")
print("   ", root_model_dir)
print()

# ============================================================== #
# 🔧 سناریوهای آنالیز حساسیت
#   ✅ اضافه شد: WEIGHT_MODE و TAU برای وزن‌دهی نرم پیک
# ============================================================== #
SCENARIOS = [
    # {"EPOCHS": 20, "ALPHA": 1.0, "THRESH": 0.5, "WEIGHT_MODE": 1, "TAU": 0.05},
    {"EPOCHS": 76, "ALPHA": 1, "THRESH": 0.5, "WEIGHT_MODE": 2, "TAU": 0.05},
    {"EPOCHS": 100, "ALPHA": 1, "THRESH": 0.5, "WEIGHT_MODE": 2, "TAU": 0.05},
    # {"EPOCHS": 20, "ALPHA": 3.0, "THRESH": 0.5, "WEIGHT_MODE": 2, "TAU": 0.05},
]

# ============================================================== #
# 🧭 بررسی وجود پوشه‌های داده
# ============================================================== #
if not os.path.isdir(gm_root_dir):
    raise FileNotFoundError(f"❌ GM root dir پیدا نشد: {gm_root_dir}")
if not os.path.isdir(tha_root_dir):
    raise FileNotFoundError(f"❌ THA root dir پیدا نشد: {tha_root_dir}")

available_heights = sorted(
    name for name in os.listdir(gm_root_dir)
    if os.path.isdir(os.path.join(gm_root_dir, name)) and name.startswith("H")
)

if not available_heights:
    raise ValueError(
        f"❌ هیچ پوشه‌ای به شکل H* در {gm_root_dir} پیدا نشد. مطمئن شو مرحله ۳ را اجرا کرده‌ای."
    )

print("📏 ارتفاع‌های موجود در داده‌ها:")
for h in available_heights:
    print("  -", h)

print("\n-------------------------------------------")
print("برای چه ارتفاع‌هایی می‌خواهی آموزش انجام شود؟")
print("🔹 می‌توانی به این شکل وارد کنی:")
print("   H2 H3 H4")
print("   یا 2 3 4")
print("   یا ترکیبی:  H2 3 4.5")
print("اگر خالی بگذاری، برای همه ارتفاع‌های موجود آموزش انجام می‌شود.")
print("-------------------------------------------\n")

heights_raw = input("ارتفاع‌ها را وارد کن (خالی = همه): ").strip()

if not heights_raw:
    height_tags = available_heights[:]   # همه‌ی ارتفاع‌ها
    print("\n✅ همه ارتفاع‌های موجود انتخاب شدند.")
else:
    tokens = heights_raw.replace(',', ' ').split()
    selected = []
    invalid  = []

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

# ---------------------------------------------------------- #
# کمکی برای تبدیل H-tag به عدد
# ---------------------------------------------------------- #
def height_value_from_tag(h_tag: str) -> float:
    """
    'H2'   -> 2.0
    'H3p5' -> 3.5
    """
    s = h_tag[1:]  # حذف H
    s = s.replace('p', '.')
    return float(s)

height_values = {h_tag: height_value_from_tag(h_tag) for h_tag in height_tags}
print("🔢 نگاشت ارتفاع‌ها (tag → value):")
for h_tag, hv in height_values.items():
    print(f"  {h_tag} → {hv}")

# ============================================================== #
# ↔️ انتخاب مود آموزش: همه ارتفاع‌ها با هم / هر ارتفاع جدا
# ============================================================== #
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

# 📂 زیرپوشهٔ مخصوص «آموزش کلی با یادگیری ارتفاع» (وابسته به کلاستر)
global_multi_root_dir = os.path.join(root_model_dir, "Global_training_with_height")
os.makedirs(global_multi_root_dir, exist_ok=True)

# ============================================================== #
# ⚙️ توابع عمومی (مشترک)
# ============================================================== #
PAD = -999.0
BATCH_SIZE = 20

def param_to_str(v):
    v = float(v)
    if v.is_integer():
        return f"{int(v)}.0"
    else:
        return str(v)

# ✅✅✅ اصلاح اصلی اینجاست: per-sample max_abs + وزن‌دهی نرم (mode 2)
def make_weighted_mse_loss(PAD_value, alpha, thresh, weight_mode=2, tau=0.05):
    """
    weight_mode:
        1 -> Binary thresholding (قدیمی)
        2 -> Soft sigmoid weighting (جدید، برای یادگیری بهتر پیک‌ها)
    tau:
        دمای سیگموید برای نرمی وزن‌دهی (کوچک‌تر => شبیه‌تر به حالت باینری)
    """
    def weighted_mse_loss(y_true, y_pred):
        PAD_val = tf.constant(PAD_value, dtype=y_true.dtype)

        # y_true: [B, T, 1] (بعد از padded_batch)
        mask = tf.cast(tf.not_equal(y_true, PAD_val), tf.float32)  # [B,T,1]
        abs_y = tf.abs(y_true) * mask                               # [B,T,1]

        # ✅ max_abs برای هر نمونه جدا (نه کل batch)
        max_abs = tf.reduce_max(abs_y, axis=1, keepdims=True) + 1e-6  # [B,1,1]
        rel = abs_y / max_abs                                          # [B,T,1] در بازه [0..1]

        if weight_mode == 1:
            peak_strength = tf.cast(rel >= thresh, tf.float32)          # [B,T,1] صفر/یک
        else:
            # وزن‌دهی نرم: نزدیک thresh آرام زیاد می‌شود
            t = tf.constant(float(tau), dtype=y_true.dtype)
            peak_strength = tf.sigmoid((rel - thresh) / (t + 1e-6))     # [B,T,1] پیوسته

        w = 1.0 + alpha * peak_strength                                 # [B,T,1]

        # ✅ نرمال‌سازی وزن‌ها برای هر نمونه (نه کل batch)
        w_sum = tf.reduce_sum(w * mask, axis=1, keepdims=True) + 1e-6   # [B,1,1]
        m_sum = tf.reduce_sum(mask, axis=1, keepdims=True) + 1e-6       # [B,1,1]
        w = w * (m_sum / w_sum)

        sq = tf.square(y_true - y_pred) * mask                           # [B,T,1]
        loss_per_sample = tf.reduce_sum(w * sq, axis=1) / m_sum          # [B,1,1]
        return tf.reduce_mean(loss_per_sample)                           # scalar

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
    adam = Adam(learning_rate=0.001)
    model.compile(loss=loss_fn, optimizer=adam, metrics=['mse'])
    return model

class PeriodicSaver(tf.keras.callbacks.Callback):
    def __init__(self, save_dir, backup_dir=None, period=10, keep_last=1):
        super().__init__()
        self.save_dir = save_dir
        self.backup_dir = backup_dir
        self.period = period
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

# ---------------------------------------------------------- #
# کمکی: ساخت مسیر داده برای هر ارتفاع بسته به USE_CLUSTERED
# ---------------------------------------------------------- #
def get_data_paths_for_height(h_tag):
    """
    اگر USE_CLUSTERED=True → سراغ cluster_balanced_global می‌رود.
    در غیر این صورت → فایل‌های اصلی X_data_H* و Y_data_H*.
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

    x_path = os.path.join(gm_dir,  x_name)
    y_path = os.path.join(tha_dir, y_name)
    return x_path, y_path

# ============================================================== #
# 🔁 مود ۱: همهٔ ارتفاع‌ها با هم + فیچر ارتفاع
# ============================================================== #
if USE_MULTI_HEIGHT:
    INPUT_DIM = 2  # [GM , H]

    X_all = []
    Y_all = []

    for h_tag in height_tags:
        print("\n" + "#" * 80)
        print(f"📥 بارگذاری داده برای ارتفاع: {h_tag}")
        print("#" * 80)

        x_data_path, y_data_path = get_data_paths_for_height(h_tag)

        if not os.path.exists(x_data_path):
            print(f"⚠️ دادهٔ X برای {h_tag} پیدا نشد: {x_data_path} → این ارتفاع رد می‌شود.")
            continue
        if not os.path.exists(y_data_path):
            print(f"⚠️ دادهٔ Y برای {h_tag} پیدا نشد: {y_data_path} → این ارتفاع رد می‌شود.")
            continue

        X_dict = np.load(x_data_path, allow_pickle=True).item()
        Y_dict = np.load(y_data_path, allow_pickle=True).item()

        common_keys = sorted(set(X_dict.keys()) & set(Y_dict.keys()))
        if not common_keys:
            print(f"❌ هیچ کلید مشترکی بین X و Y برای {h_tag} پیدا نشد. این ارتفاع رد می‌شود.")
            continue

        h_val = np.float32(height_values[h_tag])

        for k in common_keys:
            x = X_dict[k].reshape(-1, 1).astype('float32')   # GM
            y = Y_dict[k].reshape(-1, 1).astype('float32')   # پاسخ

            T = x.shape[0]
            h_col = np.full((T, 1), h_val, dtype='float32')  # ستون دوم: ارتفاع ثابت

            x_feat = np.concatenate([x, h_col], axis=1)      # [GM , H]
            X_all.append(x_feat)
            Y_all.append(y)

    num_samples = len(X_all)
    if num_samples == 0:
        raise ValueError("❌ هیچ نمونه‌ای برای آموزش پیدا نشد (بعد از فیلتر ارتفاع‌ها / کلاسترها).")

    print(f"\n📦 تعداد کل نمونه‌ها (همه ارتفاع‌ها با هم): {num_samples}")

    # 🔢 Scaling سراسری
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    scaler_Y = MinMaxScaler(feature_range=(-1, 1))

    X_concat = np.concatenate(X_all, axis=0)
    Y_concat = np.concatenate(Y_all, axis=0)

    scaler_X.fit(X_concat)
    scaler_Y.fit(Y_concat)

    X_all = [scaler_X.transform(x) for x in X_all]
    Y_all = [scaler_Y.transform(y) for y in Y_all]

    # ⬇ اسکیلرها برای مدل کلی در global_multi_root_dir ذخیره می‌شوند
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

    # 🧩 Split سراسری ثابت
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

    X_train_list = [X_all[i] for i in idx[:train_split]]
    Y_train_list = [Y_all[i] for i in idx[:train_split]]

    X_val_list   = [X_all[i] for i in idx[train_split:val_split]]
    Y_val_list   = [Y_all[i] for i in idx[train_split:val_split]]

    X_test_list  = [X_all[i] for i in idx[val_split:]]
    Y_test_list  = [Y_all[i] for i in idx[val_split:]]

    print(f"\n📦 Global split | Train: {len(X_train_list)}  Val: {len(X_val_list)}  Test: {len(X_test_list)}")

    if len(X_train_list) == 0 or len(X_val_list) == 0:
        raise ValueError("⚠️ دادهٔ کافی برای train/val وجود ندارد.")

    heights_str = ", ".join(height_tags)

    # دیتاست‌ساز بر اساس لیست‌ها
    def gen_from_lists(x_list, y_list):
        for x, y in zip(x_list, y_list):
            yield x, y

    output_signature = (
        tf.TensorSpec(shape=(None, INPUT_DIM), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
    )

    def make_datasets(X_train_list, Y_train_list, X_val_list, Y_val_list):
        ds_train = tf.data.Dataset.from_generator(
            lambda: gen_from_lists(X_train_list, Y_train_list),
            output_signature=output_signature
        ).padded_batch(
            BATCH_SIZE,
            padded_shapes=([None, INPUT_DIM], [None, 1]),
            padding_values=(PAD, PAD),
            drop_remainder=False
        ).prefetch(tf.data.AUTOTUNE)

        ds_val = tf.data.Dataset.from_generator(
            lambda: gen_from_lists(X_val_list, Y_val_list),
            output_signature=output_signature
        ).padded_batch(
            BATCH_SIZE,
            padded_shapes=([None, INPUT_DIM], [None, 1]),
            padding_values=(PAD, PAD),
            drop_remainder=False
        ).prefetch(tf.data.AUTOTUNE)

        return ds_train, ds_val

    # 🔁 حلقه روی سناریوها
    for scen in SCENARIOS:
        EPOCHS = int(scen["EPOCHS"])
        ALPHA  = float(scen["ALPHA"])
        THRESH = float(scen["THRESH"])
        WEIGHT_MODE = int(scen.get("WEIGHT_MODE", 2))
        TAU = float(scen.get("TAU", 0.05))

        scen_name = f"ep{EPOCHS}_A{param_to_str(ALPHA)}_T{param_to_str(THRESH)}_M{WEIGHT_MODE}_tau{param_to_str(TAU)}"

        model_dir = os.path.join(global_multi_root_dir, scen_name)
        os.makedirs(model_dir, exist_ok=True)

        model_path    = os.path.join(model_dir, "LSTM.keras")
        backup_dir    = os.path.join(model_dir, "backup")
        ckpt_dir      = os.path.join(model_dir, "checkpoints")
        progress_path = os.path.join(model_dir, "progress.npy")

        if is_scenario_finished(model_dir, EPOCHS):
            print("\n" + "=" * 80)
            print(f"⏩ سناریو {scen_name} قبلاً کامل شده است. رد می‌شود.")
            print("=" * 80)
            continue

        print("\n" + "=" * 80)
        print(f"🚀 شروع سناریو (Global + Height): {scen_name}")
        print(f"   → EPOCHS = {EPOCHS}, ALPHA = {ALPHA}, THRESH = {THRESH}, WEIGHT_MODE = {WEIGHT_MODE}, TAU = {TAU}")
        print(f"   → Heights used: {heights_str}")
        print(f"   → Cluster mode folder: {mode_folder_name}")
        print("=" * 80)

        os.makedirs(ckpt_dir, exist_ok=True)

        ds_train, ds_val = make_datasets(
            X_train_list, Y_train_list,
            X_val_list,   Y_val_list
        )

        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        loss_fn = make_weighted_mse_loss(PAD, ALPHA, THRESH, weight_mode=WEIGHT_MODE, tau=TAU)
        model = build_model(INPUT_DIM, PAD, loss_fn)
        print(model.summary())

        checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor='val_loss', save_best_only=True, verbose=1
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10, verbose=1, min_lr=1e-6
        )

        backup_cb = tf.keras.callbacks.BackupAndRestore(backup_dir=backup_dir)

        periodic_ckpt = PeriodicSaver(
            save_dir=ckpt_dir,
            backup_dir=backup_dir,
            period=10,
            keep_last=1
        )

        initial_epoch = 0
        if not os.path.exists(backup_dir) or not os.listdir(backup_dir):
            ckpt_path, ep = find_latest_periodic_checkpoint(ckpt_dir)
            if ckpt_path:
                try:
                    tmp_model = load_model(ckpt_path, compile=False)
                    adam = Adam(learning_rate=0.001)
                    tmp_model.compile(loss=loss_fn, optimizer=adam, metrics=['mse'])
                    model = tmp_model
                    initial_epoch = ep
                    print(f"🔁 Resuming from periodic checkpoint: {ckpt_path} (initial_epoch={initial_epoch})")
                except Exception as e:
                    print(f"⚠️ Failed to load periodic checkpoint: {e}")

        print("🚀 Training started (global, all heights)")
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
            'weight_mode': WEIGHT_MODE,
            'tau': TAU,
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

# ============================================================== #
# 🔁 مود ۲: مدل جداگانه برای هر ارتفاع (بدون فیچر H)
# ============================================================== #
else:
    INPUT_DIM = 1  # فقط GM

    for h_tag in height_tags:
        print("\n" + "#" * 80)
        print(f"🏗️ شروع آموزش جداگانه برای ارتفاع: {h_tag}")
        print("#" * 80)

        x_data_path, y_data_path = get_data_paths_for_height(h_tag)

        if not os.path.exists(x_data_path):
            print(f"⚠️ X_data برای {h_tag} پیدا نشد: {x_data_path} → این ارتفاع رد می‌شود.")
            continue
        if not os.path.exists(y_data_path):
            print(f"⚠️ Y_data برای {h_tag} پیدا نشد: {y_data_path} → این ارتفاع رد می‌شود.")
            continue

        X_dict = np.load(x_data_path, allow_pickle=True).item()
        Y_dict = np.load(y_data_path, allow_pickle=True).item()

        common_keys = sorted(set(X_dict.keys()) & set(Y_dict.keys()))
        if not common_keys:
            print(f"❌ هیچ کلید مشترکی بین X و Y برای {h_tag} پیدا نشد. این ارتفاع رد می‌شود.")
            continue

        X_all = []
        Y_all = []
        for k in common_keys:
            x = X_dict[k].reshape(-1, 1).astype('float32')   # فقط GM
            y = Y_dict[k].reshape(-1, 1).astype('float32')
            X_all.append(x)
            Y_all.append(y)

        num_samples = len(X_all)
        if num_samples == 0:
            print(f"⚠️ هیچ نمونه‌ای برای {h_tag} وجود ندارد. این ارتفاع رد می‌شود.")
            continue

        print(f"📦 تعداد نمونه‌ها برای {h_tag}: {num_samples}")

        # اسکیلینگ مخصوص این ارتفاع
        scaler_X = MinMaxScaler(feature_range=(-1, 1))
        scaler_Y = MinMaxScaler(feature_range=(-1, 1))

        X_concat = np.concatenate(X_all, axis=0)
        Y_concat = np.concatenate(Y_all, axis=0)

        scaler_X.fit(X_concat)
        scaler_Y.fit(Y_concat)

        X_all = [scaler_X.transform(x) for x in X_all]
        Y_all = [scaler_Y.transform(y) for y in Y_all]

        # مسیر ذخیره اسکیلرها برای این ارتفاع (در پوشه‌ی mode_folder_name)
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

        # اسپلیت ثابت مخصوص این ارتفاع
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

        X_train_list = [X_all[i] for i in idx[:train_split]]
        Y_train_list = [Y_all[i] for i in idx[:train_split]]

        X_val_list   = [X_all[i] for i in idx[train_split:val_split]]
        Y_val_list   = [Y_all[i] for i in idx[train_split:val_split]]

        X_test_list  = [X_all[i] for i in idx[val_split:]]
        Y_test_list  = [Y_all[i] for i in idx[val_split:]]

        print(f"📦 {h_tag} | Train: {len(X_train_list)}  Val: {len(X_val_list)}  Test: {len(X_test_list)}")

        # دیتاست
        def gen_from_lists(x_list, y_list):
            for x, y in zip(x_list, y_list):
                yield x, y

        output_signature = (
            tf.TensorSpec(shape=(None, INPUT_DIM), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        )

        def make_datasets(X_train_list, Y_train_list, X_val_list, Y_val_list):
            ds_train = tf.data.Dataset.from_generator(
                lambda: gen_from_lists(X_train_list, Y_train_list),
                output_signature=output_signature
            ).padded_batch(
                BATCH_SIZE,
                padded_shapes=([None, INPUT_DIM], [None, 1]),
                padding_values=(PAD, PAD),
                drop_remainder=False
            ).prefetch(tf.data.AUTOTUNE)

            ds_val = tf.data.Dataset.from_generator(
                lambda: gen_from_lists(X_val_list, Y_val_list),
                output_signature=output_signature
            ).padded_batch(
                BATCH_SIZE,
                padded_shapes=([None, INPUT_DIM], [None, 1]),
                padding_values=(PAD, PAD),
                drop_remainder=False
            ).prefetch(tf.data.AUTOTUNE)

            return ds_train, ds_val

        # 🔁 حلقه روی سناریوها برای این ارتفاع
        for scen in SCENARIOS:
            EPOCHS = int(scen["EPOCHS"])
            ALPHA  = float(scen["ALPHA"])
            THRESH = float(scen["THRESH"])
            WEIGHT_MODE = int(scen.get("WEIGHT_MODE", 2))
            TAU = float(scen.get("TAU", 0.05))

            scen_name = f"ep{EPOCHS}_A{param_to_str(ALPHA)}_T{param_to_str(THRESH)}_M{WEIGHT_MODE}_tau{param_to_str(TAU)}"

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
            print(f"   → EPOCHS = {EPOCHS}, ALPHA = {ALPHA}, THRESH = {THRESH}, WEIGHT_MODE = {WEIGHT_MODE}, TAU = {TAU}")
            print(f"   → Cluster mode folder: {mode_folder_name}")
            print("=" * 80)

            os.makedirs(ckpt_dir, exist_ok=True)

            ds_train, ds_val = make_datasets(
                X_train_list, Y_train_list,
                X_val_list,   Y_val_list
            )

            tf.random.set_seed(SEED)
            np.random.seed(SEED)
            random.seed(SEED)

            loss_fn = make_weighted_mse_loss(PAD, ALPHA, THRESH, weight_mode=WEIGHT_MODE, tau=TAU)
            model = build_model(INPUT_DIM, PAD, loss_fn)
            print(model.summary())

            checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
                model_path, monitor='val_loss', save_best_only=True, verbose=1
            )

            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=10, verbose=1, min_lr=1e-6
            )

            backup_cb = tf.keras.callbacks.BackupAndRestore(backup_dir=backup_dir)

            periodic_ckpt = PeriodicSaver(
                save_dir=ckpt_dir,
                backup_dir=backup_dir,
                period=10,
                keep_last=1
            )

            initial_epoch = 0
            if not os.path.exists(backup_dir) or not os.listdir(backup_dir):
                ckpt_path, ep = find_latest_periodic_checkpoint(ckpt_dir)
                if ckpt_path:
                    try:
                        tmp_model = load_model(ckpt_path, compile=False)
                        adam = Adam(learning_rate=0.001)
                        tmp_model.compile(loss=loss_fn, optimizer=adam, metrics=['mse'])
                        model = tmp_model
                        initial_epoch = ep
                        print(f"🔁 {h_tag} | Resuming from checkpoint: {ckpt_path} (initial_epoch={initial_epoch})")
                    except Exception as e:
                        print(f"⚠️ Failed to load periodic checkpoint: {e}")

            print(f"🚀 Training started ({h_tag})")
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
                'weight_mode': WEIGHT_MODE,
                'tau': TAU,
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
