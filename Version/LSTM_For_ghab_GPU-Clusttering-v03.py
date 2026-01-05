# =============================================================================
# 
# 
# 
# 
# # -*- coding: utf-8 -*-
# """
# File name      : LSTM_For_ghab_GPU-Clusttering-v03.py
# Author         : pc22
# Created on     : Sun Dec 28 08:47:13 2025
# Last modified  : Wed Dec 31 08:xx:xx 2025
# ------------------------------------------------------------
# Purpose:
#     Train an LSTM-based sequence-to-sequence model to predict
#     structural time-history responses (THA) from ground-motion (GM)
#     time series, with explicit emphasis on improving prediction
#     quality at extreme response regions (peaks).
# 
#     This version introduces a robust peak-aware training strategy:
#       (1) Peak definition per-sample using either:
#             - Relative thresholding (thresh × max of each sequence), or
#             - Quantile-based Top-K% thresholding (per-sample quantile q)
#       (2) Oversampling of peak-containing sequences to mitigate
#           peak rarity in the training set
#       (3) Safe resume/recovery via BackupAndRestore + periodic checkpoints
#       (4) Script-tagged, compact output folders to avoid overwriting
#           and improve experiment traceability across code versions.
# 
# ------------------------------------------------------------
# Description:
#     The script loads preprocessed GM/THA time-series pairs generated
#     by previous pipeline steps, scales inputs/outputs, and trains an
#     LSTM model under user-selected modes:
# 
#       - Linear vs Nonlinear training:
#           * Linear   → uses THA_linear dataset folders
#           * Nonlinear→ uses THA_nonlinear dataset folders
# 
#       - Clustered vs Non-clustered data:
#           * Clustered    → uses cluster_balanced_global subfolders
#           * Non-clustered→ uses original X_data_H* / Y_data_H* files
# 
#       - Two training modes:
#           (1) Global multi-height training (USE_MULTI_HEIGHT=1):
#               A shared model is trained on all selected heights
#               with height added as an additional input feature:
#                   X = [GM(t), H]
#           (2) Per-height independent training (USE_MULTI_HEIGHT=0):
#               Separate models are trained per height:
#                   X = [GM(t)]
# 
#     Variable-length time series are handled using padding and a
#     masking layer (PAD = -999.0). Training datasets are built with
#     tf.data and padded_batch, enabling mini-batch training on
#     sequences of varying length.
# 
#     Peak-aware learning:
#       - Weighted MSE loss is used, where sample/time-step weights
#         are increased at peak regions using:
#           w = 1 + alpha * peak_strength
#         and peak_strength is computed either by:
#           * Binary thresholding (WEIGHT_MODE=1), or
#           * Soft sigmoid weighting (WEIGHT_MODE=2) with temperature TAU
# 
#       - Oversampling is optionally applied to the training set only:
#         sequences that contain peaks are replicated OVERSAMPLE_FACTOR
#         times to increase peak frequency during optimization.
# 
#     Experiment management:
#       - Outputs are organized under a short, version-tagged script folder
#         (e.g., V03) to avoid overwriting between code variants.
#       - Each scenario run creates an isolated run folder and logs:
#           * best model (LSTM.keras)
#           * periodic checkpoints
#           * backup state for auto-resume
#           * progress.npy (training curves + metadata)
#           * runinfo.txt (human-readable run configuration)
#           * loss curve image
# 
# ------------------------------------------------------------
# Inputs:
#     Data (from previous pipeline steps):
#       - GM time series:
#           Output/3_GM_Fixed_train_linear/H*/...
#           Output/3_GM_Fixed_train_nonlinear/H*/...
#       - THA time series:
#           Output/3_THA_Fixed_train_linear/H*/...
#           Output/3_THA_Fixed_train_nonlinear/H*/...
# 
#     Runtime user inputs (interactive):
#       - Linear vs Nonlinear mode selection
#       - Clustered vs Non-clustered dataset selection
#       - Heights to include in training (H2, H3, ...)
#       - Training mode: global multi-height vs per-height models
#       - Peak definition method: relative vs quantile
#       - Quantile q (if quantile mode)
#       - Oversampling factor (integer ≥ 1)
# 
#     Scenario hyperparameters (SCENARIOS list):
#       - EPOCHS
#       - ALPHA              (peak emphasis strength)
#       - THRESH             (relative threshold; used if PEAK_METHOD="relative")
#       - WEIGHT_MODE         (1 = binary, 2 = soft sigmoid)
#       - TAU                 (sigmoid temperature; used if WEIGHT_MODE=2)
# 
#     Peak strategy parameters:
#       - PEAK_METHOD         ("relative" or "quantile")
#       - PEAK_Q              (quantile q; used if PEAK_METHOD="quantile")
#       - OVERSAMPLE_FACTOR   (integer ≥ 1; 1 disables oversampling)
# 
# ------------------------------------------------------------
# Outputs:
#     All outputs are saved under:
# 
#       Output/
#         ├── Progress_of_LSTM_linear/
#         │     └── <script_tag>/                 (e.g., V03)
#         │           ├── _META_<MODE>-<TRAIN>/   (global scalers + split indices)
#         │           ├── <run_tag>/              (one folder per scenario)
#         │           │     ├── LSTM.keras
#         │           │     ├── checkpoints/
#         │           │     ├── backup/
#         │           │     ├── progress.npy
#         │           │     ├── runinfo.txt
#         │           │     └── loss_curve_*.png
#         │           └── ...
#         └── Progress_of_LSTM_nonlinear/
#               └── <script_tag>/
#                     ...
# 
#     where:
#       - <script_tag> is inferred from the executing file name:
#             * vXX is extracted if present (e.g., "v03" → "V03")
#             * otherwise a short alphanumeric prefix is used
#       - <run_tag> is a compact code describing mode + scenario, e.g.:
#             NC-GH__E76-A1-Q70-OS3-M1-T05
#         (cluster/noCluster + training mode + scenario params)
# 
# ------------------------------------------------------------
# Changes since baseline / earlier code:
#     1) Peak-aware loss redesigned to support:
#          - Relative peaks (thresh × max per-sample)
#          - Quantile peaks (Top-K% per-sample)
#          - Binary vs soft weighting (sigmoid) controlled by WEIGHT_MODE/TAU
#     2) Oversampling of peak-containing sequences added (train set only).
#     3) Safe resume improved:
#          - BackupAndRestore enabled
#          - Periodic checkpoints saved every N epochs
#          - Optional resume from last periodic checkpoint if backup is absent
#     4) Output management updated:
#          - Version-tagged (<script_tag>) parent folder prevents overwriting
#          - Compact run tags improve readability and path length control
#          - runinfo.txt is written alongside progress.npy for traceability
# 
# ------------------------------------------------------------
# Impact of changes:
#     - Improves prediction accuracy at extreme response regions by
#       increasing gradient emphasis on peaks and reducing peak rarity.
#     - Enhances experiment reproducibility via fixed random seeds,
#       deterministic ops (where supported), and fixed split indices.
#     - Improves run traceability and reduces accidental overwriting by
#       isolating outputs per code version and scenario.
# 
# ------------------------------------------------------------
# Status:
#     Stable (training + resume supported)
# 
# ------------------------------------------------------------
# Notes:
#     - Quantile-based peak thresholding is always computed per individual
#       time series (never global).
#     - Oversampling is applied only to the training split; validation/test
#       distributions remain unchanged.
#     - Setting PEAK_METHOD="relative" and OVERSAMPLE_FACTOR=1 approximates
#       the original baseline behavior.
#     - On Windows, path length limitations can affect BackupAndRestore if
#       output roots are deeply nested; prefer short base output roots or
#       dedicated short backup roots if needed.
# """
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# import sys, io
# if hasattr(sys.stdout, "buffer"):
#     sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
# 
# import os, gc, glob, re, shutil, random
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.layers import Input, Masking, LSTM, Activation, Dense
# from tensorflow.keras.optimizers import Adam
# from sklearn.preprocessing import MinMaxScaler
# import joblib
# 
# plt.ioff()
# 
# # ==============================================================
# # 🔒 Reproducibility / Determinism
# # ==============================================================
# SEED = 1234
# os.environ["PYTHONHASHSEED"] = str(SEED)
# os.environ["TF_DETERMINISTIC_OPS"] = "1"
# 
# random.seed(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)
# try:
#     tf.keras.utils.set_random_seed(SEED)
# except Exception:
#     pass
# try:
#     tf.config.experimental.enable_op_determinism(True)
# except Exception:
#     pass
# 
# # ==============================================================
# # 🚀 GPU dynamic memory
# # ==============================================================
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         print("✅ GPU memory set dynamically.")
#     except RuntimeError as e:
#         print(e)
# 
# # ==============================================================
# # 📁 Paths + Linear / Nonlinear selection
# # ==============================================================
# base_dir = os.path.dirname(os.path.abspath(__file__))
# root_dir = os.path.abspath(os.path.join(base_dir, os.pardir))
# 
# output_root_dir = os.path.join(root_dir, "Output")
# 
# choice = input(
#     "آموزش بر اساس پاسخ خطی باشد یا غیرخطی؟ "
#     "برای خطی عدد 1 و برای غیرخطی عدد 0 را وارد کن: "
# ).strip()
# is_linear = (choice == "1")
# 
# if is_linear:
#     print("📌 آموزش بر اساس داده‌های خطی (THA_linear) انجام می‌شود.")
#     gm_root_dir  = os.path.join(output_root_dir, "3_GM_Fixed_train_linear")
#     tha_root_dir = os.path.join(output_root_dir, "3_THA_Fixed_train_linear")
#     base_model_root = os.path.join(output_root_dir, "Progress_of_LSTM_linear")
# else:
#     print("📌 آموزش بر اساس داده‌های غیرخطی (THA_nonlinear) انجام می‌شود.")
#     gm_root_dir  = os.path.join(output_root_dir, "3_GM_Fixed_train_nonlinear")
#     tha_root_dir = os.path.join(output_root_dir, "3_THA_Fixed_train_nonlinear")
#     base_model_root = os.path.join(output_root_dir, "Progress_of_LSTM_nonlinear")
# 
# os.makedirs(base_model_root, exist_ok=True)
# 
# print("📂 GM root dir :", gm_root_dir)
# print("📂 THA root dir:", tha_root_dir)
# print("📂 Base model root:", base_model_root)
# print()
# 
# # ==============================================================
# # 🧾 Script-tag isolation (avoid overwriting across versions) + SHORT
# # ==============================================================
# script_name = os.path.splitext(os.path.basename(__file__))[0]
# 
# # Short tag from script_name (prefer vXX if exists)
# m = re.search(r"(v\d+)", script_name, flags=re.IGNORECASE)
# if m:
#     script_tag = m.group(1).upper()
# else:
#     # fallback: keep it short but stable
#     script_tag = re.sub(r"[^A-Za-z0-9]+", "", script_name)[:12] or "SCRIPT"
# 
# base_model_root = os.path.join(base_model_root, script_tag)
# os.makedirs(base_model_root, exist_ok=True)
# 
# # ==============================================================
# # 🔁 Clustered vs non-clustered
# # ==============================================================
# print("-------------------------------------------")
# print("آیا می‌خواهی از داده‌های *کلاسترشده* استفاده کنم؟")
# print("   1 = بله، از فایل‌های cluster_balanced_global استفاده کن")
# print("   0 = خیر، از فایل‌های اصلی X_data_H* و Y_data_H* استفاده کن")
# print("-------------------------------------------\n")
# 
# cluster_choice = input("انتخابت را وارد کن (1 یا 0): ").strip()
# USE_CLUSTERED = (cluster_choice == "1")
# 
# cluster_label = ""
# if USE_CLUSTERED:
#     print("✅ استفاده از داده‌های کلاسترشده (cluster_balanced_global).")
#     cluster_label = input(
#         "برای اسم پوشهٔ مدل، تعداد کلاسترها (K) را بنویس "
#         "(مثلاً 4). اگر خالی بگذاری، 'clustered' نوشته می‌شود: "
#     ).strip()
# else:
#     print("✅ استفاده از داده‌های اصلی بدون کلاسترینگ.")
# 
# def compact_float_tag(x, scale=100):
#     """
#     0.70 -> '70' (default scale=100)
#     0.05 -> '05' (if scale=100)
#     1.0  -> '100'
#     """
#     try:
#         v = float(x)
#     except Exception:
#         return str(x)
#     n = int(round(v * scale))
#     width = 2 if scale == 100 else 0
#     return f"{n:0{width}d}" if width > 0 else str(n)
# 
# def compact_A_tag(a):
#     try:
#         v = float(a)
#     except Exception:
#         return str(a)
#     if abs(v - int(v)) < 1e-9:
#         return str(int(v))
#     s = str(v).replace(".", "p")
#     return s
# 
# def make_scen_short(EPOCHS, ALPHA, peak_method, PEAK_Q, THRESH, OVERSAMPLE_FACTOR, WEIGHT_MODE, TAU):
#     e = int(EPOCHS)
#     a = compact_A_tag(ALPHA)
#     if peak_method == "quantile":
#         q = compact_float_tag(PEAK_Q, scale=100)
#         peak = f"Q{q}"
#     else:
#         r = compact_float_tag(THRESH, scale=100)
#         peak = f"R{r}"
#     osf = int(OVERSAMPLE_FACTOR)
#     m_ = int(WEIGHT_MODE)
#     t = compact_float_tag(TAU, scale=100)
#     return f"E{e}-A{a}-{peak}-OS{osf}-M{m_}-T{t}"
# 
# def get_mode_code():
#     if not USE_CLUSTERED:
#         return "NC"
#     if cluster_label:
#         # keep short: CK4
#         return f"CK{cluster_label}"
#     return "CL"
# 
# def write_runinfo(run_dir, info: dict):
#     try:
#         p = os.path.join(run_dir, "runinfo.txt")
#         lines = []
#         lines.append("===== RUN INFO =====")
#         for k in sorted(info.keys()):
#             lines.append(f"{k}: {info[k]}")
#         with open(p, "w", encoding="utf-8") as f:
#             f.write("\n".join(lines) + "\n")
#     except Exception as e:
#         print(f"⚠️ Could not write runinfo.txt in {run_dir}: {e}")
# 
# MODE_CODE = get_mode_code()
# 
# print("\n📂 Base model root for this code version:")
# print("   ", base_model_root)
# print()
# 
# # ==============================================================
# # 🧠 Peak definition + Oversampling inputs
# # ==============================================================
# print("-------------------------------------------")
# print("روش تعریف پیک را انتخاب کن:")
# print("1 = relative (thresh × max هر تایم‌سری)")
# print("2 = quantile (Top-K% هر تایم‌سری)")
# print("-------------------------------------------")
# _peak_choice = input("انتخاب (1 یا 2): ").strip()
# PEAK_METHOD = "quantile" if _peak_choice == "2" else "relative"
# 
# PEAK_Q = 0.95
# if PEAK_METHOD == "quantile":
#     q_in = input("مقدار q برای Quantile را وارد کن (مثلاً 0.95 یعنی 5% بالایی). خالی = 0.95: ").strip()
#     if q_in:
#         try:
#             PEAK_Q = float(q_in)
#         except Exception:
#             PEAK_Q = 0.95
# 
# os_in = input("Oversampling factor (عدد صحیح >=1؛ 1 یعنی خاموش). خالی = 1: ").strip()
# OVERSAMPLE_FACTOR = 1
# if os_in:
#     try:
#         OVERSAMPLE_FACTOR = int(os_in)
#         if OVERSAMPLE_FACTOR < 1:
#             OVERSAMPLE_FACTOR = 1
#     except Exception:
#         OVERSAMPLE_FACTOR = 1
# 
# # ==============================================================
# # 🔧 Scenarios
# # ==============================================================
# SCENARIOS = [
#     {"EPOCHS": 76, "ALPHA": 1, "THRESH": 0.5, "WEIGHT_MODE": 1, "TAU": 0.05},
# ]
# 
# # ==============================================================
# # 🧭 Data folders check
# # ==============================================================
# if not os.path.isdir(gm_root_dir):
#     raise FileNotFoundError(f"❌ GM root dir پیدا نشد: {gm_root_dir}")
# if not os.path.isdir(tha_root_dir):
#     raise FileNotFoundError(f"❌ THA root dir پیدا نشد: {tha_root_dir}")
# 
# available_heights = sorted(
#     name for name in os.listdir(gm_root_dir)
#     if os.path.isdir(os.path.join(gm_root_dir, name)) and name.startswith("H")
# )
# if not available_heights:
#     raise ValueError(f"❌ هیچ پوشه‌ای به شکل H* در {gm_root_dir} پیدا نشد. مرحله ۳ را چک کن.")
# 
# print("📏 ارتفاع‌های موجود در داده‌ها:")
# for h in available_heights:
#     print("  -", h)
# 
# print("\n-------------------------------------------")
# print("برای چه ارتفاع‌هایی می‌خواهی آموزش انجام شود؟")
# print("مثال: H2 H3 H4  یا  2 3 4  (خالی = همه)")
# print("-------------------------------------------\n")
# 
# heights_raw = input("ارتفاع‌ها را وارد کن (خالی = همه): ").strip()
# 
# if not heights_raw:
#     height_tags = available_heights[:]
#     print("\n✅ همه ارتفاع‌های موجود انتخاب شدند.")
# else:
#     tokens = heights_raw.replace(',', ' ').split()
#     selected, invalid = [], []
#     for tok in tokens:
#         tok = tok.strip()
#         if not tok:
#             continue
#         if tok.startswith("H"):
#             tag = tok
#         else:
#             try:
#                 v = float(tok)
#                 if v.is_integer():
#                     tag = f"H{int(v)}"
#                 else:
#                     tag = "H" + str(v).replace('.', 'p')
#             except Exception:
#                 invalid.append(tok)
#                 continue
#         if tag in available_heights:
#             selected.append(tag)
#         else:
#             invalid.append(tok)
#     height_tags = list(dict.fromkeys(selected))
#     if invalid:
#         print("\n⚠️ این مقادیر معتبر نبودند یا داده‌ای برای آنها پیدا نشد و نادیده گرفته شدند:")
#         for x in invalid:
#             print("  -", x)
#     if not height_tags:
#         print("❌ هیچ ارتفاع معتبری انتخاب نشد؛ برنامه متوقف می‌شود.")
#         sys.exit(1)
# 
# print("\n📏 ارتفاع‌های نهایی انتخاب‌شده برای آموزش:")
# print("  " + ", ".join(height_tags))
# print()
# 
# def height_value_from_tag(h_tag: str) -> float:
#     s = h_tag[1:]
#     s = s.replace('p', '.')
#     return float(s)
# 
# height_values = {h_tag: height_value_from_tag(h_tag) for h_tag in height_tags}
# print("🔢 نگاشت ارتفاع‌ها (tag → value):")
# for h_tag, hv in height_values.items():
#     print(f"  {h_tag} → {hv}")
# 
# # ==============================================================
# # ↔️ Training mode selection
# # ==============================================================
# print("\n-------------------------------------------")
# print("آیا می‌خواهی همهٔ ارتفاع‌ها با هم و با فیچر ارتفاع آموزش داده شوند؟")
# print("   1 = بله، یک مدل مشترک با فیچر ارتفاع (Multi-height + Feature H)")
# print("   0 = خیر، برای هر ارتفاع جداگانه مدل مستقل آموزش داده شود")
# print("-------------------------------------------\n")
# 
# mh_choice = input("انتخابت را وارد کن (1 یا 0): ").strip()
# USE_MULTI_HEIGHT = (mh_choice == "1")
# 
# if USE_MULTI_HEIGHT:
#     print("✅ مود ۱: مدل مشترک برای همه ارتفاع‌ها با فیچر ارتفاع (X = [GM , H])")
#     TRAIN_CODE = "GH"
# else:
#     print("✅ مود ۲: مدل جداگانه برای هر ارتفاع، بدون فیچر ارتفاع (X = [GM])")
#     TRAIN_CODE = "PH"
# 
# # ==============================================================
# # ⚙️ Common settings / helpers
# # ==============================================================
# PAD = -999.0
# BATCH_SIZE = 20
# 
# def param_to_str(v):
#     v = float(v)
#     if v.is_integer():
#         return f"{int(v)}.0"
#     return str(v)
# 
# def get_data_paths_for_height(h_tag):
#     if USE_CLUSTERED:
#         gm_dir  = os.path.join(gm_root_dir,  h_tag, "cluster_balanced_global")
#         tha_dir = os.path.join(tha_root_dir, h_tag, "cluster_balanced_global")
#         x_name  = f"X_data_cluster_balanced_global_{h_tag}.npy"
#         y_name  = f"Y_data_cluster_balanced_global_{h_tag}.npy"
#     else:
#         gm_dir  = os.path.join(gm_root_dir,  h_tag)
#         tha_dir = os.path.join(tha_root_dir, h_tag)
#         x_name  = f"X_data_{h_tag}.npy"
#         y_name  = f"Y_data_{h_tag}.npy"
#     return os.path.join(gm_dir, x_name), os.path.join(tha_dir, y_name)
# 
# # ==============================================================
# # ✅ Peak-aware weighted MSE loss (relative/quantile + binary/soft)
# # ==============================================================
# def make_weighted_mse_loss(PAD_value, alpha, thresh, weight_mode=2, tau=0.05,
#                           peak_method="relative", q=0.95):
#     PAD_val = tf.constant(PAD_value, dtype=tf.float32)
# 
#     def weighted_mse_loss(y_true, y_pred):
#         y_true = tf.cast(y_true, tf.float32)
#         y_pred = tf.cast(y_pred, tf.float32)
# 
#         mask = tf.cast(tf.not_equal(y_true, PAD_val), tf.float32)
#         abs_y = tf.abs(y_true) * mask
# 
#         if peak_method == "relative":
#             max_abs = tf.reduce_max(abs_y, axis=1, keepdims=True) + 1e-6
#             rel = abs_y / max_abs
#             if weight_mode == 1:
#                 peak_strength = tf.cast(rel >= thresh, tf.float32)
#             else:
#                 t = tf.constant(float(tau), dtype=tf.float32)
#                 peak_strength = tf.sigmoid((rel - thresh) / (t + 1e-6))
# 
#         elif peak_method == "quantile":
#             abs_y_2d = tf.squeeze(abs_y, axis=-1)
#             sorted_vals = tf.sort(abs_y_2d, axis=1)
#             T = tf.shape(sorted_vals)[1]
#             q_clamped = tf.clip_by_value(tf.cast(q, tf.float32), 0.0, 1.0)
#             idx = tf.cast(tf.math.round(q_clamped * tf.cast(T - 1, tf.float32)), tf.int32)
#             idx = tf.clip_by_value(idx, 0, T - 1)
#             batch_indices = tf.range(tf.shape(sorted_vals)[0])
#             gather_idx = tf.stack([batch_indices, tf.fill(tf.shape(batch_indices), idx)], axis=1)
#             thr_vals = tf.gather_nd(sorted_vals, gather_idx)
#             thr_vals = tf.reshape(thr_vals, (-1, 1, 1))
#             if weight_mode == 1:
#                 peak_strength = tf.cast(abs_y >= thr_vals, tf.float32)
#             else:
#                 t = tf.constant(float(tau), dtype=tf.float32)
#                 peak_strength = tf.sigmoid((abs_y - thr_vals) / (t + 1e-6))
#         else:
#             raise ValueError("Invalid peak_method. Use 'relative' or 'quantile'.")
# 
#         w = 1.0 + alpha * peak_strength
#         w = w * mask
# 
#         w_sum = tf.reduce_sum(w, axis=1, keepdims=True) + 1e-6
#         m_sum = tf.reduce_sum(mask, axis=1, keepdims=True) + 1e-6
#         w = w * (m_sum / w_sum)
# 
#         sq = tf.square(y_true - y_pred) * mask
#         loss_per_sample = tf.reduce_sum(w * sq, axis=1) / m_sum
#         return tf.reduce_mean(loss_per_sample)
# 
#     return weighted_mse_loss
# 
# def build_model(input_dim, PAD_value, loss_fn):
#     inp = Input(shape=(None, input_dim), name="dense_input")
#     x = Masking(mask_value=PAD_value, name="masking")(inp)
#     x = LSTM(100, return_sequences=True, name="lstm1")(x)
#     x = Activation('relu', name="relu1")(x)
#     x = LSTM(100, return_sequences=True, name="lstm2")(x)
#     x = Activation('relu', name="relu2")(x)
#     x = Dense(100, name="dense1")(x)
#     out = Dense(1, name="dense_out")(x)
# 
#     model = Model(inputs=inp, outputs=out, name="lstm_masked_dense")
#     model.compile(loss=loss_fn, optimizer=Adam(learning_rate=0.001), metrics=['mse'])
#     return model
# 
# class PeriodicSaver(tf.keras.callbacks.Callback):
#     def __init__(self, save_dir, backup_dir=None, period=10, keep_last=1):
#         super().__init__()
#         self.save_dir = save_dir
#         self.backup_dir = backup_dir
#         self.period = int(period)
#         self.keep_last = max(int(keep_last), 1)
#         os.makedirs(self.save_dir, exist_ok=True)
# 
#     def _prune_checkpoints(self):
#         files = glob.glob(os.path.join(self.save_dir, "ckpt_epoch_*.keras"))
#         if len(files) <= self.keep_last:
#             return
# 
#         def parse_epoch(p):
#             m = re.search(r"ckpt_epoch_(\d+)\.keras$", os.path.basename(p))
#             return int(m.group(1)) if m else 0
# 
#         files.sort(key=parse_epoch)
#         for f in files[:-self.keep_last]:
#             try:
#                 os.remove(f)
#                 print(f"🧹 old checkpoint removed: {f}")
#             except Exception as e:
#                 print(f"⚠️ could not remove {f}: {e}")
# 
#     def _prune_backup(self):
#         if not self.backup_dir or not os.path.exists(self.backup_dir):
#             return
#         entries = [os.path.join(self.backup_dir, n) for n in os.listdir(self.backup_dir)]
#         if len(entries) <= self.keep_last:
#             return
#         entries.sort(key=lambda p: os.path.getmtime(p))
#         for p in entries[:-self.keep_last]:
#             try:
#                 if os.path.isdir(p):
#                     shutil.rmtree(p, ignore_errors=True)
#                 else:
#                     os.remove(p)
#                 print(f"🧹 old backup removed: {p}")
#             except Exception as e:
#                 print(f"⚠️ could not remove backup {p}: {e}")
# 
#     def on_epoch_end(self, epoch, logs=None):
#         ep = epoch + 1
#         if ep % self.period == 0:
#             path = os.path.join(self.save_dir, f"ckpt_epoch_{ep:04d}.keras")
#             self.model.save(path)
#             print(f"💾 Periodic checkpoint saved: {path}")
#             self._prune_checkpoints()
#             self._prune_backup()
# 
# def find_latest_periodic_checkpoint(directory):
#     files = glob.glob(os.path.join(directory, "ckpt_epoch_*.keras"))
#     if not files:
#         return None, 0
# 
#     def parse_epoch(p):
#         m = re.search(r"ckpt_epoch_(\d+)\.keras$", os.path.basename(p))
#         return int(m.group(1)) if m else 0
# 
#     files.sort(key=parse_epoch)
#     latest = files[-1]
#     return latest, parse_epoch(latest)
# 
# def is_scenario_finished(model_dir, expected_epochs):
#     progress_path = os.path.join(model_dir, "progress.npy")
#     if not os.path.exists(progress_path):
#         return False
#     try:
#         data = np.load(progress_path, allow_pickle=True).item()
#         trained = int(data.get("epochs_trained", data.get("epochs", 0)))
#         return trained >= expected_epochs
#     except Exception as e:
#         print(f"⚠️ could not read progress for {model_dir}: {e}")
#         return False
# 
# # ==============================================================
# # Oversampling helpers (train only)
# # ==============================================================
# def has_peak_np(y, peak_method, thresh, q):
#     abs_y = np.abs(y.reshape(-1))
#     if abs_y.size == 0:
#         return False
#     if peak_method == "relative":
#         m = abs_y.max()
#         if m <= 1e-12:
#             return False
#         rel = abs_y / m
#         return np.any(rel >= thresh)
#     else:
#         thr = np.quantile(abs_y, q)
#         return np.any(abs_y >= thr)
# 
# def oversample_lists(X_list, Y_list, peak_method, thresh, q, factor):
#     if factor <= 1:
#         return X_list, Y_list
#     X_out, Y_out = [], []
#     for x, y in zip(X_list, Y_list):
#         X_out.append(x); Y_out.append(y)
#         if has_peak_np(y, peak_method, thresh, q):
#             for _ in range(factor - 1):
#                 X_out.append(x)
#                 Y_out.append(y)
#     return X_out, Y_out
# 
# # ==============================================================
# # Dataset builders (list -> tf.data)
# # ==============================================================
# def gen_from_lists(x_list, y_list):
#     for x, y in zip(x_list, y_list):
#         yield x, y
# 
# def make_datasets(X_train_list, Y_train_list, X_val_list, Y_val_list, input_dim):
#     output_signature = (
#         tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32),
#         tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
#     )
# 
#     ds_train = tf.data.Dataset.from_generator(
#         lambda: gen_from_lists(X_train_list, Y_train_list),
#         output_signature=output_signature
#     ).padded_batch(
#         BATCH_SIZE,
#         padded_shapes=([None, input_dim], [None, 1]),
#         padding_values=(PAD, PAD),
#         drop_remainder=False
#     ).prefetch(tf.data.AUTOTUNE)
# 
#     ds_val = tf.data.Dataset.from_generator(
#         lambda: gen_from_lists(X_val_list, Y_val_list),
#         output_signature=output_signature
#     ).padded_batch(
#         BATCH_SIZE,
#         padded_shapes=([None, input_dim], [None, 1]),
#         padding_values=(PAD, PAD),
#         drop_remainder=False
#     ).prefetch(tf.data.AUTOTUNE)
# 
#     return ds_train, ds_val
# 
# # ==============================================================
# # 🚀 MODE 1: Global multi-height (+ height feature)
# # ==============================================================
# if USE_MULTI_HEIGHT:
#     INPUT_DIM = 2  # [GM, H]
#     X_all, Y_all = [], []
# 
#     for h_tag in height_tags:
#         print("\n" + "#" * 80)
#         print(f"📥 بارگذاری داده برای ارتفاع: {h_tag}")
#         print("#" * 80)
# 
#         x_data_path, y_data_path = get_data_paths_for_height(h_tag)
# 
#         if not os.path.exists(x_data_path):
#             print(f"⚠️ X برای {h_tag} پیدا نشد: {x_data_path} → رد می‌شود.")
#             continue
#         if not os.path.exists(y_data_path):
#             print(f"⚠️ Y برای {h_tag} پیدا نشد: {y_data_path} → رد می‌شود.")
#             continue
# 
#         X_dict = np.load(x_data_path, allow_pickle=True).item()
#         Y_dict = np.load(y_data_path, allow_pickle=True).item()
# 
#         common_keys = sorted(set(X_dict.keys()) & set(Y_dict.keys()))
#         if not common_keys:
#             print(f"❌ هیچ کلید مشترکی برای {h_tag} پیدا نشد. رد می‌شود.")
#             continue
# 
#         h_val = np.float32(height_values[h_tag])
# 
#         for k in common_keys:
#             x = X_dict[k].reshape(-1, 1).astype('float32')
#             y = Y_dict[k].reshape(-1, 1).astype('float32')
# 
#             T = x.shape[0]
#             h_col = np.full((T, 1), h_val, dtype='float32')
#             x_feat = np.concatenate([x, h_col], axis=1)
# 
#             X_all.append(x_feat)
#             Y_all.append(y)
# 
#     num_samples = len(X_all)
#     if num_samples == 0:
#         raise ValueError("❌ هیچ نمونه‌ای برای آموزش پیدا نشد (بعد از فیلتر ارتفاع‌ها/کلاسترها).")
# 
#     print(f"\n📦 تعداد کل نمونه‌ها (همه ارتفاع‌ها با هم): {num_samples}")
# 
#     scaler_X = MinMaxScaler(feature_range=(-1, 1))
#     scaler_Y = MinMaxScaler(feature_range=(-1, 1))
# 
#     X_concat = np.concatenate(X_all, axis=0)
#     Y_concat = np.concatenate(Y_all, axis=0)
# 
#     scaler_X.fit(X_concat)
#     scaler_Y.fit(Y_concat)
# 
#     X_all = [scaler_X.transform(x) for x in X_all]
#     Y_all = [scaler_Y.transform(y) for y in Y_all]
# 
#     # short meta dir
#     meta_dir = os.path.join(base_model_root, f"_META_{MODE_CODE}-{TRAIN_CODE}")
#     os.makedirs(meta_dir, exist_ok=True)
# 
#     if is_linear:
#         scaler_X_path = os.path.join(meta_dir, "scaler_X_linear.pkl")
#         scaler_Y_path = os.path.join(meta_dir, "scaler_Y_linear.pkl")
#     else:
#         scaler_X_path = os.path.join(meta_dir, "scaler_X_nonlinear.pkl")
#         scaler_Y_path = os.path.join(meta_dir, "scaler_Y_nonlinear.pkl")
# 
#     joblib.dump(scaler_X, scaler_X_path)
#     joblib.dump(scaler_Y, scaler_Y_path)
#     print("\n💾 Global scalers saved:")
#     print("  →", scaler_X_path)
#     print("  →", scaler_Y_path)
# 
#     split_idx_path = os.path.join(meta_dir, f"split_idx_seed{SEED}.npy")
# 
#     if os.path.exists(split_idx_path):
#         idx = np.load(split_idx_path)
#         if len(idx) != num_samples:
#             idx = np.arange(num_samples)
#             rng = np.random.default_rng(SEED)
#             rng.shuffle(idx)
#             np.save(split_idx_path, idx)
#             print("⚠️ طول داده‌ها تغییر کرده بود؛ split جدید ساخته شد.")
#         else:
#             print("✅ Fixed global split indices loaded:", split_idx_path)
#     else:
#         idx = np.arange(num_samples)
#         rng = np.random.default_rng(SEED)
#         rng.shuffle(idx)
#         np.save(split_idx_path, idx)
#         print("✅ Fixed global split indices saved:", split_idx_path)
# 
#     train_split = int(0.50 * num_samples)
#     val_split   = int(0.63 * num_samples)
# 
#     X_train_base = [X_all[i] for i in idx[:train_split]]
#     Y_train_base = [Y_all[i] for i in idx[:train_split]]
# 
#     X_val_list   = [X_all[i] for i in idx[train_split:val_split]]
#     Y_val_list   = [Y_all[i] for i in idx[train_split:val_split]]
# 
#     print(f"\n📦 Global split | Train: {len(X_train_base)}  Val: {len(X_val_list)}  Test: {num_samples - val_split}")
# 
#     heights_str = ", ".join(height_tags)
# 
#     for scen in SCENARIOS:
#         EPOCHS = int(scen["EPOCHS"])
#         ALPHA  = float(scen["ALPHA"])
#         THRESH = float(scen["THRESH"])
#         WEIGHT_MODE = int(scen.get("WEIGHT_MODE", 2))
#         TAU = float(scen.get("TAU", 0.05))
# 
#         X_train_list, Y_train_list = oversample_lists(
#             X_train_base, Y_train_base,
#             PEAK_METHOD, THRESH, PEAK_Q, OVERSAMPLE_FACTOR
#         )
# 
#         scen_short = make_scen_short(EPOCHS, ALPHA, PEAK_METHOD, PEAK_Q, THRESH, OVERSAMPLE_FACTOR, WEIGHT_MODE, TAU)
#         run_tag = f"{MODE_CODE}-{TRAIN_CODE}__{scen_short}"
#         model_dir = os.path.join(base_model_root, run_tag)
#         os.makedirs(model_dir, exist_ok=True)
# 
#         model_path    = os.path.join(model_dir, "LSTM.keras")
#         backup_dir    = os.path.join(model_dir, "backup")
#         ckpt_dir      = os.path.join(model_dir, "checkpoints")
#         progress_path = os.path.join(model_dir, "progress.npy")
#         os.makedirs(backup_dir, exist_ok=True)
#         os.makedirs(ckpt_dir, exist_ok=True)
# 
#         if is_scenario_finished(model_dir, EPOCHS):
#             print("\n" + "=" * 80)
#             print(f"⏩ سناریو {run_tag} قبلاً کامل شده است. رد می‌شود.")
#             print("=" * 80)
#             continue
# 
#         print("\n" + "=" * 80)
#         print(f"🚀 شروع سناریو (Global + Height): {run_tag}")
#         print(f"   → EPOCHS={EPOCHS}, ALPHA={ALPHA}, THRESH={THRESH}, WEIGHT_MODE={WEIGHT_MODE}, TAU={TAU}")
#         print(f"   → PeakMethod={PEAK_METHOD}, Q={PEAK_Q}, Oversample={OVERSAMPLE_FACTOR}")
#         print(f"   → Heights: {heights_str}")
#         print("=" * 80)
# 
#         ds_train, ds_val = make_datasets(X_train_list, Y_train_list, X_val_list, Y_val_list, INPUT_DIM)
# 
#         tf.random.set_seed(SEED)
#         np.random.seed(SEED)
#         random.seed(SEED)
# 
#         loss_fn = make_weighted_mse_loss(
#             PAD, ALPHA, THRESH,
#             weight_mode=WEIGHT_MODE, tau=TAU,
#             peak_method=PEAK_METHOD, q=PEAK_Q
#         )
#         model = build_model(INPUT_DIM, PAD, loss_fn)
# 
#         checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
#             model_path, monitor='val_loss', save_best_only=True, verbose=1
#         )
#         reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
#             monitor='val_loss', factor=0.5, patience=10, verbose=1, min_lr=1e-6
#         )
#         backup_cb = tf.keras.callbacks.BackupAndRestore(backup_dir=backup_dir)
#         periodic_ckpt = PeriodicSaver(
#             save_dir=ckpt_dir, backup_dir=backup_dir,
#             period=10, keep_last=1
#         )
# 
#         initial_epoch = 0
#         if not os.path.exists(backup_dir) or not os.listdir(backup_dir):
#             ckpt_path, ep = find_latest_periodic_checkpoint(ckpt_dir)
#             if ckpt_path:
#                 try:
#                     tmp_model = load_model(ckpt_path, compile=False)
#                     tmp_model.compile(loss=loss_fn, optimizer=Adam(learning_rate=0.001), metrics=['mse'])
#                     model = tmp_model
#                     initial_epoch = ep
#                     print(f"🔁 Resuming from periodic checkpoint: {ckpt_path} (initial_epoch={initial_epoch})")
#                 except Exception as e:
#                     print(f"⚠️ Failed to load periodic checkpoint: {e}")
# 
#         history = model.fit(
#             ds_train,
#             validation_data=ds_val,
#             epochs=EPOCHS,
#             initial_epoch=initial_epoch,
#             callbacks=[backup_cb, checkpoint_best, periodic_ckpt, reduce_lr],
#             verbose=1
#         )
# 
#         epochs_trained = initial_epoch + len(history.history.get('loss', []))
# 
#         progress_data = {
#             'train_loss': history.history.get('loss', []),
#             'val_loss': history.history.get('val_loss', []),
#             'best_val_loss': float(min(history.history.get('val_loss', [np.inf]))),
#             'epochs': EPOCHS,
#             'epochs_trained': int(epochs_trained),
#             'heights_used': height_tags,
#             'use_clustered': USE_CLUSTERED,
#             'cluster_label': cluster_label,
#             'mode_code': MODE_CODE,
#             'train_code': TRAIN_CODE,
#             'peak_method': PEAK_METHOD,
#             'peak_q': float(PEAK_Q),
#             'oversample_factor': int(OVERSAMPLE_FACTOR),
#             'alpha': float(ALPHA),
#             'thresh': float(THRESH),
#             'weight_mode': int(WEIGHT_MODE),
#             'tau': float(TAU),
#             'seed': SEED,
#             'script_name': script_name,
#             'script_tag': script_tag,
#             'gm_root_dir': gm_root_dir,
#             'tha_root_dir': tha_root_dir,
#             'scaler_X_path': scaler_X_path,
#             'scaler_Y_path': scaler_Y_path,
#             'split_idx_path': split_idx_path,
#         }
#         np.save(progress_path, progress_data)
#         print("💾 Progress saved:", progress_path)
# 
#         runinfo = dict(progress_data)
#         runinfo['model_dir'] = model_dir
#         runinfo['model_path'] = model_path
#         runinfo['backup_dir'] = backup_dir
#         runinfo['ckpt_dir'] = ckpt_dir
#         runinfo['progress_path'] = progress_path
#         write_runinfo(model_dir, runinfo)
# 
#         try:
#             plt.figure()
#             plt.plot(progress_data['train_loss'], label='Train Loss')
#             plt.plot(progress_data['val_loss'], label='Val Loss')
#             plt.xlabel('Epoch')
#             plt.ylabel('Loss')
#             plt.title(f'All Heights [{heights_str}] - Loss - {run_tag}')
#             plt.legend()
#             plt.tight_layout()
# 
#             loss_plot_path = os.path.join(model_dir, f"loss_curve_{run_tag}.png")
#             plt.savefig(loss_plot_path, dpi=300)
#             plt.close()
#             print("📊 Loss curve saved to:", loss_plot_path)
#         except Exception as e:
#             print(f"⚠️ Plot error (loss curve): {e}")
# 
#         print(f"✅ Scenario {run_tag} finished.\n")
# 
#     print("🎉 همه سناریوها در مود Multi-height تمام شدند.")
# 
# # ==============================================================
# # 🚀 MODE 2: Per-height (no height feature)
# # ==============================================================
# else:
#     INPUT_DIM = 1
# 
#     for h_tag in height_tags:
#         print("\n" + "#" * 80)
#         print(f"🏗️ شروع آموزش جداگانه برای ارتفاع: {h_tag}")
#         print("#" * 80)
# 
#         x_data_path, y_data_path = get_data_paths_for_height(h_tag)
# 
#         if not os.path.exists(x_data_path):
#             print(f"⚠️ X_data برای {h_tag} پیدا نشد: {x_data_path} → رد می‌شود.")
#             continue
#         if not os.path.exists(y_data_path):
#             print(f"⚠️ Y_data برای {h_tag} پیدا نشد: {y_data_path} → رد می‌شود.")
#             continue
# 
#         X_dict = np.load(x_data_path, allow_pickle=True).item()
#         Y_dict = np.load(y_data_path, allow_pickle=True).item()
# 
#         common_keys = sorted(set(X_dict.keys()) & set(Y_dict.keys()))
#         if not common_keys:
#             print(f"❌ هیچ کلید مشترکی بین X و Y برای {h_tag} پیدا نشد. رد می‌شود.")
#             continue
# 
#         X_all, Y_all = [], []
#         for k in common_keys:
#             x = X_dict[k].reshape(-1, 1).astype('float32')
#             y = Y_dict[k].reshape(-1, 1).astype('float32')
#             X_all.append(x)
#             Y_all.append(y)
# 
#         num_samples = len(X_all)
#         if num_samples == 0:
#             print(f"⚠️ هیچ نمونه‌ای برای {h_tag} وجود ندارد. رد می‌شود.")
#             continue
# 
#         print(f"📦 تعداد نمونه‌ها برای {h_tag}: {num_samples}")
# 
#         scaler_X = MinMaxScaler(feature_range=(-1, 1))
#         scaler_Y = MinMaxScaler(feature_range=(-1, 1))
# 
#         X_concat = np.concatenate(X_all, axis=0)
#         Y_concat = np.concatenate(Y_all, axis=0)
# 
#         scaler_X.fit(X_concat)
#         scaler_Y.fit(Y_concat)
# 
#         X_all = [scaler_X.transform(x) for x in X_all]
#         Y_all = [scaler_Y.transform(y) for y in Y_all]
# 
#         height_meta_dir = os.path.join(base_model_root, f"_META_{MODE_CODE}-{TRAIN_CODE}_{h_tag}")
#         os.makedirs(height_meta_dir, exist_ok=True)
# 
#         if is_linear:
#             scaler_X_path = os.path.join(height_meta_dir, "scaler_X_linear.pkl")
#             scaler_Y_path = os.path.join(height_meta_dir, "scaler_Y_linear.pkl")
#         else:
#             scaler_X_path = os.path.join(height_meta_dir, "scaler_X_nonlinear.pkl")
#             scaler_Y_path = os.path.join(height_meta_dir, "scaler_Y_nonlinear.pkl")
# 
#         joblib.dump(scaler_X, scaler_X_path)
#         joblib.dump(scaler_Y, scaler_Y_path)
#         print("💾 height-scalers saved:")
#         print("  →", scaler_X_path)
#         print("  →", scaler_Y_path)
# 
#         split_idx_path = os.path.join(height_meta_dir, f"split_idx_seed{SEED}.npy")
# 
#         if os.path.exists(split_idx_path):
#             idx = np.load(split_idx_path)
#             if len(idx) != num_samples:
#                 idx = np.arange(num_samples)
#                 rng = np.random.default_rng(SEED)
#                 rng.shuffle(idx)
#                 np.save(split_idx_path, idx)
#                 print("⚠️ طول داده‌ها تغییر کرده بود؛ split جدید ساخته شد.")
#             else:
#                 print("✅ Fixed split indices loaded:", split_idx_path)
#         else:
#             idx = np.arange(num_samples)
#             rng = np.random.default_rng(SEED)
#             rng.shuffle(idx)
#             np.save(split_idx_path, idx)
#             print("✅ Fixed split indices saved:", split_idx_path)
# 
#         train_split = int(0.50 * num_samples)
#         val_split   = int(0.63 * num_samples)
# 
#         X_train_base = [X_all[i] for i in idx[:train_split]]
#         Y_train_base = [Y_all[i] for i in idx[:train_split]]
# 
#         X_val_list   = [X_all[i] for i in idx[train_split:val_split]]
#         Y_val_list   = [Y_all[i] for i in idx[train_split:val_split]]
# 
#         for scen in SCENARIOS:
#             EPOCHS = int(scen["EPOCHS"])
#             ALPHA  = float(scen["ALPHA"])
#             THRESH = float(scen["THRESH"])
#             WEIGHT_MODE = int(scen.get("WEIGHT_MODE", 2))
#             TAU = float(scen.get("TAU", 0.05))
# 
#             X_train_list, Y_train_list = oversample_lists(
#                 X_train_base, Y_train_base,
#                 PEAK_METHOD, THRESH, PEAK_Q, OVERSAMPLE_FACTOR
#             )
# 
#             scen_short = make_scen_short(EPOCHS, ALPHA, PEAK_METHOD, PEAK_Q, THRESH, OVERSAMPLE_FACTOR, WEIGHT_MODE, TAU)
#             run_tag = f"{MODE_CODE}-{TRAIN_CODE}_{h_tag}__{scen_short}"
#             model_dir = os.path.join(base_model_root, run_tag)
#             os.makedirs(model_dir, exist_ok=True)
# 
#             model_path    = os.path.join(model_dir, "LSTM.keras")
#             backup_dir    = os.path.join(model_dir, "backup")
#             ckpt_dir      = os.path.join(model_dir, "checkpoints")
#             progress_path = os.path.join(model_dir, "progress.npy")
# 
#             if is_scenario_finished(model_dir, EPOCHS):
#                 print("\n" + "=" * 80)
#                 print(f"⏩ {h_tag} | سناریو {run_tag} قبلاً کامل شده است. رد می‌شود.")
#                 print("=" * 80)
#                 continue
# 
#             print("\n" + "=" * 80)
#             print(f"🚀 {h_tag} | شروع سناریو: {run_tag}")
#             print(f"   → EPOCHS={EPOCHS}, ALPHA={ALPHA}, THRESH={THRESH}, WEIGHT_MODE={WEIGHT_MODE}, TAU={TAU}")
#             print(f"   → PeakMethod={PEAK_METHOD}, Q={PEAK_Q}, Oversample={OVERSAMPLE_FACTOR}")
#             print("=" * 80)
# 
#             os.makedirs(ckpt_dir, exist_ok=True)
#             os.makedirs(backup_dir, exist_ok=True)
# 
#             ds_train, ds_val = make_datasets(X_train_list, Y_train_list, X_val_list, Y_val_list, INPUT_DIM)
# 
#             tf.random.set_seed(SEED)
#             np.random.seed(SEED)
#             random.seed(SEED)
# 
#             loss_fn = make_weighted_mse_loss(
#                 PAD, ALPHA, THRESH,
#                 weight_mode=WEIGHT_MODE, tau=TAU,
#                 peak_method=PEAK_METHOD, q=PEAK_Q
#             )
#             model = build_model(INPUT_DIM, PAD, loss_fn)
# 
#             checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
#                 model_path, monitor='val_loss', save_best_only=True, verbose=1
#             )
#             reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
#                 monitor='val_loss', factor=0.5, patience=10, verbose=1, min_lr=1e-6
#             )
#             backup_cb = tf.keras.callbacks.BackupAndRestore(backup_dir=backup_dir)
#             periodic_ckpt = PeriodicSaver(
#                 save_dir=ckpt_dir, backup_dir=backup_dir,
#                 period=10, keep_last=1
#             )
# 
#             initial_epoch = 0
#             if not os.path.exists(backup_dir) or not os.listdir(backup_dir):
#                 ckpt_path, ep = find_latest_periodic_checkpoint(ckpt_dir)
#                 if ckpt_path:
#                     try:
#                         tmp_model = load_model(ckpt_path, compile=False)
#                         tmp_model.compile(loss=loss_fn, optimizer=Adam(learning_rate=0.001), metrics=['mse'])
#                         model = tmp_model
#                         initial_epoch = ep
#                         print(f"🔁 {h_tag} | Resuming from checkpoint: {ckpt_path} (initial_epoch={initial_epoch})")
#                     except Exception as e:
#                         print(f"⚠️ Failed to load periodic checkpoint: {e}")
# 
#             history = model.fit(
#                 ds_train,
#                 validation_data=ds_val,
#                 epochs=EPOCHS,
#                 initial_epoch=initial_epoch,
#                 callbacks=[backup_cb, checkpoint_best, periodic_ckpt, reduce_lr],
#                 verbose=1
#             )
# 
#             epochs_trained = initial_epoch + len(history.history.get('loss', []))
# 
#             progress_data = {
#                 'train_loss': history.history.get('loss', []),
#                 'val_loss': history.history.get('val_loss', []),
#                 'best_val_loss': float(min(history.history.get('val_loss', [np.inf]))),
#                 'epochs': EPOCHS,
#                 'epochs_trained': int(epochs_trained),
#                 'height': h_tag,
#                 'use_clustered': USE_CLUSTERED,
#                 'cluster_label': cluster_label,
#                 'mode_code': MODE_CODE,
#                 'train_code': TRAIN_CODE,
#                 'peak_method': PEAK_METHOD,
#                 'peak_q': float(PEAK_Q),
#                 'oversample_factor': int(OVERSAMPLE_FACTOR),
#                 'alpha': float(ALPHA),
#                 'thresh': float(THRESH),
#                 'weight_mode': int(WEIGHT_MODE),
#                 'tau': float(TAU),
#                 'seed': SEED,
#                 'script_name': script_name,
#                 'script_tag': script_tag,
#                 'gm_root_dir': gm_root_dir,
#                 'tha_root_dir': tha_root_dir,
#                 'scaler_X_path': scaler_X_path,
#                 'scaler_Y_path': scaler_Y_path,
#                 'split_idx_path': split_idx_path,
#             }
#             np.save(progress_path, progress_data)
#             print("💾 Progress saved:", progress_path)
# 
#             runinfo = dict(progress_data)
#             runinfo['model_dir'] = model_dir
#             runinfo['model_path'] = model_path
#             runinfo['backup_dir'] = backup_dir
#             runinfo['ckpt_dir'] = ckpt_dir
#             runinfo['progress_path'] = progress_path
#             write_runinfo(model_dir, runinfo)
# 
#             try:
#                 plt.figure()
#                 plt.plot(progress_data['train_loss'], label='Train Loss')
#                 plt.plot(progress_data['val_loss'], label='Val Loss')
#                 plt.xlabel('Epoch')
#                 plt.ylabel('Loss')
#                 plt.title(f'{h_tag} - Loss - {run_tag}')
#                 plt.legend()
#                 plt.tight_layout()
# 
#                 loss_plot_path = os.path.join(model_dir, f"loss_curve_{run_tag}.png")
#                 plt.savefig(loss_plot_path, dpi=300)
#                 plt.close()
#                 print("📊 Loss curve saved to:", loss_plot_path)
#             except Exception as e:
#                 print(f"⚠️ Plot error (loss curve): {e}")
# 
#             print(f"✅ {h_tag} | Scenario {run_tag} finished.\n")
# 
#     print("🎉 آموزش همه ارتفاع‌ها در مود per-height تمام شد.")
# 
# 
# =============================================================================


# =============================================================================
# 
# # -*- coding: utf-8 -*-
# """
# File name      : LSTM_For_ghab_GPU-Clusttering-v03.py
# Author         : pc22
# Created on     : Sun Dec 28 08:47:13 2025
# Last modified  : Wed Dec 31 09:xx:xx 2025
# ------------------------------------------------------------
# Purpose:g set
#       (3) Safe resume/recovery via BackupAndRestore + periodic checkpoints
#       (4) Script-tagged, compact output folders to avoid overwriting
#           and improve experiment traceability ac
#     Train an LSTM-based sequence-to-sequence model to predict
#     structural time-history responses (THA) from ground-motion (GM)
#     time series, with explicit emphasis on improving prediction
#     quality at extreme response regions (peaks).
# 
#     This version introduces a robust peak-aware training strategy:
#       (1) Peak definition per-sample using either:
#             - Relative thresholding (thresh × max of each sequence), or
#             - Quantile-based Top-K% thresholding (per-sample quantile q)
#       (2) Oversampling of peak-containing sequences to mitigate
#           peak rarity in the traininross code versions
#       (5) runinfo.txt lifecycle improved:
#             - runinfo.txt is created at scenario start (status=STARTED)
#             - runinfo.txt is updated at scenario end (status=FINISHED)
# 
# ------------------------------------------------------------
# Description:
#     The script loads preprocessed GM/THA time-series pairs generated
#     by previous pipeline steps, scales inputs/outputs, and trains an
#     LSTM model under user-selected modes:
# 
#       - Linear vs Nonlinear training:
#           * Linear    → uses THA_linear dataset folders
#           * Nonlinear → uses THA_nonlinear dataset folders
# 
#       - Clustered vs Non-clustered data:
#           * Clustered     → uses cluster_balanced_global subfolders
#           * Non-clustered → uses original X_data_H* / Y_data_H* files
# 
#       - Two training modes:
#           (1) Global multi-height training (USE_MULTI_HEIGHT=1):
#               A shared model is trained on all selected heights
#               with height added as an additional input feature:
#                   X = [GM(t), H]
#           (2) Per-height independent training (USE_MULTI_HEIGHT=0):
#               Separate models are trained per height:
#                   X = [GM(t)]
# 
#     Variable-length time series are handled using padding and a
#     masking layer (PAD = -999.0). Training datasets are built with
#     tf.data and padded_batch, enabling mini-batch training on
#     sequences of varying length.
# 
#     Peak-aware learning:
#       - Weighted MSE loss is used, where sample/time-step weights
#         are increased at peak regions using:
#           w = 1 + alpha * peak_strength
#         and peak_strength is computed either by:
#           * Binary thresholding (WEIGHT_MODE=1), or
#           * Soft sigmoid weighting (WEIGHT_MODE=2) with temperature TAU
# 
#       - Oversampling is optionally applied to the training set only:
#         sequences that contain peaks are replicated OVERSAMPLE_FACTOR
#         times to increase peak frequency during optimization.
# 
#     Experiment management:
#       - Outputs are organized under a short, version-tagged script folder
#         (e.g., V03) to avoid overwriting between code variants.
#       - Each scenario run creates an isolated run folder and logs:
#           * best model (LSTM.keras)
#           * periodic checkpoints
#           * backup state for auto-resume
#           * progress.npy (training curves + metadata)
#           * runinfo.txt (human-readable run configuration; created early and updated)
#           * loss curve image
# 
# ------------------------------------------------------------
# Inputs:
#     Data (from previous pipeline steps):
#       - GM time series:
#           Output/3_GM_Fixed_train_linear/H*/...
#           Output/3_GM_Fixed_train_nonlinear/H*/...
#       - THA time series:
#           Output/3_THA_Fixed_train_linear/H*/...
#           Output/3_THA_Fixed_train_nonlinear/H*/...
# 
#     Runtime user inputs (interactive):
#       - Linear vs Nonlinear mode selection
#       - Clustered vs Non-clustered dataset selection
#       - Heights to include in training (H2, H3, ...)
#       - Training mode: global multi-height vs per-height models
#       - Peak definition method: relative vs quantile
#       - Quantile q (if quantile mode)
#       - Oversampling factor (integer ≥ 1)
# 
#     Scenario hyperparameters (SCENARIOS list):
#       - EPOCHS
#       - ALPHA               (peak emphasis strength)
#       - THRESH              (relative threshold; used if PEAK_METHOD="relative")
#       - WEIGHT_MODE         (1 = binary, 2 = soft sigmoid)
#       - TAU                 (sigmoid temperature; used if WEIGHT_MODE=2)
# 
#     Peak strategy parameters:
#       - PEAK_METHOD         ("relative" or "quantile")
#       - PEAK_Q              (quantile q; used if PEAK_METHOD="quantile")
#       - OVERSAMPLE_FACTOR   (integer ≥ 1; 1 disables oversampling)
# 
# ------------------------------------------------------------
# Outputs:
#     All outputs are saved under:
# 
#       Output/
#         ├── Progress_of_LSTM_linear/
#         │     └── <script_tag>/                  (e.g., V03)
#         │           ├── _META_<MODE>-<TRAIN>/    (global scalers + split indices)
#         │           ├── <run_tag>/               (one folder per scenario)
#         │           │     ├── LSTM.keras
#         │           │     ├── checkpoints/
#         │           │     ├── backup/
#         │           │     ├── progress.npy
#         │           │     ├── runinfo.txt
#         │           │     └── loss_curve_*.png
#         │           └── ...
#         └── Progress_of_LSTM_nonlinear/
#               └── <script_tag>/
#                     ...
# 
#     where:
#       - <script_tag> is inferred from the executing file name:
#             * vXX is extracted if present (e.g., "v03" → "V03")
#             * otherwise a short alphanumeric prefix is used
#       - <run_tag> is a compact code describing mode + scenario, e.g.:
#             NC-GH__E76-A1-Q70-OS3-M1-T05
#         (cluster/noCluster + training mode + scenario params)
# 
# ------------------------------------------------------------
# Changes since baseline / earlier code:
#     1) Peak-aware loss redesigned to support:
#          - Relative peaks (thresh × max per-sample)
#          - Quantile peaks (Top-K% per-sample)
#          - Binary vs soft weighting (sigmoid) controlled by WEIGHT_MODE/TAU
#     2) Oversampling of peak-containing sequences added (train set only).
#     3) Safe resume improved:
#          - BackupAndRestore enabled
#          - Periodic checkpoints saved every N epochs
#          - Optional resume from last periodic checkpoint if backup is absent
#     4) Output management updated:
#          - Version-tagged (<script_tag>) parent folder prevents overwriting
#          - Compact run tags improve readability and path length control
#          - runinfo.txt is written alongside progress.npy for traceability
#     5) runinfo lifecycle updated:
#          - runinfo.txt is written at scenario start (status=STARTED)
#          - runinfo.txt is updated at scenario end (status=FINISHED)
# 
# ------------------------------------------------------------
# Impact of changes:
#     - Improves prediction accuracy at extreme response regions by
#       increasing gradient emphasis on peaks and reducing peak rarity.
#     - Enhances experiment reproducibility via fixed random seeds,
#       deterministic ops (where supported), and fixed split indices.
#     - Improves run traceability and reduces accidental overwriting by
#       isolating outputs per code version and scenario.
# 
# ------------------------------------------------------------
# Status:
#     Stable (training + resume supported)
# 
# ------------------------------------------------------------
# Notes:
#     - Quantile-based peak thresholding is always computed per individual
#       time series (never global).
#     - Oversampling is applied only to the training split; validation/test
#       distributions remain unchanged.
#     - Setting PEAK_METHOD="relative" and OVERSAMPLE_FACTOR=1 approximates
#       the original baseline behavior.
#     - On Windows, path length limitations can affect BackupAndRestore if
#       output roots are deeply nested; prefer short base output roots or
#       dedicated short backup roots if needed.
# """
# 
# 
# import sys, io
# if hasattr(sys.stdout, "buffer"):
#     sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
# 
# import os, gc, glob, re, shutil, random
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.layers import Input, Masking, LSTM, Activation, Dense
# from tensorflow.keras.optimizers import Adam
# from sklearn.preprocessing import MinMaxScaler
# import joblib
# 
# plt.ioff()
# 
# # ==============================================================
# # 🔒 Reproducibility / Determinism
# # ==============================================================
# SEED = 1234
# os.environ["PYTHONHASHSEED"] = str(SEED)
# os.environ["TF_DETERMINISTIC_OPS"] = "1"
# 
# random.seed(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)
# try:
#     tf.keras.utils.set_random_seed(SEED)
# except Exception:
#     pass
# try:
#     tf.config.experimental.enable_op_determinism(True)
# except Exception:
#     pass
# 
# # ==============================================================
# # 🚀 GPU dynamic memory
# # ==============================================================
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         print("✅ GPU memory set dynamically.")
#     except RuntimeError as e:
#         print(e)
# 
# # ==============================================================
# # 📁 Paths + Linear / Nonlinear selection
# # ==============================================================
# base_dir = os.path.dirname(os.path.abspath(__file__))
# root_dir = os.path.abspath(os.path.join(base_dir, os.pardir))
# 
# output_root_dir = os.path.join(root_dir, "Output")
# 
# choice = input(
#     "آموزش بر اساس پاسخ خطی باشد یا غیرخطی؟ "
#     "برای خطی عدد 1 و برای غیرخطی عدد 0 را وارد کن: "
# ).strip()
# is_linear = (choice == "1")
# 
# if is_linear:
#     print("📌 آموزش بر اساس داده‌های خطی (THA_linear) انجام می‌شود.")
#     gm_root_dir  = os.path.join(output_root_dir, "3_GM_Fixed_train_linear")
#     tha_root_dir = os.path.join(output_root_dir, "3_THA_Fixed_train_linear")
#     base_model_root = os.path.join(output_root_dir, "Progress_of_LSTM_linear")
# else:
#     print("📌 آموزش بر اساس داده‌های غیرخطی (THA_nonlinear) انجام می‌شود.")
#     gm_root_dir  = os.path.join(output_root_dir, "3_GM_Fixed_train_nonlinear")
#     tha_root_dir = os.path.join(output_root_dir, "3_THA_Fixed_train_nonlinear")
#     base_model_root = os.path.join(output_root_dir, "Progress_of_LSTM_nonlinear")
# 
# os.makedirs(base_model_root, exist_ok=True)
# 
# print("📂 GM root dir :", gm_root_dir)
# print("📂 THA root dir:", tha_root_dir)
# print("📂 Base model root:", base_model_root)
# print()
# 
# # ==============================================================
# # 🧾 Script-tag isolation (avoid overwriting across versions) + SHORT
# # ==============================================================
# script_name = os.path.splitext(os.path.basename(__file__))[0]
# 
# # Short tag from script_name (prefer vXX if exists)
# m = re.search(r"(v\d+)", script_name, flags=re.IGNORECASE)
# if m:
#     script_tag = m.group(1).upper()
# else:
#     # fallback: keep it short but stable
#     script_tag = re.sub(r"[^A-Za-z0-9]+", "", script_name)[:12] or "SCRIPT"
# 
# base_model_root = os.path.join(base_model_root, script_tag)
# os.makedirs(base_model_root, exist_ok=True)
# 
# # ==============================================================
# # 🔁 Clustered vs non-clustered
# # ==============================================================
# print("-------------------------------------------")
# print("آیا می‌خواهی از داده‌های *کلاسترشده* استفاده کنم؟")
# print("   1 = بله، از فایل‌های cluster_balanced_global استفاده کن")
# print("   0 = خیر، از فایل‌های اصلی X_data_H* و Y_data_H* استفاده کن")
# print("-------------------------------------------\n")
# 
# cluster_choice = input("انتخابت را وارد کن (1 یا 0): ").strip()
# USE_CLUSTERED = (cluster_choice == "1")
# 
# cluster_label = ""
# if USE_CLUSTERED:
#     print("✅ استفاده از داده‌های کلاسترشده (cluster_balanced_global).")
#     cluster_label = input(
#         "برای اسم پوشهٔ مدل، تعداد کلاسترها (K) را بنویس "
#         "(مثلاً 4). اگر خالی بگذاری، 'clustered' نوشته می‌شود: "
#     ).strip()
# else:
#     print("✅ استفاده از داده‌های اصلی بدون کلاسترینگ.")
# 
# def compact_float_tag(x, scale=100):
#     """
#     0.70 -> '70' (default scale=100)
#     0.05 -> '05' (if scale=100)
#     1.0  -> '100'
#     """
#     try:
#         v = float(x)
#     except Exception:
#         return str(x)
#     n = int(round(v * scale))
#     width = 2 if scale == 100 else 0
#     return f"{n:0{width}d}" if width > 0 else str(n)
# 
# def compact_A_tag(a):
#     try:
#         v = float(a)
#     except Exception:
#         return str(a)
#     if abs(v - int(v)) < 1e-9:
#         return str(int(v))
#     s = str(v).replace(".", "p")
#     return s
# 
# def make_scen_short(EPOCHS, ALPHA, peak_method, PEAK_Q, THRESH, OVERSAMPLE_FACTOR, WEIGHT_MODE, TAU):
#     e = int(EPOCHS)
#     a = compact_A_tag(ALPHA)
#     if peak_method == "quantile":
#         q = compact_float_tag(PEAK_Q, scale=100)
#         peak = f"Q{q}"
#     else:
#         r = compact_float_tag(THRESH, scale=100)
#         peak = f"R{r}"
#     osf = int(OVERSAMPLE_FACTOR)
#     m_ = int(WEIGHT_MODE)
#     t = compact_float_tag(TAU, scale=100)
#     return f"E{e}-A{a}-{peak}-OS{osf}-M{m_}-T{t}"
# 
# def get_mode_code():
#     if not USE_CLUSTERED:
#         return "NC"
#     if cluster_label:
#         # keep short: CK4
#         return f"CK{cluster_label}"
#     return "CL"
# 
# def write_runinfo(run_dir, info: dict):
#     try:
#         p = os.path.join(run_dir, "runinfo.txt")
#         lines = []
#         lines.append("===== RUN INFO =====")
#         for k in sorted(info.keys()):
#             lines.append(f"{k}: {info[k]}")
#         with open(p, "w", encoding="utf-8") as f:
#             f.write("\n".join(lines) + "\n")
#     except Exception as e:
#         print(f"⚠️ Could not write runinfo.txt in {run_dir}: {e}")
# 
# MODE_CODE = get_mode_code()
# 
# print("\n📂 Base model root for this code version:")
# print("   ", base_model_root)
# print()
# 
# # ==============================================================
# # 🧠 Peak definition + Oversampling inputs
# # ==============================================================
# print("-------------------------------------------")
# print("روش تعریف پیک را انتخاب کن:")
# print("1 = relative (thresh × max هر تایم‌سری)")
# print("2 = quantile (Top-K% هر تایم‌سری)")
# print("-------------------------------------------")
# _peak_choice = input("انتخاب (1 یا 2): ").strip()
# PEAK_METHOD = "quantile" if _peak_choice == "2" else "relative"
# 
# PEAK_Q = 0.95
# if PEAK_METHOD == "quantile":
#     q_in = input("مقدار q برای Quantile را وارد کن (مثلاً 0.95 یعنی 5% بالایی). خالی = 0.95: ").strip()
#     if q_in:
#         try:
#             PEAK_Q = float(q_in)
#         except Exception:
#             PEAK_Q = 0.95
# 
# os_in = input("Oversampling factor (عدد صحیح >=1؛ 1 یعنی خاموش). خالی = 1: ").strip()
# OVERSAMPLE_FACTOR = 1
# if os_in:
#     try:
#         OVERSAMPLE_FACTOR = int(os_in)
#         if OVERSAMPLE_FACTOR < 1:
#             OVERSAMPLE_FACTOR = 1
#     except Exception:
#         OVERSAMPLE_FACTOR = 1
# 
# # ==============================================================
# # 🔧 Scenarios
# # ==============================================================
# SCENARIOS = [
#     {"EPOCHS": 76, "ALPHA": 1, "THRESH": 0.5, "WEIGHT_MODE": 1, "TAU": 0.05},
# ]
# 
# # ==============================================================
# # 🧭 Data folders check
# # ==============================================================
# if not os.path.isdir(gm_root_dir):
#     raise FileNotFoundError(f"❌ GM root dir پیدا نشد: {gm_root_dir}")
# if not os.path.isdir(tha_root_dir):
#     raise FileNotFoundError(f"❌ THA root dir پیدا نشد: {tha_root_dir}")
# 
# available_heights = sorted(
#     name for name in os.listdir(gm_root_dir)
#     if os.path.isdir(os.path.join(gm_root_dir, name)) and name.startswith("H")
# )
# if not available_heights:
#     raise ValueError(f"❌ هیچ پوشه‌ای به شکل H* در {gm_root_dir} پیدا نشد. مرحله ۳ را چک کن.")
# 
# print("📏 ارتفاع‌های موجود در داده‌ها:")
# for h in available_heights:
#     print("  -", h)
# 
# print("\n-------------------------------------------")
# print("برای چه ارتفاع‌هایی می‌خواهی آموزش انجام شود؟")
# print("مثال: H2 H3 H4  یا  2 3 4  (خالی = همه)")
# print("-------------------------------------------\n")
# 
# heights_raw = input("ارتفاع‌ها را وارد کن (خالی = همه): ").strip()
# 
# if not heights_raw:
#     height_tags = available_heights[:]
#     print("\n✅ همه ارتفاع‌های موجود انتخاب شدند.")
# else:
#     tokens = heights_raw.replace(',', ' ').split()
#     selected, invalid = [], []
#     for tok in tokens:
#         tok = tok.strip()
#         if not tok:
#             continue
#         if tok.startswith("H"):
#             tag = tok
#         else:
#             try:
#                 v = float(tok)
#                 if v.is_integer():
#                     tag = f"H{int(v)}"
#                 else:
#                     tag = "H" + str(v).replace('.', 'p')
#             except Exception:
#                 invalid.append(tok)
#                 continue
#         if tag in available_heights:
#             selected.append(tag)
#         else:
#             invalid.append(tok)
#     height_tags = list(dict.fromkeys(selected))
#     if invalid:
#         print("\n⚠️ این مقادیر معتبر نبودند یا داده‌ای برای آنها پیدا نشد و نادیده گرفته شدند:")
#         for x in invalid:
#             print("  -", x)
#     if not height_tags:
#         print("❌ هیچ ارتفاع معتبری انتخاب نشد؛ برنامه متوقف می‌شود.")
#         sys.exit(1)
# 
# print("\n📏 ارتفاع‌های نهایی انتخاب‌شده برای آموزش:")
# print("  " + ", ".join(height_tags))
# print()
# 
# def height_value_from_tag(h_tag: str) -> float:
#     s = h_tag[1:]
#     s = s.replace('p', '.')
#     return float(s)
# 
# height_values = {h_tag: height_value_from_tag(h_tag) for h_tag in height_tags}
# print("🔢 نگاشت ارتفاع‌ها (tag → value):")
# for h_tag, hv in height_values.items():
#     print(f"  {h_tag} → {hv}")
# 
# # ==============================================================
# # ↔️ Training mode selection
# # ==============================================================
# print("\n-------------------------------------------")
# print("آیا می‌خواهی همهٔ ارتفاع‌ها با هم و با فیچر ارتفاع آموزش داده شوند؟")
# print("   1 = بله، یک مدل مشترک با فیچر ارتفاع (Multi-height + Feature H)")
# print("   0 = خیر، برای هر ارتفاع جداگانه مدل مستقل آموزش داده شود")
# print("-------------------------------------------\n")
# 
# mh_choice = input("انتخابت را وارد کن (1 یا 0): ").strip()
# USE_MULTI_HEIGHT = (mh_choice == "1")
# 
# if USE_MULTI_HEIGHT:
#     print("✅ مود ۱: مدل مشترک برای همه ارتفاع‌ها با فیچر ارتفاع (X = [GM , H])")
#     TRAIN_CODE = "GH"
# else:
#     print("✅ مود ۲: مدل جداگانه برای هر ارتفاع، بدون فیچر ارتفاع (X = [GM])")
#     TRAIN_CODE = "PH"
# 
# # ==============================================================
# # ⚙️ Common settings / helpers
# # ==============================================================
# PAD = -999.0
# BATCH_SIZE = 20
# 
# def param_to_str(v):
#     v = float(v)
#     if v.is_integer():
#         return f"{int(v)}.0"
#     return str(v)
# 
# def get_data_paths_for_height(h_tag):
#     if USE_CLUSTERED:
#         gm_dir  = os.path.join(gm_root_dir,  h_tag, "cluster_balanced_global")
#         tha_dir = os.path.join(tha_root_dir, h_tag, "cluster_balanced_global")
#         x_name  = f"X_data_cluster_balanced_global_{h_tag}.npy"
#         y_name  = f"Y_data_cluster_balanced_global_{h_tag}.npy"
#     else:
#         gm_dir  = os.path.join(gm_root_dir,  h_tag)
#         tha_dir = os.path.join(tha_root_dir, h_tag)
#         x_name  = f"X_data_{h_tag}.npy"
#         y_name  = f"Y_data_{h_tag}.npy"
#     return os.path.join(gm_dir, x_name), os.path.join(tha_dir, y_name)
# 
# # ==============================================================
# # ✅ Peak-aware weighted MSE loss (relative/quantile + binary/soft)
# # ==============================================================
# def make_weighted_mse_loss(PAD_value, alpha, thresh, weight_mode=2, tau=0.05,
#                           peak_method="relative", q=0.95):
#     PAD_val = tf.constant(PAD_value, dtype=tf.float32)
# 
#     def weighted_mse_loss(y_true, y_pred):
#         y_true = tf.cast(y_true, tf.float32)
#         y_pred = tf.cast(y_pred, tf.float32)
# 
#         mask = tf.cast(tf.not_equal(y_true, PAD_val), tf.float32)
#         abs_y = tf.abs(y_true) * mask
# 
#         if peak_method == "relative":
#             max_abs = tf.reduce_max(abs_y, axis=1, keepdims=True) + 1e-6
#             rel = abs_y / max_abs
#             if weight_mode == 1:
#                 peak_strength = tf.cast(rel >= thresh, tf.float32)
#             else:
#                 t = tf.constant(float(tau), dtype=tf.float32)
#                 peak_strength = tf.sigmoid((rel - thresh) / (t + 1e-6))
# 
#         elif peak_method == "quantile":
#             abs_y_2d = tf.squeeze(abs_y, axis=-1)
#             sorted_vals = tf.sort(abs_y_2d, axis=1)
#             T = tf.shape(sorted_vals)[1]
#             q_clamped = tf.clip_by_value(tf.cast(q, tf.float32), 0.0, 1.0)
#             idx = tf.cast(tf.math.round(q_clamped * tf.cast(T - 1, tf.float32)), tf.int32)
#             idx = tf.clip_by_value(idx, 0, T - 1)
#             batch_indices = tf.range(tf.shape(sorted_vals)[0])
#             gather_idx = tf.stack([batch_indices, tf.fill(tf.shape(batch_indices), idx)], axis=1)
#             thr_vals = tf.gather_nd(sorted_vals, gather_idx)
#             thr_vals = tf.reshape(thr_vals, (-1, 1, 1))
#             if weight_mode == 1:
#                 peak_strength = tf.cast(abs_y >= thr_vals, tf.float32)
#             else:
#                 t = tf.constant(float(tau), dtype=tf.float32)
#                 peak_strength = tf.sigmoid((abs_y - thr_vals) / (t + 1e-6))
#         else:
#             raise ValueError("Invalid peak_method. Use 'relative' or 'quantile'.")
# 
#         w = 1.0 + alpha * peak_strength
#         w = w * mask
# 
#         w_sum = tf.reduce_sum(w, axis=1, keepdims=True) + 1e-6
#         m_sum = tf.reduce_sum(mask, axis=1, keepdims=True) + 1e-6
#         w = w * (m_sum / w_sum)
# 
#         sq = tf.square(y_true - y_pred) * mask
#         loss_per_sample = tf.reduce_sum(w * sq, axis=1) / m_sum
#         return tf.reduce_mean(loss_per_sample)
# 
#     return weighted_mse_loss
# 
# def build_model(input_dim, PAD_value, loss_fn):
#     inp = Input(shape=(None, input_dim), name="dense_input")
#     x = Masking(mask_value=PAD_value, name="masking")(inp)
#     x = LSTM(100, return_sequences=True, name="lstm1")(x)
#     x = Activation('relu', name="relu1")(x)
#     x = LSTM(100, return_sequences=True, name="lstm2")(x)
#     x = Activation('relu', name="relu2")(x)
#     x = Dense(100, name="dense1")(x)
#     out = Dense(1, name="dense_out")(x)
# 
#     model = Model(inputs=inp, outputs=out, name="lstm_masked_dense")
#     model.compile(loss=loss_fn, optimizer=Adam(learning_rate=0.001), metrics=['mse'])
#     return model
# 
# class PeriodicSaver(tf.keras.callbacks.Callback):
#     def __init__(self, save_dir, backup_dir=None, period=10, keep_last=1):
#         super().__init__()
#         self.save_dir = save_dir
#         self.backup_dir = backup_dir
#         self.period = int(period)
#         self.keep_last = max(int(keep_last), 1)
#         os.makedirs(self.save_dir, exist_ok=True)
# 
#     def _prune_checkpoints(self):
#         files = glob.glob(os.path.join(self.save_dir, "ckpt_epoch_*.keras"))
#         if len(files) <= self.keep_last:
#             return
# 
#         def parse_epoch(p):
#             m = re.search(r"ckpt_epoch_(\d+)\.keras$", os.path.basename(p))
#             return int(m.group(1)) if m else 0
# 
#         files.sort(key=parse_epoch)
#         for f in files[:-self.keep_last]:
#             try:
#                 os.remove(f)
#                 print(f"🧹 old checkpoint removed: {f}")
#             except Exception as e:
#                 print(f"⚠️ could not remove {f}: {e}")
# 
#     def _prune_backup(self):
#         if not self.backup_dir or not os.path.exists(self.backup_dir):
#             return
#         entries = [os.path.join(self.backup_dir, n) for n in os.listdir(self.backup_dir)]
#         if len(entries) <= self.keep_last:
#             return
#         entries.sort(key=lambda p: os.path.getmtime(p))
#         for p in entries[:-self.keep_last]:
#             try:
#                 if os.path.isdir(p):
#                     shutil.rmtree(p, ignore_errors=True)
#                 else:
#                     os.remove(p)
#                 print(f"🧹 old backup removed: {p}")
#             except Exception as e:
#                 print(f"⚠️ could not remove backup {p}: {e}")
# 
#     def on_epoch_end(self, epoch, logs=None):
#         ep = epoch + 1
#         if ep % self.period == 0:
#             path = os.path.join(self.save_dir, f"ckpt_epoch_{ep:04d}.keras")
#             self.model.save(path)
#             print(f"💾 Periodic checkpoint saved: {path}")
#             self._prune_checkpoints()
#             self._prune_backup()
# 
# def find_latest_periodic_checkpoint(directory):
#     files = glob.glob(os.path.join(directory, "ckpt_epoch_*.keras"))
#     if not files:
#         return None, 0
# 
#     def parse_epoch(p):
#         m = re.search(r"ckpt_epoch_(\d+)\.keras$", os.path.basename(p))
#         return int(m.group(1)) if m else 0
# 
#     files.sort(key=parse_epoch)
#     latest = files[-1]
#     return latest, parse_epoch(latest)
# 
# def is_scenario_finished(model_dir, expected_epochs):
#     progress_path = os.path.join(model_dir, "progress.npy")
#     if not os.path.exists(progress_path):
#         return False
#     try:
#         data = np.load(progress_path, allow_pickle=True).item()
#         trained = int(data.get("epochs_trained", data.get("epochs", 0)))
#         return trained >= expected_epochs
#     except Exception as e:
#         print(f"⚠️ could not read progress for {model_dir}: {e}")
#         return False
# 
# # ==============================================================
# # Oversampling helpers (train only)
# # ==============================================================
# def has_peak_np(y, peak_method, thresh, q):
#     abs_y = np.abs(y.reshape(-1))
#     if abs_y.size == 0:
#         return False
#     if peak_method == "relative":
#         m = abs_y.max()
#         if m <= 1e-12:
#             return False
#         rel = abs_y / m
#         return np.any(rel >= thresh)
#     else:
#         thr = np.quantile(abs_y, q)
#         return np.any(abs_y >= thr)
# 
# def oversample_lists(X_list, Y_list, peak_method, thresh, q, factor):
#     if factor <= 1:
#         return X_list, Y_list
#     X_out, Y_out = [], []
#     for x, y in zip(X_list, Y_list):
#         X_out.append(x); Y_out.append(y)
#         if has_peak_np(y, peak_method, thresh, q):
#             for _ in range(factor - 1):
#                 X_out.append(x)
#                 Y_out.append(y)
#     return X_out, Y_out
# 
# # ==============================================================
# # Dataset builders (list -> tf.data)
# # ==============================================================
# def gen_from_lists(x_list, y_list):
#     for x, y in zip(x_list, y_list):
#         yield x, y
# 
# def make_datasets(X_train_list, Y_train_list, X_val_list, Y_val_list, input_dim):
#     output_signature = (
#         tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32),
#         tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
#     )
# 
#     ds_train = tf.data.Dataset.from_generator(
#         lambda: gen_from_lists(X_train_list, Y_train_list),
#         output_signature=output_signature
#     ).padded_batch(
#         BATCH_SIZE,
#         padded_shapes=([None, input_dim], [None, 1]),
#         padding_values=(PAD, PAD),
#         drop_remainder=False
#     ).prefetch(tf.data.AUTOTUNE)
# 
#     ds_val = tf.data.Dataset.from_generator(
#         lambda: gen_from_lists(X_val_list, Y_val_list),
#         output_signature=output_signature
#     ).padded_batch(
#         BATCH_SIZE,
#         padded_shapes=([None, input_dim], [None, 1]),
#         padding_values=(PAD, PAD),
#         drop_remainder=False
#     ).prefetch(tf.data.AUTOTUNE)
# 
#     return ds_train, ds_val
# 
# # ==============================================================
# # 🚀 MODE 1: Global multi-height (+ height feature)
# # ==============================================================
# if USE_MULTI_HEIGHT:
#     INPUT_DIM = 2  # [GM, H]
#     X_all, Y_all = [], []
# 
#     for h_tag in height_tags:
#         print("\n" + "#" * 80)
#         print(f"📥 بارگذاری داده برای ارتفاع: {h_tag}")
#         print("#" * 80)
# 
#         x_data_path, y_data_path = get_data_paths_for_height(h_tag)
# 
#         if not os.path.exists(x_data_path):
#             print(f"⚠️ X برای {h_tag} پیدا نشد: {x_data_path} → رد می‌شود.")
#             continue
#         if not os.path.exists(y_data_path):
#             print(f"⚠️ Y برای {h_tag} پیدا نشد: {y_data_path} → رد می‌شود.")
#             continue
# 
#         X_dict = np.load(x_data_path, allow_pickle=True).item()
#         Y_dict = np.load(y_data_path, allow_pickle=True).item()
# 
#         common_keys = sorted(set(X_dict.keys()) & set(Y_dict.keys()))
#         if not common_keys:
#             print(f"❌ هیچ کلید مشترکی برای {h_tag} پیدا نشد. رد می‌شود.")
#             continue
# 
#         h_val = np.float32(height_values[h_tag])
# 
#         for k in common_keys:
#             x = X_dict[k].reshape(-1, 1).astype('float32')
#             y = Y_dict[k].reshape(-1, 1).astype('float32')
# 
#             T = x.shape[0]
#             h_col = np.full((T, 1), h_val, dtype='float32')
#             x_feat = np.concatenate([x, h_col], axis=1)
# 
#             X_all.append(x_feat)
#             Y_all.append(y)
# 
#     num_samples = len(X_all)
#     if num_samples == 0:
#         raise ValueError("❌ هیچ نمونه‌ای برای آموزش پیدا نشد (بعد از فیلتر ارتفاع‌ها/کلاسترها).")
# 
#     print(f"\n📦 تعداد کل نمونه‌ها (همه ارتفاع‌ها با هم): {num_samples}")
# 
#     scaler_X = MinMaxScaler(feature_range=(-1, 1))
#     scaler_Y = MinMaxScaler(feature_range=(-1, 1))
# 
#     X_concat = np.concatenate(X_all, axis=0)
#     Y_concat = np.concatenate(Y_all, axis=0)
# 
#     scaler_X.fit(X_concat)
#     scaler_Y.fit(Y_concat)
# 
#     X_all = [scaler_X.transform(x) for x in X_all]
#     Y_all = [scaler_Y.transform(y) for y in Y_all]
# 
#     # short meta dir
#     meta_dir = os.path.join(base_model_root, f"_META_{MODE_CODE}-{TRAIN_CODE}")
#     os.makedirs(meta_dir, exist_ok=True)
# 
#     if is_linear:
#         scaler_X_path = os.path.join(meta_dir, "scaler_X_linear.pkl")
#         scaler_Y_path = os.path.join(meta_dir, "scaler_Y_linear.pkl")
#     else:
#         scaler_X_path = os.path.join(meta_dir, "scaler_X_nonlinear.pkl")
#         scaler_Y_path = os.path.join(meta_dir, "scaler_Y_nonlinear.pkl")
# 
#     joblib.dump(scaler_X, scaler_X_path)
#     joblib.dump(scaler_Y, scaler_Y_path)
#     print("\n💾 Global scalers saved:")
#     print("  →", scaler_X_path)
#     print("  →", scaler_Y_path)
# 
#     split_idx_path = os.path.join(meta_dir, f"split_idx_seed{SEED}.npy")
# 
#     if os.path.exists(split_idx_path):
#         idx = np.load(split_idx_path)
#         if len(idx) != num_samples:
#             idx = np.arange(num_samples)
#             rng = np.random.default_rng(SEED)
#             rng.shuffle(idx)
#             np.save(split_idx_path, idx)
#             print("⚠️ طول داده‌ها تغییر کرده بود؛ split جدید ساخته شد.")
#         else:
#             print("✅ Fixed global split indices loaded:", split_idx_path)
#     else:
#         idx = np.arange(num_samples)
#         rng = np.random.default_rng(SEED)
#         rng.shuffle(idx)
#         np.save(split_idx_path, idx)
#         print("✅ Fixed global split indices saved:", split_idx_path)
# 
#     train_split = int(0.50 * num_samples)
#     val_split   = int(0.63 * num_samples)
# 
#     X_train_base = [X_all[i] for i in idx[:train_split]]
#     Y_train_base = [Y_all[i] for i in idx[:train_split]]
# 
#     X_val_list   = [X_all[i] for i in idx[train_split:val_split]]
#     Y_val_list   = [Y_all[i] for i in idx[train_split:val_split]]
# 
#     print(f"\n📦 Global split | Train: {len(X_train_base)}  Val: {len(X_val_list)}  Test: {num_samples - val_split}")
# 
#     heights_str = ", ".join(height_tags)
# 
#     for scen in SCENARIOS:
#         EPOCHS = int(scen["EPOCHS"])
#         ALPHA  = float(scen["ALPHA"])
#         THRESH = float(scen["THRESH"])
#         WEIGHT_MODE = int(scen.get("WEIGHT_MODE", 2))
#         TAU = float(scen.get("TAU", 0.05))
# 
#         X_train_list, Y_train_list = oversample_lists(
#             X_train_base, Y_train_base,
#             PEAK_METHOD, THRESH, PEAK_Q, OVERSAMPLE_FACTOR
#         )
# 
#         scen_short = make_scen_short(EPOCHS, ALPHA, PEAK_METHOD, PEAK_Q, THRESH, OVERSAMPLE_FACTOR, WEIGHT_MODE, TAU)
#         run_tag = f"{MODE_CODE}-{TRAIN_CODE}__{scen_short}"
#         model_dir = os.path.join(base_model_root, run_tag)
#         os.makedirs(model_dir, exist_ok=True)
# 
#         model_path    = os.path.join(model_dir, "LSTM.keras")
#         backup_dir    = os.path.join(model_dir, "backup")
#         ckpt_dir      = os.path.join(model_dir, "checkpoints")
#         progress_path = os.path.join(model_dir, "progress.npy")
#         os.makedirs(backup_dir, exist_ok=True)
#         os.makedirs(ckpt_dir, exist_ok=True)
# 
#         if is_scenario_finished(model_dir, EPOCHS):
#             print("\n" + "=" * 80)
#             print(f"⏩ سناریو {run_tag} قبلاً کامل شده است. رد می‌شود.")
#             print("=" * 80)
#             continue
# 
#         print("\n" + "=" * 80)
#         print(f"🚀 شروع سناریو (Global + Height): {run_tag}")
#         print(f"   → EPOCHS={EPOCHS}, ALPHA={ALPHA}, THRESH={THRESH}, WEIGHT_MODE={WEIGHT_MODE}, TAU={TAU}")
#         print(f"   → PeakMethod={PEAK_METHOD}, Q={PEAK_Q}, Oversample={OVERSAMPLE_FACTOR}")
#         print(f"   → Heights: {heights_str}")
#         print("=" * 80)
# 
#         # --- NEW: write initial runinfo at scenario start ---
#         runinfo_init = {
#             'status': 'STARTED',
#             'run_tag': run_tag,
#             'script_name': script_name,
#             'script_tag': script_tag,
#             'seed': SEED,
#             'mode_code': MODE_CODE,
#             'train_code': TRAIN_CODE,
#             'use_clustered': USE_CLUSTERED,
#             'cluster_label': cluster_label,
#             'peak_method': PEAK_METHOD,
#             'peak_q': float(PEAK_Q),
#             'oversample_factor': int(OVERSAMPLE_FACTOR),
#             'alpha': float(ALPHA),
#             'thresh': float(THRESH),
#             'weight_mode': int(WEIGHT_MODE),
#             'tau': float(TAU),
#             'epochs_planned': int(EPOCHS),
#             'gm_root_dir': gm_root_dir,
#             'tha_root_dir': tha_root_dir,
#             'model_dir': model_dir,
#             'model_path': model_path,
#             'backup_dir': backup_dir,
#             'ckpt_dir': ckpt_dir,
#             'progress_path': progress_path,
#             'heights_used': height_tags,
#         }
#         write_runinfo(model_dir, runinfo_init)
# 
#         ds_train, ds_val = make_datasets(X_train_list, Y_train_list, X_val_list, Y_val_list, INPUT_DIM)
# 
#         tf.random.set_seed(SEED)
#         np.random.seed(SEED)
#         random.seed(SEED)
# 
#         loss_fn = make_weighted_mse_loss(
#             PAD, ALPHA, THRESH,
#             weight_mode=WEIGHT_MODE, tau=TAU,
#             peak_method=PEAK_METHOD, q=PEAK_Q
#         )
#         model = build_model(INPUT_DIM, PAD, loss_fn)
# 
#         checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
#             model_path, monitor='val_loss', save_best_only=True, verbose=1
#         )
#         reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
#             monitor='val_loss', factor=0.5, patience=10, verbose=1, min_lr=1e-6
#         )
#         backup_cb = tf.keras.callbacks.BackupAndRestore(backup_dir=backup_dir)
#         periodic_ckpt = PeriodicSaver(
#             save_dir=ckpt_dir, backup_dir=backup_dir,
#             period=10, keep_last=1
#         )
# 
#         initial_epoch = 0
#         if not os.path.exists(backup_dir) or not os.listdir(backup_dir):
#             ckpt_path, ep = find_latest_periodic_checkpoint(ckpt_dir)
#             if ckpt_path:
#                 try:
#                     tmp_model = load_model(ckpt_path, compile=False)
#                     tmp_model.compile(loss=loss_fn, optimizer=Adam(learning_rate=0.001), metrics=['mse'])
#                     model = tmp_model
#                     initial_epoch = ep
#                     print(f"🔁 Resuming from periodic checkpoint: {ckpt_path} (initial_epoch={initial_epoch})")
#                 except Exception as e:
#                     print(f"⚠️ Failed to load periodic checkpoint: {e}")
# 
#         history = model.fit(
#             ds_train,
#             validation_data=ds_val,
#             epochs=EPOCHS,
#             initial_epoch=initial_epoch,
#             callbacks=[backup_cb, checkpoint_best, periodic_ckpt, reduce_lr],
#             verbose=1
#         )
# 
#         epochs_trained = initial_epoch + len(history.history.get('loss', []))
# 
#         progress_data = {
#             'train_loss': history.history.get('loss', []),
#             'val_loss': history.history.get('val_loss', []),
#             'best_val_loss': float(min(history.history.get('val_loss', [np.inf]))),
#             'epochs': EPOCHS,
#             'epochs_trained': int(epochs_trained),
#             'heights_used': height_tags,
#             'use_clustered': USE_CLUSTERED,
#             'cluster_label': cluster_label,
#             'mode_code': MODE_CODE,
#             'train_code': TRAIN_CODE,
#             'peak_method': PEAK_METHOD,
#             'peak_q': float(PEAK_Q),
#             'oversample_factor': int(OVERSAMPLE_FACTOR),
#             'alpha': float(ALPHA),
#             'thresh': float(THRESH),
#             'weight_mode': int(WEIGHT_MODE),
#             'tau': float(TAU),
#             'seed': SEED,
#             'script_name': script_name,
#             'script_tag': script_tag,
#             'gm_root_dir': gm_root_dir,
#             'tha_root_dir': tha_root_dir,
#             'scaler_X_path': scaler_X_path,
#             'scaler_Y_path': scaler_Y_path,
#             'split_idx_path': split_idx_path,
#         }
#         np.save(progress_path, progress_data)
#         print("💾 Progress saved:", progress_path)
# 
#         runinfo = dict(progress_data)
#         runinfo['status'] = 'FINISHED'
#         runinfo['model_dir'] = model_dir
#         runinfo['model_path'] = model_path
#         runinfo['backup_dir'] = backup_dir
#         runinfo['ckpt_dir'] = ckpt_dir
#         runinfo['progress_path'] = progress_path
#         write_runinfo(model_dir, runinfo)
# 
#         try:
#             plt.figure()
#             plt.plot(progress_data['train_loss'], label='Train Loss')
#             plt.plot(progress_data['val_loss'], label='Val Loss')
#             plt.xlabel('Epoch')
#             plt.ylabel('Loss')
#             plt.title(f'All Heights [{heights_str}] - Loss - {run_tag}')
#             plt.legend()
#             plt.tight_layout()
# 
#             loss_plot_path = os.path.join(model_dir, f"loss_curve_{run_tag}.png")
#             plt.savefig(loss_plot_path, dpi=300)
#             plt.close()
#             print("📊 Loss curve saved to:", loss_plot_path)
#         except Exception as e:
#             print(f"⚠️ Plot error (loss curve): {e}")
# 
#         print(f"✅ Scenario {run_tag} finished.\n")
# 
#     print("🎉 همه سناریوها در مود Multi-height تمام شدند.")
# 
# # ==============================================================
# # 🚀 MODE 2: Per-height (no height feature)
# # ==============================================================
# else:
#     INPUT_DIM = 1
# 
#     for h_tag in height_tags:
#         print("\n" + "#" * 80)
#         print(f"🏗️ شروع آموزش جداگانه برای ارتفاع: {h_tag}")
#         print("#" * 80)
# 
#         x_data_path, y_data_path = get_data_paths_for_height(h_tag)
# 
#         if not os.path.exists(x_data_path):
#             print(f"⚠️ X_data برای {h_tag} پیدا نشد: {x_data_path} → رد می‌شود.")
#             continue
#         if not os.path.exists(y_data_path):
#             print(f"⚠️ Y_data برای {h_tag} پیدا نشد: {y_data_path} → رد می‌شود.")
#             continue
# 
#         X_dict = np.load(x_data_path, allow_pickle=True).item()
#         Y_dict = np.load(y_data_path, allow_pickle=True).item()
# 
#         common_keys = sorted(set(X_dict.keys()) & set(Y_dict.keys()))
#         if not common_keys:
#             print(f"❌ هیچ کلید مشترکی بین X و Y برای {h_tag} پیدا نشد. رد می‌شود.")
#             continue
# 
#         X_all, Y_all = [], []
#         for k in common_keys:
#             x = X_dict[k].reshape(-1, 1).astype('float32')
#             y = Y_dict[k].reshape(-1, 1).astype('float32')
#             X_all.append(x)
#             Y_all.append(y)
# 
#         num_samples = len(X_all)
#         if num_samples == 0:
#             print(f"⚠️ هیچ نمونه‌ای برای {h_tag} وجود ندارد. رد می‌شود.")
#             continue
# 
#         print(f"📦 تعداد نمونه‌ها برای {h_tag}: {num_samples}")
# 
#         scaler_X = MinMaxScaler(feature_range=(-1, 1))
#         scaler_Y = MinMaxScaler(feature_range=(-1, 1))
# 
#         X_concat = np.concatenate(X_all, axis=0)
#         Y_concat = np.concatenate(Y_all, axis=0)
# 
#         scaler_X.fit(X_concat)
#         scaler_Y.fit(Y_concat)
# 
#         X_all = [scaler_X.transform(x) for x in X_all]
#         Y_all = [scaler_Y.transform(y) for y in Y_all]
# 
#         height_meta_dir = os.path.join(base_model_root, f"_META_{MODE_CODE}-{TRAIN_CODE}_{h_tag}")
#         os.makedirs(height_meta_dir, exist_ok=True)
# 
#         if is_linear:
#             scaler_X_path = os.path.join(height_meta_dir, "scaler_X_linear.pkl")
#             scaler_Y_path = os.path.join(height_meta_dir, "scaler_Y_linear.pkl")
#         else:
#             scaler_X_path = os.path.join(height_meta_dir, "scaler_X_nonlinear.pkl")
#             scaler_Y_path = os.path.join(height_meta_dir, "scaler_Y_nonlinear.pkl")
# 
#         joblib.dump(scaler_X, scaler_X_path)
#         joblib.dump(scaler_Y, scaler_Y_path)
#         print("💾 height-scalers saved:")
#         print("  →", scaler_X_path)
#         print("  →", scaler_Y_path)
# 
#         split_idx_path = os.path.join(height_meta_dir, f"split_idx_seed{SEED}.npy")
# 
#         if os.path.exists(split_idx_path):
#             idx = np.load(split_idx_path)
#             if len(idx) != num_samples:
#                 idx = np.arange(num_samples)
#                 rng = np.random.default_rng(SEED)
#                 rng.shuffle(idx)
#                 np.save(split_idx_path, idx)
#                 print("⚠️ طول داده‌ها تغییر کرده بود؛ split جدید ساخته شد.")
#             else:
#                 print("✅ Fixed split indices loaded:", split_idx_path)
#         else:
#             idx = np.arange(num_samples)
#             rng = np.random.default_rng(SEED)
#             rng.shuffle(idx)
#             np.save(split_idx_path, idx)
#             print("✅ Fixed split indices saved:", split_idx_path)
# 
#         train_split = int(0.50 * num_samples)
#         val_split   = int(0.63 * num_samples)
# 
#         X_train_base = [X_all[i] for i in idx[:train_split]]
#         Y_train_base = [Y_all[i] for i in idx[:train_split]]
# 
#         X_val_list   = [X_all[i] for i in idx[train_split:val_split]]
#         Y_val_list   = [Y_all[i] for i in idx[train_split:val_split]]
# 
#         for scen in SCENARIOS:
#             EPOCHS = int(scen["EPOCHS"])
#             ALPHA  = float(scen["ALPHA"])
#             THRESH = float(scen["THRESH"])
#             WEIGHT_MODE = int(scen.get("WEIGHT_MODE", 2))
#             TAU = float(scen.get("TAU", 0.05))
# 
#             X_train_list, Y_train_list = oversample_lists(
#                 X_train_base, Y_train_base,
#                 PEAK_METHOD, THRESH, PEAK_Q, OVERSAMPLE_FACTOR
#             )
# 
#             scen_short = make_scen_short(EPOCHS, ALPHA, PEAK_METHOD, PEAK_Q, THRESH, OVERSAMPLE_FACTOR, WEIGHT_MODE, TAU)
#             run_tag = f"{MODE_CODE}-{TRAIN_CODE}_{h_tag}__{scen_short}"
#             model_dir = os.path.join(base_model_root, run_tag)
#             os.makedirs(model_dir, exist_ok=True)
# 
#             model_path    = os.path.join(model_dir, "LSTM.keras")
#             backup_dir    = os.path.join(model_dir, "backup")
#             ckpt_dir      = os.path.join(model_dir, "checkpoints")
#             progress_path = os.path.join(model_dir, "progress.npy")
# 
#             if is_scenario_finished(model_dir, EPOCHS):
#                 print("\n" + "=" * 80)
#                 print(f"⏩ {h_tag} | سناریو {run_tag} قبلاً کامل شده است. رد می‌شود.")
#                 print("=" * 80)
#                 continue
# 
#             print("\n" + "=" * 80)
#             print(f"🚀 {h_tag} | شروع سناریو: {run_tag}")
#             print(f"   → EPOCHS={EPOCHS}, ALPHA={ALPHA}, THRESH={THRESH}, WEIGHT_MODE={WEIGHT_MODE}, TAU={TAU}")
#             print(f"   → PeakMethod={PEAK_METHOD}, Q={PEAK_Q}, Oversample={OVERSAMPLE_FACTOR}")
#             print("=" * 80)
# 
#             os.makedirs(ckpt_dir, exist_ok=True)
#             os.makedirs(backup_dir, exist_ok=True)
# 
#             # --- NEW: write initial runinfo at scenario start ---
#             runinfo_init = {
#                 'status': 'STARTED',
#                 'run_tag': run_tag,
#                 'height': h_tag,
#                 'script_name': script_name,
#                 'script_tag': script_tag,
#                 'seed': SEED,
#                 'mode_code': MODE_CODE,
#                 'train_code': TRAIN_CODE,
#                 'use_clustered': USE_CLUSTERED,
#                 'cluster_label': cluster_label,
#                 'peak_method': PEAK_METHOD,
#                 'peak_q': float(PEAK_Q),
#                 'oversample_factor': int(OVERSAMPLE_FACTOR),
#                 'alpha': float(ALPHA),
#                 'thresh': float(THRESH),
#                 'weight_mode': int(WEIGHT_MODE),
#                 'tau': float(TAU),
#                 'epochs_planned': int(EPOCHS),
#                 'gm_root_dir': gm_root_dir,
#                 'tha_root_dir': tha_root_dir,
#                 'model_dir': model_dir,
#                 'model_path': model_path,
#                 'backup_dir': backup_dir,
#                 'ckpt_dir': ckpt_dir,
#                 'progress_path': progress_path,
#             }
#             write_runinfo(model_dir, runinfo_init)
# 
#             ds_train, ds_val = make_datasets(X_train_list, Y_train_list, X_val_list, Y_val_list, INPUT_DIM)
# 
#             tf.random.set_seed(SEED)
#             np.random.seed(SEED)
#             random.seed(SEED)
# 
#             loss_fn = make_weighted_mse_loss(
#                 PAD, ALPHA, THRESH,
#                 weight_mode=WEIGHT_MODE, tau=TAU,
#                 peak_method=PEAK_METHOD, q=PEAK_Q
#             )
#             model = build_model(INPUT_DIM, PAD, loss_fn)
# 
#             checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
#                 model_path, monitor='val_loss', save_best_only=True, verbose=1
#             )
#             reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
#                 monitor='val_loss', factor=0.5, patience=10, verbose=1, min_lr=1e-6
#             )
#             backup_cb = tf.keras.callbacks.BackupAndRestore(backup_dir=backup_dir)
#             periodic_ckpt = PeriodicSaver(
#                 save_dir=ckpt_dir, backup_dir=backup_dir,
#                 period=10, keep_last=1
#             )
# 
#             initial_epoch = 0
#             if not os.path.exists(backup_dir) or not os.listdir(backup_dir):
#                 ckpt_path, ep = find_latest_periodic_checkpoint(ckpt_dir)
#                 if ckpt_path:
#                     try:
#                         tmp_model = load_model(ckpt_path, compile=False)
#                         tmp_model.compile(loss=loss_fn, optimizer=Adam(learning_rate=0.001), metrics=['mse'])
#                         model = tmp_model
#                         initial_epoch = ep
#                         print(f"🔁 {h_tag} | Resuming from checkpoint: {ckpt_path} (initial_epoch={initial_epoch})")
#                     except Exception as e:
#                         print(f"⚠️ Failed to load periodic checkpoint: {e}")
# 
#             history = model.fit(
#                 ds_train,
#                 validation_data=ds_val,
#                 epochs=EPOCHS,
#                 initial_epoch=initial_epoch,
#                 callbacks=[backup_cb, checkpoint_best, periodic_ckpt, reduce_lr],
#                 verbose=1
#             )
# 
#             epochs_trained = initial_epoch + len(history.history.get('loss', []))
# 
#             progress_data = {
#                 'train_loss': history.history.get('loss', []),
#                 'val_loss': history.history.get('val_loss', []),
#                 'best_val_loss': float(min(history.history.get('val_loss', [np.inf]))),
#                 'epochs': EPOCHS,
#                 'epochs_trained': int(epochs_trained),
#                 'height': h_tag,
#                 'use_clustered': USE_CLUSTERED,
#                 'cluster_label': cluster_label,
#                 'mode_code': MODE_CODE,
#                 'train_code': TRAIN_CODE,
#                 'peak_method': PEAK_METHOD,
#                 'peak_q': float(PEAK_Q),
#                 'oversample_factor': int(OVERSAMPLE_FACTOR),
#                 'alpha': float(ALPHA),
#                 'thresh': float(THRESH),
#                 'weight_mode': int(WEIGHT_MODE),
#                 'tau': float(TAU),
#                 'seed': SEED,
#                 'script_name': script_name,
#                 'script_tag': script_tag,
#                 'gm_root_dir': gm_root_dir,
#                 'tha_root_dir': tha_root_dir,
#                 'scaler_X_path': scaler_X_path,
#                 'scaler_Y_path': scaler_Y_path,
#                 'split_idx_path': split_idx_path,
#             }
#             np.save(progress_path, progress_data)
#             print("💾 Progress saved:", progress_path)
# 
#             runinfo = dict(progress_data)
#             runinfo['status'] = 'FINISHED'
#             runinfo['model_dir'] = model_dir
#             runinfo['model_path'] = model_path
#             runinfo['backup_dir'] = backup_dir
#             runinfo['ckpt_dir'] = ckpt_dir
#             runinfo['progress_path'] = progress_path
#             write_runinfo(model_dir, runinfo)
# 
#             try:
#                 plt.figure()
#                 plt.plot(progress_data['train_loss'], label='Train Loss')
#                 plt.plot(progress_data['val_loss'], label='Val Loss')
#                 plt.xlabel('Epoch')
#                 plt.ylabel('Loss')
#                 plt.title(f'{h_tag} - Loss - {run_tag}')
#                 plt.legend()
#                 plt.tight_layout()
# 
#                 loss_plot_path = os.path.join(model_dir, f"loss_curve_{run_tag}.png")
#                 plt.savefig(loss_plot_path, dpi=300)
#                 plt.close()
#                 print("📊 Loss curve saved to:", loss_plot_path)
#             except Exception as e:
#                 print(f"⚠️ Plot error (loss curve): {e}")
# 
#             print(f"✅ {h_tag} | Scenario {run_tag} finished.\n")
# 
#     print("🎉 آموزش همه ارتفاع‌ها در مود per-height تمام شد.")
# 
# =============================================================================







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
# 🧾 Script-tag isolation (avoid overwriting across versions) + SHORT
# ==============================================================
script_name = os.path.splitext(os.path.basename(__file__))[0]

# Short tag from script_name (prefer vXX if exists)
m = re.search(r"(v\d+)", script_name, flags=re.IGNORECASE)
if m:
    script_tag = m.group(1).upper()
else:
    # fallback: keep it short but stable
    script_tag = re.sub(r"[^A-Za-z0-9]+", "", script_name)[:12] or "SCRIPT"

base_model_root = os.path.join(base_model_root, script_tag)
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

cluster_label = ""
if USE_CLUSTERED:
    print("✅ استفاده از داده‌های کلاسترشده (cluster_balanced_global).")
    cluster_label = input(
        "برای اسم پوشهٔ مدل، تعداد کلاسترها (K) را بنویس "
        "(مثلاً 4). اگر خالی بگذاری، 'clustered' نوشته می‌شود: "
    ).strip()
else:
    print("✅ استفاده از داده‌های اصلی بدون کلاسترینگ.")

def compact_float_tag(x, scale=100):
    """
    0.70 -> '70' (default scale=100)
    0.05 -> '05' (if scale=100)
    1.0  -> '100'
    """
    try:
        v = float(x)
    except Exception:
        return str(x)
    n = int(round(v * scale))
    width = 2 if scale == 100 else 0
    return f"{n:0{width}d}" if width > 0 else str(n)

def compact_A_tag(a):
    try:
        v = float(a)
    except Exception:
        return str(a)
    if abs(v - int(v)) < 1e-9:
        return str(int(v))
    s = str(v).replace(".", "p")
    return s

def make_scen_short(EPOCHS, ALPHA, peak_method, PEAK_Q, THRESH, OVERSAMPLE_FACTOR, WEIGHT_MODE, TAU):
    e = int(EPOCHS)
    a = compact_A_tag(ALPHA)
    if peak_method == "quantile":
        q = compact_float_tag(PEAK_Q, scale=100)
        peak = f"Q{q}"
    else:
        r = compact_float_tag(THRESH, scale=100)
        peak = f"R{r}"
    osf = int(OVERSAMPLE_FACTOR)
    m_ = int(WEIGHT_MODE)
    t = compact_float_tag(TAU, scale=100)
    return f"E{e}-A{a}-{peak}-OS{osf}-M{m_}-T{t}"

def get_mode_code():
    if not USE_CLUSTERED:
        return "NC"
    if cluster_label:
        # keep short: CK4
        return f"CK{cluster_label}"
    return "CL"

def write_runinfo(run_dir, info: dict):
    try:
        p = os.path.join(run_dir, "runinfo.txt")
        lines = []
        lines.append("===== RUN INFO =====")
        for k in sorted(info.keys()):
            lines.append(f"{k}: {info[k]}")
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        print(f"⚠️ Could not write runinfo.txt in {run_dir}: {e}")

def plot_loss_from_progress(model_dir, run_tag, title_prefix=None):
    """
    ✅ Only-for-plotting helper:
    Create loss_curve_*.png from progress.npy (works even after resume/skip).
    Does NOT change training behavior.
    """
    progress_path = os.path.join(model_dir, "progress.npy")
    if not os.path.exists(progress_path):
        print(f"⚠️ progress.npy not found for plotting: {progress_path}")
        return False

    try:
        data = np.load(progress_path, allow_pickle=True).item()
    except Exception as e:
        print(f"⚠️ Could not read progress.npy for plotting: {e}")
        return False

    train_loss = data.get("train_loss", []) or []
    val_loss   = data.get("val_loss", []) or []

    if len(train_loss) == 0:
        print(f"⚠️ No train_loss in progress.npy → skip plot: {progress_path}")
        return False

    try:
        plt.figure(figsize=(6, 4))
        plt.plot(train_loss, label="Train Loss", linewidth=2)

        # Plot val only if present and same length
        if len(val_loss) > 0 and len(val_loss) == len(train_loss):
            plt.plot(val_loss, label="Val Loss", linewidth=2)
        else:
            print(f"ℹ️ val_loss missing/mismatch → plotting train only for {run_tag}")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        if title_prefix:
            plt.title(f"{title_prefix} - {run_tag}")
        else:
            plt.title(f"Loss Curve - {run_tag}")

        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        loss_plot_path = os.path.join(model_dir, f"loss_curve_{run_tag}.png")
        plt.savefig(loss_plot_path, dpi=300)
        plt.close()
        print("📊 Loss curve saved to:", loss_plot_path)
        return True
    except Exception as e:
        try:
            plt.close()
        except Exception:
            pass
        print(f"⚠️ Plot error (loss curve): {e}")
        return False

MODE_CODE = get_mode_code()

print("\n📂 Base model root for this code version:")
print("   ", base_model_root)
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
    TRAIN_CODE = "GH"
else:
    print("✅ مود ۲: مدل جداگانه برای هر ارتفاع، بدون فیچر ارتفاع (X = [GM])")
    TRAIN_CODE = "PH"

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
    PAD_val = tf.constant(PAD_value, dtype=tf.float32)

    def weighted_mse_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        mask = tf.cast(tf.not_equal(y_true, PAD_val), tf.float32)
        abs_y = tf.abs(y_true) * mask

        if peak_method == "relative":
            max_abs = tf.reduce_max(abs_y, axis=1, keepdims=True) + 1e-6
            rel = abs_y / max_abs
            if weight_mode == 1:
                peak_strength = tf.cast(rel >= thresh, tf.float32)
            else:
                t = tf.constant(float(tau), dtype=tf.float32)
                peak_strength = tf.sigmoid((rel - thresh) / (t + 1e-6))

        elif peak_method == "quantile":
            abs_y_2d = tf.squeeze(abs_y, axis=-1)
            sorted_vals = tf.sort(abs_y_2d, axis=1)
            T = tf.shape(sorted_vals)[1]
            q_clamped = tf.clip_by_value(tf.cast(q, tf.float32), 0.0, 1.0)
            idx = tf.cast(tf.math.round(q_clamped * tf.cast(T - 1, tf.float32)), tf.int32)
            idx = tf.clip_by_value(idx, 0, T - 1)
            batch_indices = tf.range(tf.shape(sorted_vals)[0])
            gather_idx = tf.stack([batch_indices, tf.fill(tf.shape(batch_indices), idx)], axis=1)
            thr_vals = tf.gather_nd(sorted_vals, gather_idx)
            thr_vals = tf.reshape(thr_vals, (-1, 1, 1))
            if weight_mode == 1:
                peak_strength = tf.cast(abs_y >= thr_vals, tf.float32)
            else:
                t = tf.constant(float(tau), dtype=tf.float32)
                peak_strength = tf.sigmoid((abs_y - thr_vals) / (t + 1e-6))
        else:
            raise ValueError("Invalid peak_method. Use 'relative' or 'quantile'.")

        w = 1.0 + alpha * peak_strength
        w = w * mask

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
            x_feat = np.concatenate([x, h_col], axis=1)

            X_all.append(x_feat)
            Y_all.append(y)

    num_samples = len(X_all)
    if num_samples == 0:
        raise ValueError("❌ هیچ نمونه‌ای برای آموزش پیدا نشد (بعد از فیلتر ارتفاع‌ها/کلاسترها).")

    print(f"\n📦 تعداد کل نمونه‌ها (همه ارتفاع‌ها با هم): {num_samples}")

    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    scaler_Y = MinMaxScaler(feature_range=(-1, 1))

    X_concat = np.concatenate(X_all, axis=0)
    Y_concat = np.concatenate(Y_all, axis=0)

    scaler_X.fit(X_concat)
    scaler_Y.fit(Y_concat)

    X_all = [scaler_X.transform(x) for x in X_all]
    Y_all = [scaler_Y.transform(y) for y in Y_all]

    # short meta dir
    meta_dir = os.path.join(base_model_root, f"_META_{MODE_CODE}-{TRAIN_CODE}")
    os.makedirs(meta_dir, exist_ok=True)

    if is_linear:
        scaler_X_path = os.path.join(meta_dir, "scaler_X_linear.pkl")
        scaler_Y_path = os.path.join(meta_dir, "scaler_Y_linear.pkl")
    else:
        scaler_X_path = os.path.join(meta_dir, "scaler_X_nonlinear.pkl")
        scaler_Y_path = os.path.join(meta_dir, "scaler_Y_nonlinear.pkl")

    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_Y, scaler_Y_path)
    print("\n💾 Global scalers saved:")
    print("  →", scaler_X_path)
    print("  →", scaler_Y_path)

    split_idx_path = os.path.join(meta_dir, f"split_idx_seed{SEED}.npy")

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

        scen_short = make_scen_short(EPOCHS, ALPHA, PEAK_METHOD, PEAK_Q, THRESH, OVERSAMPLE_FACTOR, WEIGHT_MODE, TAU)
        run_tag = f"{MODE_CODE}-{TRAIN_CODE}__{scen_short}"
        model_dir = os.path.join(base_model_root, run_tag)
        os.makedirs(model_dir, exist_ok=True)

        model_path    = os.path.join(model_dir, "LSTM.keras")
        backup_dir    = os.path.join(model_dir, "backup")
        ckpt_dir      = os.path.join(model_dir, "checkpoints")
        progress_path = os.path.join(model_dir, "progress.npy")
        os.makedirs(backup_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        if is_scenario_finished(model_dir, EPOCHS):
            print("" + "=" * 80)
            print(f"⏩ سناریو {run_tag} قبلاً کامل شده است. تلاش برای ساخت loss_curve از روی progress.npy ...")
            print("=" * 80)
            plot_loss_from_progress(model_dir, run_tag, title_prefix=f"All Heights [{heights_str}] - Loss")
            continue

        print("\n" + "=" * 80)
        print(f"🚀 شروع سناریو (Global + Height): {run_tag}")
        print(f"   → EPOCHS={EPOCHS}, ALPHA={ALPHA}, THRESH={THRESH}, WEIGHT_MODE={WEIGHT_MODE}, TAU={TAU}")
        print(f"   → PeakMethod={PEAK_METHOD}, Q={PEAK_Q}, Oversample={OVERSAMPLE_FACTOR}")
        print(f"   → Heights: {heights_str}")
        print("=" * 80)

        # --- NEW: write initial runinfo at scenario start ---
        runinfo_init = {
            'status': 'STARTED',
            'run_tag': run_tag,
            'script_name': script_name,
            'script_tag': script_tag,
            'seed': SEED,
            'mode_code': MODE_CODE,
            'train_code': TRAIN_CODE,
            'use_clustered': USE_CLUSTERED,
            'cluster_label': cluster_label,
            'peak_method': PEAK_METHOD,
            'peak_q': float(PEAK_Q),
            'oversample_factor': int(OVERSAMPLE_FACTOR),
            'alpha': float(ALPHA),
            'thresh': float(THRESH),
            'weight_mode': int(WEIGHT_MODE),
            'tau': float(TAU),
            'epochs_planned': int(EPOCHS),
            'gm_root_dir': gm_root_dir,
            'tha_root_dir': tha_root_dir,
            'model_dir': model_dir,
            'model_path': model_path,
            'backup_dir': backup_dir,
            'ckpt_dir': ckpt_dir,
            'progress_path': progress_path,
            'heights_used': height_tags,
        }
        write_runinfo(model_dir, runinfo_init)

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
            'cluster_label': cluster_label,
            'mode_code': MODE_CODE,
            'train_code': TRAIN_CODE,
            'peak_method': PEAK_METHOD,
            'peak_q': float(PEAK_Q),
            'oversample_factor': int(OVERSAMPLE_FACTOR),
            'alpha': float(ALPHA),
            'thresh': float(THRESH),
            'weight_mode': int(WEIGHT_MODE),
            'tau': float(TAU),
            'seed': SEED,
            'script_name': script_name,
            'script_tag': script_tag,
            'gm_root_dir': gm_root_dir,
            'tha_root_dir': tha_root_dir,
            'scaler_X_path': scaler_X_path,
            'scaler_Y_path': scaler_Y_path,
            'split_idx_path': split_idx_path,
        }
        np.save(progress_path, progress_data)
        print("💾 Progress saved:", progress_path)

        # ✅ Plot-only: ensure loss curve is created from progress.npy
        plot_loss_from_progress(model_dir, run_tag, title_prefix=f"All Heights [{heights_str}] - Loss")


        runinfo = dict(progress_data)
        runinfo['status'] = 'FINISHED'
        runinfo['model_dir'] = model_dir
        runinfo['model_path'] = model_path
        runinfo['backup_dir'] = backup_dir
        runinfo['ckpt_dir'] = ckpt_dir
        runinfo['progress_path'] = progress_path
        write_runinfo(model_dir, runinfo)

        try:
            plt.figure()
            plt.plot(progress_data['train_loss'], label='Train Loss')
            plt.plot(progress_data['val_loss'], label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'All Heights [{heights_str}] - Loss - {run_tag}')
            plt.legend()
            plt.tight_layout()

            loss_plot_path = os.path.join(model_dir, f"loss_curve_{run_tag}.png")
            plt.savefig(loss_plot_path, dpi=300)
            plt.close()
            print("📊 Loss curve saved to:", loss_plot_path)
        except Exception as e:
            print(f"⚠️ Plot error (loss curve): {e}")

        print(f"✅ Scenario {run_tag} finished.\n")

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

        height_meta_dir = os.path.join(base_model_root, f"_META_{MODE_CODE}-{TRAIN_CODE}_{h_tag}")
        os.makedirs(height_meta_dir, exist_ok=True)

        if is_linear:
            scaler_X_path = os.path.join(height_meta_dir, "scaler_X_linear.pkl")
            scaler_Y_path = os.path.join(height_meta_dir, "scaler_Y_linear.pkl")
        else:
            scaler_X_path = os.path.join(height_meta_dir, "scaler_X_nonlinear.pkl")
            scaler_Y_path = os.path.join(height_meta_dir, "scaler_Y_nonlinear.pkl")

        joblib.dump(scaler_X, scaler_X_path)
        joblib.dump(scaler_Y, scaler_Y_path)
        print("💾 height-scalers saved:")
        print("  →", scaler_X_path)
        print("  →", scaler_Y_path)

        split_idx_path = os.path.join(height_meta_dir, f"split_idx_seed{SEED}.npy")

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

            scen_short = make_scen_short(EPOCHS, ALPHA, PEAK_METHOD, PEAK_Q, THRESH, OVERSAMPLE_FACTOR, WEIGHT_MODE, TAU)
            run_tag = f"{MODE_CODE}-{TRAIN_CODE}_{h_tag}__{scen_short}"
            model_dir = os.path.join(base_model_root, run_tag)
            os.makedirs(model_dir, exist_ok=True)

            model_path    = os.path.join(model_dir, "LSTM.keras")
            backup_dir    = os.path.join(model_dir, "backup")
            ckpt_dir      = os.path.join(model_dir, "checkpoints")
            progress_path = os.path.join(model_dir, "progress.npy")

            if is_scenario_finished(model_dir, EPOCHS):
                print("" + "=" * 80)
                print(f"⏩ {h_tag} | سناریو {run_tag} قبلاً کامل شده است. تلاش برای ساخت loss_curve از روی progress.npy ...")
                print("=" * 80)
                plot_loss_from_progress(model_dir, run_tag, title_prefix=f"{h_tag} - Loss")
                continue

            print("\n" + "=" * 80)
            print(f"🚀 {h_tag} | شروع سناریو: {run_tag}")
            print(f"   → EPOCHS={EPOCHS}, ALPHA={ALPHA}, THRESH={THRESH}, WEIGHT_MODE={WEIGHT_MODE}, TAU={TAU}")
            print(f"   → PeakMethod={PEAK_METHOD}, Q={PEAK_Q}, Oversample={OVERSAMPLE_FACTOR}")
            print("=" * 80)

            os.makedirs(ckpt_dir, exist_ok=True)
            os.makedirs(backup_dir, exist_ok=True)

            # --- NEW: write initial runinfo at scenario start ---
            runinfo_init = {
                'status': 'STARTED',
                'run_tag': run_tag,
                'height': h_tag,
                'script_name': script_name,
                'script_tag': script_tag,
                'seed': SEED,
                'mode_code': MODE_CODE,
                'train_code': TRAIN_CODE,
                'use_clustered': USE_CLUSTERED,
                'cluster_label': cluster_label,
                'peak_method': PEAK_METHOD,
                'peak_q': float(PEAK_Q),
                'oversample_factor': int(OVERSAMPLE_FACTOR),
                'alpha': float(ALPHA),
                'thresh': float(THRESH),
                'weight_mode': int(WEIGHT_MODE),
                'tau': float(TAU),
                'epochs_planned': int(EPOCHS),
                'gm_root_dir': gm_root_dir,
                'tha_root_dir': tha_root_dir,
                'model_dir': model_dir,
                'model_path': model_path,
                'backup_dir': backup_dir,
                'ckpt_dir': ckpt_dir,
                'progress_path': progress_path,
            }
            write_runinfo(model_dir, runinfo_init)

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
                'cluster_label': cluster_label,
                'mode_code': MODE_CODE,
                'train_code': TRAIN_CODE,
                'peak_method': PEAK_METHOD,
                'peak_q': float(PEAK_Q),
                'oversample_factor': int(OVERSAMPLE_FACTOR),
                'alpha': float(ALPHA),
                'thresh': float(THRESH),
                'weight_mode': int(WEIGHT_MODE),
                'tau': float(TAU),
                'seed': SEED,
                'script_name': script_name,
                'script_tag': script_tag,
                'gm_root_dir': gm_root_dir,
                'tha_root_dir': tha_root_dir,
                'scaler_X_path': scaler_X_path,
                'scaler_Y_path': scaler_Y_path,
                'split_idx_path': split_idx_path,
            }
            np.save(progress_path, progress_data)
            print("💾 Progress saved:", progress_path)

            # ✅ Plot-only: ensure loss curve is created from progress.npy
            plot_loss_from_progress(model_dir, run_tag, title_prefix=f"{h_tag} - Loss")


            runinfo = dict(progress_data)
            runinfo['status'] = 'FINISHED'
            runinfo['model_dir'] = model_dir
            runinfo['model_path'] = model_path
            runinfo['backup_dir'] = backup_dir
            runinfo['ckpt_dir'] = ckpt_dir
            runinfo['progress_path'] = progress_path
            write_runinfo(model_dir, runinfo)

            try:
                plt.figure()
                plt.plot(progress_data['train_loss'], label='Train Loss')
                plt.plot(progress_data['val_loss'], label='Val Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'{h_tag} - Loss - {run_tag}')
                plt.legend()
                plt.tight_layout()

                loss_plot_path = os.path.join(model_dir, f"loss_curve_{run_tag}.png")
                plt.savefig(loss_plot_path, dpi=300)
                plt.close()
                print("📊 Loss curve saved to:", loss_plot_path)
            except Exception as e:
                print(f"⚠️ Plot error (loss curve): {e}")

            print(f"✅ {h_tag} | Scenario {run_tag} finished.\n")

    print("🎉 آموزش همه ارتفاع‌ها در مود per-height تمام شد.")
