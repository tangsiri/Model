


# -*- coding: utf-8 -*-
"""
File name      : predict_and_compare.py
Author         : pc22
Created on     : Sat Dec 27 13:09:18 2025
Last modified  : Sat Dec 27 13:09:18 2025
------------------------------------------------------------
Purpose:
    Prediction and post-processing of structural response
    time histories using trained LSTM models, with support
    for linear and nonlinear responses, clustered and
    non-clustered training configurations, and multi-height
    global models.

    The script is intended to evaluate trained models under
    unseen earthquake records and to generate quantitative
    and visual performance metrics.
------------------------------------------------------------
Description:
    This script performs time-history prediction of structural
    responses using previously trained LSTM models.

    The script automatically detects available training
    configurations (clustered / non-clustered) and allows
    the user to select the desired model set at runtime.

    Two prediction modes are supported:
      1) Height-specific prediction:
         - Each structural height uses its own trained model.
         - Input features: ground motion only (X = [GM]).
      2) Global multi-height prediction:
         - A single model trained on all heights is used.
         - Structural height is added as an explicit feature
           (X = [GM, H]).

    For each selected height and training scenario, the script:
      - Loads trained models and corresponding scalers
      - Predicts response time histories
      - Applies bias correction
      - Computes performance metrics (RMSE, correlation
        coefficient, peak error)
      - Generates response comparison plots
      - Produces normalized error PDFs
      - Exports a summary of metrics to Excel files

    The script is fully non-interactive in terms of plotting
    (no GUI backend) and is suitable for batch execution.
------------------------------------------------------------
Inputs:
    - Trained LSTM models:
        LSTM.keras
      Stored under:
        Progress_of_LSTM_linear/
        Progress_of_LSTM_nonlinear/
        (clustered or noCluster subfolders)

    - Scalers associated with training:
        scaler_X_*.pkl
        scaler_Y_*.pkl

    - Prediction datasets:
        X_data_H*.npy   (GM inputs)
        Y_data_H*.npy   (reference responses)

    - Raw ground motion records for plotting:
        Output/1_IDA_Records_predict/

    - User inputs at runtime:
        * Linear vs. nonlinear prediction
        * Training configuration (cluster / noCluster)
        * Global vs. per-height model usage
        * Heights to be predicted
        * Training scenarios to evaluate
------------------------------------------------------------
Outputs:
    - Predicted vs. true response plots (PNG) for each:
        height × scenario × earthquake

    - Error probability density functions (PDFs):
        error_pdf_all_scenarios.png

    - Excel summary of performance metrics:
        metrics_summary.xlsx
      Including:
        RMSE, correlation coefficient (CC),
        and peak response error (%)

    - Organized output directory structure:
        Output/predict_linear/
        Output/predict_nonlinear/
        (mirroring training configuration)
------------------------------------------------------------
Changes since previous version:
    - Added automatic detection of clustered and non-clustered
      training directories.
    - Enabled prediction using global multi-height models.
    - Improved output directory safety (no deletion of
      previous prediction results).
    - Added unified Excel reporting for quantitative metrics.
------------------------------------------------------------
Impact of changes:
    - Enables systematic and reproducible comparison of
      different training scenarios and architectures.
    - Simplifies post-processing and result interpretation
      for thesis and publication purposes.
    - Improves robustness when running large-scale
      prediction studies across multiple heights and models.
------------------------------------------------------------
Status:
    Stable (Research / Evaluation phase)

------------------------------------------------------------
Notes:
    - Bias correction is applied to predicted responses
      before error evaluation.
    - All plots are generated using a non-GUI backend and
      saved directly to disk.
    - The script assumes that all required training and
      preprocessing steps have been completed beforehand.
"""


# -*- coding: utf-8 -*-
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")  # ✅ backend بدون GUI
import matplotlib.pyplot as plt
plt.ioff()             # ✅ خاموش کردن حالت تعاملی

import tensorflow as tf
import joblib
import pandas as pd    # برای خروجی xlsx

# ============================================================== #
# 📁 مسیرها + سوال خطی / غیرخطی
# ============================================================== #
base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(base_dir, os.pardir))

choice = input("پیش‌بینی خطی؟ (1=خطی / 0=غیرخطی): ").strip()
is_linear = (choice == "1")

if is_linear:
    print("📌 پیش‌بینی بر اساس مدل خطی انجام می‌شود.")

    # ✅ فقط این خط تغییر کرد: مدل‌ها از Output خوانده می‌شوند
    base_model_root = os.path.join(root_dir, "Output", "Progress_of_LSTM_linear")

    gm_root_dir     = os.path.join(root_dir, "Output", "3_GM_Fixed_predict_linear")
    tha_root_dir    = os.path.join(root_dir, "Output", "3_THA_Fixed_predict_linear")
    gm_raw_dir      = os.path.join(root_dir, "Output", "1_IDA_Records_predict", "zire ham")
    output_base_root = os.path.join(root_dir, "Output", "predict_linear")
else:
    print("📌 پیش‌بینی بر اساس مدل غیرخطی انجام می‌شود.")

    # ✅ فقط این خط تغییر کرد: مدل‌ها از Output خوانده می‌شوند
    base_model_root = os.path.join(root_dir, "Output", "Progress_of_LSTM_nonlinear")

    gm_root_dir     = os.path.join(root_dir, "Output", "3_GM_Fixed_predict_nonlinear")
    tha_root_dir    = os.path.join(root_dir, "Output", "3_THA_Fixed_predict_nonlinear")
    gm_raw_dir      = os.path.join(root_dir, "Output", "1_IDA_Records_predict", "zire ham")
    output_base_root = os.path.join(root_dir, "Output", "predict_nonlinear")

if not os.path.isdir(base_model_root):
    raise FileNotFoundError(f"❌ مسیر پایه مدل‌ها پیدا نشد: {base_model_root}")

# ============================================================== #
# 🎛️ انتخاب پوشهٔ آموزش (cluster / noCluster / قدیمی)
# ============================================================== #
train_config_dirs = [
    d for d in os.listdir(base_model_root)
    if os.path.isdir(os.path.join(base_model_root, d))
    and ("cluster" in d.lower() or "nocluster" in d.lower())
]

if train_config_dirs:
    print("\n📂 پوشه‌های آموزش موجود (cluster / noCluster) زیر:")
    print("   ", base_model_root)
    for i, d in enumerate(sorted(train_config_dirs)):
        print(f"  [{i}] {d}")

    train_config_dirs = sorted(train_config_dirs)

    if len(train_config_dirs) == 1:
        selected_train_dir = train_config_dirs[0]
        print(f"\n✅ فقط یک پوشه پیدا شد، همان استفاده می‌شود: {selected_train_dir}")
    else:
        sel = input(
            "\nنام یا شمارهٔ پوشهٔ آموزش مورد نظر را وارد کن "
            "(مثال: 0 یا clusterK4_allHeights ، خالی = 0): "
        ).strip()

        if sel == "":
            idx = 0
        elif sel.isdigit():
            idx = int(sel)
            if idx < 0 or idx >= len(train_config_dirs):
                print("⚠️ شماره نامعتبر بود، پیش‌فرض 0 استفاده می‌شود.")
                idx = 0
        else:
            if sel in train_config_dirs:
                idx = train_config_dirs.index(sel)
            else:
                print("⚠️ نام پوشه پیدا نشد، پیش‌فرض 0 استفاده می‌شود.")
                idx = 0

        selected_train_dir = train_config_dirs[idx]

    model_root_dir = os.path.join(base_model_root, selected_train_dir)
    output_root    = os.path.join(output_base_root, selected_train_dir)

    print("\n📂 پوشهٔ آموزش انتخاب‌شده:")
    print("   ", model_root_dir)
    print("📂 پوشهٔ خروجی این اجرا:")
    print("   ", output_root)
else:
    model_root_dir = base_model_root
    output_root    = output_base_root

    print("\n📂 هیچ پوشهٔ cluster/noCluster جداگانه‌ای پیدا نشد.")
    print("   از همین مسیر به‌عنوان root مدل استفاده می‌شود:")
    print("   ", model_root_dir)
    print("📂 پوشهٔ خروجی:")
    print("   ", output_root)

# 📂 ریشه‌ی مدل‌های کلی (آموزش همه ارتفاع‌ها با فیچر H)
global_multi_root = os.path.join(model_root_dir, "Global_training_with_height")

# ❓ انتخاب نوع مدل: کلی (multi-height) یا مدل‌های جداگانه برای هر ارتفاع
use_global_model = False
if os.path.isdir(global_multi_root):
    ans = input("از مدل کلی آموزش‌دیده با همه ارتفاع‌ها استفاده شود؟ (1=مدل کلی / 0=مدل‌های جداگانه): ").strip()
    use_global_model = (ans == "1")
    if use_global_model:
        print("✅ استفاده از مدل کلی (Global_training_with_height) برای پیش‌بینی.")
    else:
        print("✅ استفاده از مدل‌های جداگانه‌ی هر ارتفاع.")
else:
    print("⚠️ پوشه Global_training_with_height یافت نشد؛ فقط مدل‌های جداگانه‌ی هر ارتفاع قابل استفاده هستند.")
    use_global_model = False

os.makedirs(output_root, exist_ok=True)

if use_global_model:
    os.makedirs(os.path.join(output_root, "Global_training_with_height"), exist_ok=True)

# ============================================================== #
# 🔍 تعیین ارتفاع‌ها از روی داده‌های پیش‌بینی (GM)
# ============================================================== #
if not os.path.isdir(gm_root_dir):
    raise FileNotFoundError(f"❌ مسیر داده‌های GM پیش‌بینی پیدا نشد: {gm_root_dir}")

height_tags = sorted(
    d for d in os.listdir(gm_root_dir)
    if os.path.isdir(os.path.join(gm_root_dir, d)) and d.startswith("H")
)

if not height_tags:
    raise RuntimeError(f"❌ هیچ پوشه ارتفاع (H*) زیر {gm_root_dir} پیدا نشد.")

print("\n📏 ارتفاع‌های موجود در داده‌های پیش‌بینی (GM):")
for h in height_tags:
    print("  -", h)

def height_value_from_tag(h_tag: str) -> float:
    s = h_tag[1:]
    s = s.replace('p', '.')
    return float(s)

use_all_heights = input("\nبرای همه ارتفاع‌ها پیش‌بینی انجام شود؟ (y/n): ").strip().lower() == "y"

if not use_all_heights:
    print("مثال ورودی:  H3 H4  یا  H3")
    h_items = input("نام ارتفاع‌ها را وارد کن: ").strip().split()
    selected_heights = []
    for h in h_items:
        if h in height_tags:
            selected_heights.append(h)
        else:
            print(f"⚠️ ارتفاع {h} در داده‌های پیش‌بینی پیدا نشد و نادیده گرفته می‌شود.")
    selected_heights = list(dict.fromkeys(selected_heights))
    if not selected_heights:
        raise ValueError("❌ هیچ ارتفاع معتبری انتخاب نشد.")
else:
    selected_heights = height_tags

print("\n✅ ارتفاع‌های انتخاب‌شده برای پیش‌بینی:")
for h in selected_heights:
    print("   →", h)
print()

# ============================================================== #
# 📥 فایل‌های GM خام (برای رسم پاسخ زمانی)
# ============================================================== #
gm_files = sorted(os.listdir(gm_raw_dir))
print(f"📌 تعداد فایل‌های GM خام: {len(gm_files)}\n")

# ==============================================================
# ✅ CHANGE 1: سناریوها فقط یک بار پرسیده شوند (نه برای هر ارتفاع)
#   - منبع لیست سناریوها:
#       * اگر global model: از global_multi_root
#       * اگر per-height: از اولین ارتفاع انتخاب‌شده
# ============================================================== #
if use_global_model:
    scenario_base_dir_global = global_multi_root
else:
    scenario_base_dir_global = os.path.join(model_root_dir, selected_heights[0])

scenario_dirs_global = sorted(
    d for d in os.listdir(scenario_base_dir_global)
    if os.path.isdir(os.path.join(scenario_base_dir_global, d)) and d.startswith("ep")
)

if not scenario_dirs_global:
    raise RuntimeError("❌ هیچ پوشه سناریویی پیدا نشد (برای انتخاب اولیه سناریوها).")

print("\n📂 سناریوهای موجود برای انتخاب (فقط یک بار انتخاب می‌شود):")
for i, nm in enumerate(scenario_dirs_global):
    print(f"  {i}. {nm}")

run_all_scen_global = input("\nهمه سناریوها اجرا شوند؟ (y/n): ").strip().lower() == "y"

if not run_all_scen_global:
    print("\nمثال ورودی شماره‌ها:   0 3 6 9")
    print("مثال ورودی نام‌ها:     ep100_A1.0_T0.50 ep60_A0.5_T0.20")
    print("یا ترکیبی:             0 ep60_A0.5_T0.20 7")

    scen_items = input("شماره‌ها یا نام‌های سناریو را وارد کن: ").strip().split()

    selected_scen_global = []
    invalid_scen = []

    for item in scen_items:
        if item.isdigit():
            idx = int(item)
            if 0 <= idx < len(scenario_dirs_global):
                selected_scen_global.append(scenario_dirs_global[idx])
            else:
                invalid_scen.append(item)
        else:
            if item in scenario_dirs_global:
                selected_scen_global.append(item)
            else:
                invalid_scen.append(item)

    selected_scen_global = list(dict.fromkeys(selected_scen_global))

    if not selected_scen_global:
        raise ValueError("❌ هیچ سناریوی معتبری وارد نشده است.")

    if invalid_scen:
        print("\n⚠️ موارد نامعتبر نادیده گرفته شدند:")
        for itm in invalid_scen:
            print("  -", itm)
else:
    selected_scen_global = scenario_dirs_global

print("\n✅ سناریوهای انتخاب‌شده (برای همه ارتفاع‌ها):")
for s in selected_scen_global:
    print("   -", s)
print()

# ============================================================== #
# 🔁 حلقه روی ارتفاع‌ها
# ============================================================== #
for h_tag in selected_heights:

    print("\n" + "#" * 80)
    print(f"🏗️ شروع پیش‌بینی برای ارتفاع ستون: {h_tag}")
    print("#" * 80)

    x_data_path = os.path.join(gm_root_dir,  h_tag, f"X_data_{h_tag}.npy")
    y_data_path = os.path.join(tha_root_dir, h_tag, f"Y_data_{h_tag}.npy")

    if not os.path.exists(x_data_path):
        print(f"❌ X_data برای {h_tag} پیدا نشد: {x_data_path} → این ارتفاع رد می‌شود.\n")
        continue
    if not os.path.exists(y_data_path):
        print(f"❌ Y_data برای {h_tag} پیدا نشد: {y_data_path} → این ارتفاع رد می‌شود.\n")
        continue

    # اسکیلرها
    if use_global_model:
        if is_linear:
            scaler_x_path = os.path.join(global_multi_root, "scaler_X_linear.pkl")
            scaler_y_path = os.path.join(global_multi_root, "scaler_Y_linear.pkl")
        else:
            scaler_x_path = os.path.join(global_multi_root, "scaler_X_nonlinear.pkl")
            scaler_y_path = os.path.join(global_multi_root, "scaler_Y_nonlinear.pkl")
    else:
        if is_linear:
            scaler_x_path = os.path.join(model_root_dir, h_tag, "scaler_X_linear.pkl")
            scaler_y_path = os.path.join(model_root_dir, h_tag, "scaler_Y_linear.pkl")
        else:
            scaler_x_path = os.path.join(model_root_dir, h_tag, "scaler_X_nonlinear.pkl")
            scaler_y_path = os.path.join(model_root_dir, h_tag, "scaler_Y_nonlinear.pkl")

    if not os.path.exists(scaler_x_path) or not os.path.exists(scaler_y_path):
        print(f"❌ اسکیلرهای لازم برای {h_tag} پیدا نشدند → این ارتفاع رد می‌شود.\n")
        continue

    if use_global_model:
        output_h_dir = os.path.join(output_root, "Global_training_with_height", h_tag)
    else:
        output_h_dir = os.path.join(output_root, h_tag)
    os.makedirs(output_h_dir, exist_ok=True)

    print("🔄 Loading scalers for", h_tag)
    scaler_X = joblib.load(scaler_x_path)
    scaler_Y = joblib.load(scaler_y_path)
    print("✅ Scalers loaded.\n")

    X_data = np.load(x_data_path, allow_pickle=True).item()
    Y_data = np.load(y_data_path, allow_pickle=True).item()

    keys = sorted(X_data.keys())

    Y_list = [np.asarray(Y_data[k], dtype=np.float32).reshape(-1, 1) for k in keys]

    if use_global_model:
        h_val = np.float32(height_value_from_tag(h_tag))
        X_list = []
        for k in keys:
            x_gm = np.asarray(X_data[k], dtype=np.float32).reshape(-1, 1)
            T = x_gm.shape[0]
            h_col = np.full((T, 1), h_val, dtype=np.float32)
            x_feat = np.concatenate([x_gm, h_col], axis=1)
            X_list.append(x_feat)
    else:
        X_list = [np.asarray(X_data[k], dtype=np.float32).reshape(-1, 1) for k in keys]

    X_scaled_list = [scaler_X.transform(x) for x in X_list]

    num_to_plot = min(len(keys), len(gm_files))

    print(f"📌 {h_tag} → تعداد رکوردهای پاسخ: {len(keys)}")
    print(f"📌 {h_tag} → تعداد نمودار برای هر سناریو: {num_to_plot}\n")

    # ============================================================== #
    # ✅ CHANGE 2: محور قائم یکسان برای همه سناریوها (در همین ارتفاع)
    #   راهکار: برای این ارتفاع، ابتدا همه سناریوهای انتخابی را پیش‌بینی می‌گیریم،
    #   min/max را از True و Pred به‌صورت مشترک حساب می‌کنیم،
    #   بعد نمودارها را با ylim ثابت ذخیره می‌کنیم.
    # ============================================================== #

    # مسیر پایه سناریوها برای این ارتفاع
    if use_global_model:
        scenario_base_dir = global_multi_root
    else:
        scenario_base_dir = os.path.join(model_root_dir, h_tag)

    # سناریوهای قابل اجرا برای همین ارتفاع (ممکن است بعضی سناریوها موجود نباشند)
    available_scen_for_height = sorted(
        d for d in os.listdir(scenario_base_dir)
        if os.path.isdir(os.path.join(scenario_base_dir, d)) and d.startswith("ep")
    )

    # فیلتر: فقط سناریوهایی که کاربر انتخاب کرده و واقعاً اینجا موجودند
    selected_scen = [s for s in selected_scen_global if s in available_scen_for_height]

    if not selected_scen:
        print(f"❌ هیچ‌کدام از سناریوهای انتخاب‌شده برای {h_tag} در مسیر {scenario_base_dir} وجود ندارند. این ارتفاع رد می‌شود.\n")
        continue

    print("✅ سناریوهای قابل اجرا برای", h_tag, ":")
    for s in selected_scen:
        print("   -", s)
    print()

    # ---------------------------------------------------------- #
    # 📝 اکسل + دیکشنری خطا
    # ---------------------------------------------------------- #
    excel_rows = []
    excel_columns = [
        "Scenario", "Earthquake", "Epochs", "Alpha", "Thresh",
        "RMSE", "CC", "PeakErr"
    ]
    scenario_errors = {}

    # ---------------------------------------------------------- #
    # ✅ PASS 1: اجرای پیش‌بینی همه سناریوها + ذخیره نتایج و min/max
    # ---------------------------------------------------------- #
    results_by_scen = {}   # scen -> list of dict per record
    global_ymin = +np.inf
    global_ymax = -np.inf

    num_local = min(num_to_plot, len(Y_list), len(X_scaled_list), len(gm_files))

    for scen_name in selected_scen:

        if use_global_model:
            model_dir = os.path.join(global_multi_root, scen_name)
        else:
            model_dir = os.path.join(scenario_base_dir, scen_name)

        model_path = os.path.join(model_dir, "LSTM.keras")
        if not os.path.exists(model_path):
            print(f"⚠️ مدل یافت نشد: {model_path} → این سناریو برای {h_tag} رد می‌شود.")
            continue

        # parse سناریو
        try:
            parts = scen_name.split("_")
            epochs_val = int(parts[0].replace("ep", ""))
            alpha_val  = float(parts[1].replace("A", ""))
            thresh_val = float(parts[2].replace("T", ""))
        except Exception:
            epochs_val = alpha_val = thresh_val = None

        model = tf.keras.models.load_model(model_path, compile=False)

        # پیش‌بینی
        Y_pred_list = []
        for x_sc in X_scaled_list:
            pred = model.predict(x_sc[np.newaxis, ...], verbose=0)[0]
            Y_pred_list.append(pred)

        Y_pred_list = [scaler_Y.inverse_transform(y) for y in Y_pred_list]
        Y_true_list = [y.astype(np.float32) for y in Y_list]

        # Bias correction (همان منطق خودت)
        Y_true_concat = np.concatenate(Y_true_list, axis=0)
        Y_pred_concat = np.concatenate(Y_pred_list, axis=0)
        bias = np.mean(Y_true_concat) - np.mean(Y_pred_concat)
        Y_pred_bc_list = [y + bias for y in Y_pred_list]

        # ذخیره رکورد-به-رکورد + آپدیت ymin/ymax
        per_records = []
        for i in range(num_local):
            gm_file = gm_files[i]
            gm_name = os.path.splitext(gm_file)[0]

            y_true = Y_true_list[i].flatten()
            y_pred = Y_pred_bc_list[i].flatten()

            # هم‌طول‌سازی با GM خام (همان منطق خودت)
            gm_path = os.path.join(gm_raw_dir, gm_file)
            gm_raw = np.loadtxt(gm_path)

            L = min(len(gm_raw), len(y_true), len(y_pred))
            y_true = y_true[:L]
            y_pred = y_pred[:L]

            # update global y-limits for THIS HEIGHT
            local_min = float(min(np.min(y_true), np.min(y_pred)))
            local_max = float(max(np.max(y_true), np.max(y_pred)))
            global_ymin = min(global_ymin, local_min)
            global_ymax = max(global_ymax, local_max)

            # metrics (برای اکسل)
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            cc   = np.corrcoef(y_true, y_pred)[0, 1]
            peak = (
                np.abs(np.max(np.abs(y_pred)) - np.max(np.abs(y_true)))
                / (np.max(np.abs(y_true)) + 1e-12)
                * 100.0
            )

            peak_true = np.max(np.abs(y_true)) + 1e-12
            norm_err_series = (y_pred - y_true) / peak_true * 100.0

            per_records.append({
                "i": i,
                "gm_name": gm_name,
                "y_true": y_true,
                "y_pred": y_pred,
                "rmse": rmse,
                "cc": cc,
                "peak": peak,
                "norm_err_series": norm_err_series
            })

            excel_rows.append({
                "Scenario": scen_name,
                "Earthquake": gm_name,
                "Epochs": epochs_val,
                "Alpha": alpha_val,
                "Thresh": thresh_val,
                "RMSE": rmse,
                "CC": cc,
                "PeakErr": peak
            })

        # خطاها برای PDF
        all_norm_errors = np.concatenate([r["norm_err_series"] for r in per_records], axis=0)
        if all_norm_errors.size > 0:
            scenario_errors[scen_name] = all_norm_errors.astype(np.float32)

        results_by_scen[scen_name] = {
            "epochs": epochs_val,
            "alpha": alpha_val,
            "thresh": thresh_val,
            "records": per_records
        }

        print(f"✅ {h_tag} | آماده‌سازی داده‌ها تمام شد: {scen_name}")

    if not results_by_scen:
        print(f"❌ هیچ سناریویی برای {h_tag} قابل اجرا نبود. این ارتفاع رد می‌شود.\n")
        continue

    # حاشیه کوچک برای خوانایی
    pad = 0.05 * (global_ymax - global_ymin + 1e-12)
    global_ymin -= pad
    global_ymax += pad

    print(f"\n📌 {h_tag} | y-limits مشترک برای همه سناریوها:")
    print(f"    ymin={global_ymin:.6g} , ymax={global_ymax:.6g}\n")

    # ---------------------------------------------------------- #
    # ✅ PASS 2: رسم و ذخیره نمودارها با ylim مشترک
    # ---------------------------------------------------------- #
    for scen_name, payload in results_by_scen.items():

        print("\n" + "=" * 80)
        print(f"🚀 {h_tag} | رسم و ذخیره سناریو با ylim ثابت: {scen_name}")
        print("=" * 80)

        scenario_output_dir = os.path.join(output_h_dir, scen_name)
        os.makedirs(scenario_output_dir, exist_ok=True)

        epochs_val = payload["epochs"]
        alpha_val  = payload["alpha"]
        thresh_val = payload["thresh"]

        print(f"📌 Epochs={epochs_val}, Alpha={alpha_val}, Thresh={thresh_val}")

        for rec in payload["records"]:
            i = rec["i"]
            gm_name = rec["gm_name"]
            y_true = rec["y_true"]
            y_pred = rec["y_pred"]
            rmse = rec["rmse"]
            cc = rec["cc"]
            peak = rec["peak"]

            plt.figure(figsize=(12, 6))
            plt.plot(y_true, color="black", linewidth=0.4, label="True")
            plt.plot(y_pred, color="blue",  linewidth=0.4, label="Predicted")

            # ✅ محور قائم مشترک برای همه سناریوهای همین ارتفاع
            plt.ylim(global_ymin, global_ymax)

            txt = f"{h_tag} - {scen_name}\nRMSE={rmse:.4f}  CC={cc:.4f}  PeakErr={peak:.2f}%"
            plt.text(
                0.98, 0.02, txt,
                transform=plt.gca().transAxes,
                ha="right",
                bbox=dict(facecolor='white', alpha=0.6)
            )

            plt.xlabel("Time step")
            plt.ylabel("Response")
            plt.grid(True)
            plt.legend()

            save_path = os.path.join(scenario_output_dir, f"{i:03d}_{gm_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"✔ {h_tag} | {scen_name} | ذخیره شد: {save_path}")

        print(f"✅ {h_tag} | پایان سناریو: {scen_name}")

    # ---------------------------------------------------------- #
    # 📈 PDF خطا برای همه سناریوهای این ارتفاع
    # ---------------------------------------------------------- #
    if scenario_errors:
        all_vals = np.concatenate(list(scenario_errors.values()), axis=0)
        xmin = np.percentile(all_vals, 1)
        xmax = np.percentile(all_vals, 99)
        dx = (xmax - xmin) * 0.1
        xmin -= dx
        xmax += dx

        num_bins = 80
        bins = np.linspace(xmin, xmax, num_bins + 1)

        plt.rcParams['axes.prop_cycle'] = plt.cycler(
            color=['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9']
        )

        plt.figure(figsize=(9, 5))
        linestyles = ['-', '--', '-.', ':']

        for idx, (scen_name, err_arr) in enumerate(scenario_errors.items()):
            hist, bin_edges = np.histogram(err_arr, bins=bins, density=True)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            plt.plot(
                bin_centers,
                hist,
                linewidth=0.9,
                linestyle=linestyles[idx % len(linestyles)],
                label=scen_name
            )

        plt.axvline(+10.0, linestyle="--", linewidth=0.9, color="k", label="+/-10% Error")
        plt.axvline(-10.0, linestyle="--", linewidth=0.9, color="k")

        plt.xlabel("Normalized error (%)")
        plt.ylabel("PDF")
        plt.title(f"{h_tag} - Normalized Error PDF - All Scenarios")
        plt.grid(True, alpha=0.4)
        plt.legend(fontsize=7)

        pdf_all_path = os.path.join(output_h_dir, "error_pdf_all_scenarios.png")
        plt.savefig(pdf_all_path, dpi=300, bbox_inches="tight")
        plt.close()

        print("\n📈 نمودار PDF همه سناریوهای این ارتفاع ذخیره شد:")
        print("   →", os.path.abspath(pdf_all_path))

    # ---------------------------------------------------------- #
    # 📊 ذخیره اکسل برای این ارتفاع
    # ---------------------------------------------------------- #
    df = pd.DataFrame(excel_rows, columns=excel_columns)
    if not df.empty:
        df = df.sort_values(by="CC", ascending=False)

    excel_path = os.path.join(output_h_dir, "metrics_summary.xlsx")
    df.to_excel(excel_path, index=False)

    print("\n📑 فایل اکسل معیارها برای این ارتفاع ذخیره شد:")
    print("   →", os.path.abspath(excel_path))
    print(f"\n🎯 پیش‌بینی برای ارتفاع {h_tag} تمام شد.\n")

print("🎉 تمام پیش‌بینی‌های همه ارتفاع‌های انتخاب‌شده و سناریوها انجام شد.")













