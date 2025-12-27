# -*- coding: utf-8 -*-
"""
File name      :step1_generate_records.py
Author         : pc22
Created on     : Wed Dec 24 09:21:04 2025
Last modified  : Wed Dec 24 09:21:04 2025

------------------------------------------------------------
Purpose:
------------------------------------------------------------
To preprocess PEER *.AT2 ground-motion records for LSTM-based seismic
response modeling by trimming each record based on a user-defined
percentage of PGA (Peak Ground Acceleration), optionally adding
pre/post padding in seconds, and (for training mode) generating
multiple scaled versions of each record to augment the training set.

------------------------------------------------------------
Description:
------------------------------------------------------------
This script:
1) Selects the working mode (train / predict).
2) Reads PEER *.AT2 files (excluding files containing "-UP").
3) Tries to extract DT from the 4th header line (if present).
4) Trims the signal to the significant shaking window defined by:
      |a(t)| >= (percent/100) * PGA
   and extends the window by user-specified padding before/after
   (in seconds), when DT is available.
5) In train mode only:
   - Asks for a multiplier range (min, max, step) and generates a list
     of scaling factors to create additional augmented records.
   In predict mode:
   - Uses only multiplier = 1.0 (no augmentation).
6) Saves outputs into the original project folder structure:
   - Writes scaled trimmed AT2 files into:
       Output/1_IDA_Records_{train|predict}/
   - Writes ML-ready text files (one value per line) into:
       Output/1_IDA_Records_{train|predict}/zire ham/
     using the suffix "_for_ML.txt".

------------------------------------------------------------
Inputs:
------------------------------------------------------------
- Input folders (selected by mode):
    * train  : Input/PEER_train/*.AT2
    * predict: Input/PEER_Predict/*.AT2
  (Files containing "-UP" are ignored.)
- User inputs at runtime:
    * mode (0=train, 1=predict)
    * percent threshold of PGA (e.g., 5)
    * pad_before_sec, pad_after_sec (seconds)
    * (train only) multiplier min, max, step

------------------------------------------------------------
Outputs:
------------------------------------------------------------
For train:
- Output/1_IDA_Records_train/<record>_x<mult>.AT2
- Output/1_IDA_Records_train/zire ham/<record>_x<mult>_for_ML.txt

For predict:
- Output/1_IDA_Records_predict/<record>_x1.AT2
- Output/1_IDA_Records_predict/zire ham/<record>_x1_for_ML.txt

Note: Output folders are fully cleaned (deleted and recreated) each run.

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
- DT is parsed only if "DT=" exists in the 4th header line; otherwise
  padding is set to 0 samples (no extension) and trimming uses only
  the threshold indices.
- Trimming is based on a PGA-relative threshold and may remove long
  low-amplitude tails; choose percent/padding carefully depending on
  your downstream analysis requirements.
- This script prepares the inputs for the later steps (e.g., THA runs
  and fixed X/Y dataset generation).
"""

import sys, io

# =====================================
#  رفع مشکل UnicodeEncodeError
# =====================================
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')

import numpy as np
import os
import shutil
from typing import Optional   # ✅ برای سازگاری با Python 3.9

# =====================================
#  توابع مسیرها
# =====================================
def build_paths(base_dir: str, mode: str):
    """بر اساس mode مسیرهای ورودی/خروجی را می‌سازد."""
    mode = mode.lower().strip()
    if mode not in {"train", "predict"}:
        raise ValueError("mode must be 'train' یا 'predict باشد.")

    data_dir = os.path.abspath(base_dir)

    if mode == "predict":
        input_folder  = os.path.join(data_dir, 'Input', 'PEER_Predict')
        output_folder = os.path.join(data_dir, 'Output', '1_IDA_Records_predict')
    else:
        input_folder  = os.path.join(data_dir, 'Input', 'PEER_train')
        output_folder = os.path.join(data_dir, 'Output', '1_IDA_Records_train')

    ml_output_folder = os.path.join(output_folder, 'zire ham')

    return input_folder, output_folder, ml_output_folder


def ensure_clean_dir(path: str):
    """پاک‌سازی کامل پوشه خروجی."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


# =====================================
#  خواندن فایل AT2
# =====================================
def read_peer_at2(filepath: str):
    with open(filepath, 'r') as f:
        header = [next(f) for _ in range(4)]
        data = []
        for line in f:
            parts = line.strip().split()
            if parts:
                data.extend(float(x) for x in parts)

    # استخراج dt اگر داخل خط 4 باشد
    dt = None
    line4 = header[3]
    if "DT=" in line4:
        try:
            dt_str = line4.split("DT=")[1].split()[0]
            dt = float(dt_str)
        except Exception:
            dt = None

    return header, np.asarray(data, dtype=np.float64), dt


# =====================================
#  نوشتن AT2
# =====================================
def write_at2(filepath: str, header, data: np.ndarray, per_line: int = 8):
    with open(filepath, 'w') as f:
        f.writelines(header)
        for i in range(0, len(data), per_line):
            chunk = data[i:i+per_line]
            line = ' '.join(f"{val:12.7E}" for val in chunk)
            f.write(line + "\n")


# =====================================
#  Trim بر اساس درصد دلخواه PGA
# =====================================
def trim_by_percent_pga(data: np.ndarray,
                        dt: Optional[float],       # ✅ سازگار با Python 3.9
                        percent: float,
                        pad_before_sec: float,
                        pad_after_sec: float) -> np.ndarray:

    abs_data = np.abs(data)
    pga = abs_data.max()

    if pga == 0:
        print("⚠ PGA = 0 → رکورد بدون تغییر.")
        return data

    threshold = (percent / 100.0) * pga
    idx = np.where(abs_data >= threshold)[0]

    if len(idx) == 0:
        print(f"⚠ هیچ نقطه‌ای ≥ {percent}% PGA نبود → رکورد بدون تغییر.")
        return data

    first_idx = idx[0]
    last_idx  = idx[-1]

    if dt is not None and dt > 0:
        pad_before = int(round(pad_before_sec / dt))
        pad_after  = int(round(pad_after_sec  / dt))
    else:
        pad_before = pad_after = 0
        print("⚠ DT یافت نشد → پدینگ 0 در نظر گرفته شد.")

    start = max(first_idx - pad_before, 0)
    end   = min(last_idx + pad_after, len(data) - 1)

    print(f"  ▪ PGA = {pga:.5g}")
    print(f"  ▪ Threshold = {threshold:.5g}  ({percent}%)")
    print(f"  ▪ Start = {start}, End = {end}")
    print(f"  ▪ Length before = {len(data)}, after = {end - start + 1}")

    return data[start:end+1]


# =====================================
#  تابع اصلی
# =====================================
def main():
    # Mode
    user_choice = input("برای train عدد 0 و برای predict عدد 1 را وارد کن: ").strip()
    if user_choice == "0":
        mode = "train"
    elif user_choice == "1":
        mode = "predict"
    else:
        print("❌ فقط 0 یا 1.")
        return

    # درصد آستانه
    percent_str = input("چند درصد PGA به عنوان حد آستانه در نظر گرفته شود؟ (مثلاً 5): ").strip()
    try:
        percent = float(percent_str)
        if percent <= 0 or percent >= 100:
            raise ValueError
    except Exception:
        print("❌ مقدار درصد معتبر نیست.")
        return

    # پدینگ قبل
    pad_before_str = input("چند ثانیه قبل از شروع زلزله نگه داشته شود؟ (مثلاً 2): ").strip()
    try:
        pad_before_sec = float(pad_before_str)
    except Exception:
        print("❌ مقدار پدینگ قبل معتبر نیست.")
        return

    # پدینگ بعد
    pad_after_str = input("چند ثانیه بعد از پایان زلزله نگه داشته شود؟ (مثلاً 2): ").strip()
    try:
        pad_after_sec = float(pad_after_str)
    except Exception:
        print("❌ مقدار پدینگ بعد معتبر نیست.")
        return

    # ✅ تنظیم ضرایب ضرب فقط برای حالت train
    if mode == "train":
        print("\n📈 تنظیم ضرایب ضرب برای افزایش تعداد رکوردهای آموزشی:")
        min_str = input("کمترین ضریب (مثلاً 0.1): ").strip()
        max_str = input("بیشترین ضریب (مثلاً 10): ").strip()
        step_str = input("گام تغییر ضریب (مثلاً 0.1): ").strip()

        try:
            min_m = float(min_str)
            max_m = float(max_str)
            step  = float(step_str)

            if step <= 0 or max_m < min_m:
                raise ValueError
        except Exception:
            print("❌ مقادیر ضرایب/گام معتبر نیست.")
            return

        # ساخت لیست ضرایب از min_m تا max_m با گام step
        n_steps = int(np.floor((max_m - min_m) / step)) + 1
        multipliers = [min_m + i * step for i in range(n_steps)]

        print(f"\n🔢 ضرایب مورد استفاده برای train:")
        print(", ".join(f"{m:.4g}" for m in multipliers))
    else:
        # برای predict فقط همان رکورد اصلی (بدون ضرب) استفاده می‌شود
        multipliers = [1.0]
        print("\n📌 MODE = predict → فقط ضریب 1.0 استفاده می‌شود (رکوردها ضرب نمی‌شوند).")

    # base_dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir   = os.path.abspath(os.path.join(script_dir, ".."))

    # مسیرها
    input_folder, output_folder, ml_output_folder = build_paths(base_dir, mode)

    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"❌ پوشه ورودی یافت نشد: {input_folder}")

    ensure_clean_dir(output_folder)
    ensure_clean_dir(ml_output_folder)

    print(f"\n📂 MODE = {mode}")
    print(f"📥 Input  = {input_folder}")
    print(f"📤 Output = {output_folder}")
    print(f"🧪 ML OUT = {ml_output_folder}")
    print(f"🔻 Threshold = {percent}% PGA")
    print(f"🔻 Pad Before = {pad_before_sec} sec   |   Pad After = {pad_after_sec} sec\n")

    # فایل‌ها
    files = [f for f in os.listdir(input_folder) if f.endswith(".AT2") and ("-UP" not in f)]
    if not files:
        print("⚠ هیچ فایل AT2 بدون -UP یافت نشد.")
        return

    # پردازش رکوردها
    for input_file in files:
        print(f"\n▶ پردازش: {input_file}")

        input_path = os.path.join(input_folder, input_file)
        base_name  = os.path.splitext(input_file)[0]

        header, data, dt = read_peer_at2(input_path)

        # Trim با درصد و پدینگ دلخواه
        trimmed = trim_by_percent_pga(data, dt, percent,
                                      pad_before_sec=pad_before_sec,
                                      pad_after_sec=pad_after_sec)

        # خروجی‌ها برای هر ضریب
        for m in multipliers:
            modified = trimmed * m
            safe_m = f"x{m:.3f}".rstrip("0").rstrip(".").replace(".", "_")

            out_at2 = os.path.join(output_folder, f"{base_name}_{safe_m}.AT2")
            out_ml  = os.path.join(ml_output_folder, f"{base_name}_{safe_m}_for_ML.txt")

            np.savetxt(out_ml, modified, fmt="%.7f")
            write_at2(out_at2, header, modified)

            print(f"  ✓ ذخیره شد → {base_name}_{safe_m}")

    print("\n🎉 پردازش همه رکوردها با موفقیت تمام شد.")


# =====================================
#  اجرای مستقیم
# =====================================
if __name__ == "__main__":
    main()
