# -*- coding: utf-8 -*-
"""
File name      : step2_run_opensees.py
Author         : pc22
Created on     : Sat Dec 27 13:11:43 2025
Last modified  : Sat Dec 27 13:11:43 2025
------------------------------------------------------------
Purpose:
    Automated Time History Analysis (THA) in OpenSeesPy for
    generating structural response datasets under multiple
    column heights and multiple earthquake records, in both
    training and prediction workflows, and for both linear
    and nonlinear structural models.
------------------------------------------------------------
Description:
    This script is the main runner for dynamic time history
    analysis using OpenSeesPy.

    It supports two execution modes:
      1) train  : runs THA for a set of ground motion records
                 intended to produce training datasets.
      2) predict: runs THA for a separate set of records used
                 for model evaluation / comparison.

    It also supports two structural model types:
      - linear   : executes model_linear.py
      - nonlinear: executes model_nonlinear.py

    For each selected column height (H) and each earthquake
    record (.AT2), the script:
      - sets the height via environment variable (H_COL)
      - creates a dedicated output folder: H*/<record_name>/
      - builds the OpenSees model (linear/nonlinear)
      - defines damping and recorders
      - reads and transforms the ground motion record
      - applies the excitation using UniformExcitation
      - runs dynamic analysis via doDynamicAnalysis()
      - saves recorder outputs to the record-specific folder

    The script is designed for batch processing and is robust
    to failures: if an individual record fails, it is logged
    and execution continues for the remaining records.
------------------------------------------------------------
Inputs:
    - User inputs at runtime:
        * RUN_MODE: train (0) / predict (1)
        * Model type: linear (1) / nonlinear (0)
        * Column height(s): one or multiple values (e.g., 3 4 5)

    - Earthquake records:
        Output/1_IDA_Records_train/*.AT2   (for train mode)
        Output/1_IDA_Records_predict/*.AT2 (for predict mode)

    - Dependent scripts (called via exec/import):
        * model_linear.py / model_nonlinear.py
        * defineDamping.py
        * defineRecorders.py
        * ReadRecord.py  (ReadRecord function)
        * doDynamicAnalysis.py (doDynamicAnalysis function)

    - Key environment variables set by this script:
        * RUN_MODE  : 'train' or 'predict'
        * IS_LINEAR : '1' or '0'
        * H_COL     : column height value for model scripts
------------------------------------------------------------
Outputs:
    - THA results saved per height and per record:
        Output/2_THA_train_linear/H*/<record_name>/
        Output/2_THA_train_nonlinear/H*/<record_name>/
        Output/2_THA_predict_linear/H*/<record_name>/
        Output/2_THA_predict_nonlinear/H*/<record_name>/

      (Exact files depend on defineRecorders.py, typically
       time-history outputs such as displacement/acceleration/
       forces, etc.)

    - Failed record logs per height (text file in run folder):
        failed_records_<mode>_<linear/nonlinear>_<Htag>.txt
------------------------------------------------------------
Changes since previous version:
    - Added unified train/predict switch to route inputs and
      outputs automatically.
    - Added linear/nonlinear switch to execute the correct
      model definition.
    - Added multi-height batch execution (loop over heights).
    - Added per-record subfolder outputs for cleaner dataset
      organization and traceability.
    - Added failure logging without stopping the full run.
------------------------------------------------------------
Impact of changes:
    - Enables systematic dataset generation across multiple
      structural configurations (different heights).
    - Improves reproducibility and experiment traceability by
      enforcing a consistent folder hierarchy.
    - Reduces manual effort and prevents mixing outputs across
      modes (train vs predict) and model types.
    - Makes large batch THA runs more robust (continues after
      individual record failures).
------------------------------------------------------------
Status:
    Stable (Batch processing / Dataset generation)

------------------------------------------------------------
Notes:
    - This script deletes and recreates the output folder for
      each height at the start of execution; previous results
      for that height will be removed.
    - Scaling factor is set as: scaleFac = 10 * 9.81.
    - Ground motions are applied using UniformExcitation with
      a Path timeSeries generated from transformed record data.
    - Recorder definitions fully control which response
      quantities are saved.
"""




# # -*- coding: utf-8 -*-
# import sys, io

# # ✅ جلوگیری از خطای UnicodeEncodeError در محیط‌های مختلف (Spyder، CMD، Run-Codes)
# if hasattr(sys.stdout, "buffer"):
#     sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')

# import os
# import shutil
# from openseespy.opensees import *
# from ReadRecord import ReadRecord
# from analyzeAndAnimate import analyzeAndAnimateTHA
# import vfo.vfo as vfo
# import opsvis as opsv
# from doDynamicAnalysis import doDynamicAnalysis

# # ------------------------------------------------------------
# # 🔧 انتخاب حالت train / predict
# # ------------------------------------------------------------
# choice = input("برای train عدد 0 و برای predict عدد 1 را وارد کن: ").strip()
# if choice == "0":
#     RUN_MODE = "train"
# elif choice == "1":
#     RUN_MODE = "predict"
# else:
#     print("❌ فقط عدد 0 یا 1 مجاز است.")
#     sys.exit(1)

# # ------------------------------------------------------------
# # 🔧 انتخاب مدل خطی / غیرخطی
# # ------------------------------------------------------------
# lin_choice = input("مدل خطی باشد یا غیرخطی؟ برای مدل خطی عدد 1 و برای غیرخطی عدد 0 را وارد کن: ").strip()
# IS_LINEAR = (lin_choice == "1")

# print(f"📌 حالت اجرا: {RUN_MODE} | مدل: {'خطی' if IS_LINEAR else 'غیرخطی'}")

# # همچنین به مدل خبر می‌دهیم (در صورت نیاز)
# os.environ["RUN_MODE"] = RUN_MODE
# os.environ["IS_LINEAR"] = "1" if IS_LINEAR else "0"

# # ------------------------------------------------------------
# # 📏 دریافت ارتفاع ستون (یک یا چند مقدار)
# # ------------------------------------------------------------
# heights_raw = input("ارتفاع ستون‌ها را وارد کن (مثلاً: 3 یا 3 4 5): ").strip()

# if not heights_raw:
#     print("⚠️ هیچ ارتفاعی وارد نشد؛ مقدار پیش‌فرض 3 متر در نظر گرفته می‌شود.")
#     heights = [3.0]
# else:
#     heights = []
#     for token in heights_raw.replace(',', ' ').split():
#         try:
#             h_val = float(token)
#             heights.append(h_val)
#         except ValueError:
#             print(f"⚠️ مقدار «{token}» عدد معتبری نیست و نادیده گرفته می‌شود.")

#     if not heights:
#         print("❌ هیچ ارتفاع معتبری وارد نشد. اجرای برنامه متوقف شد.")
#         sys.exit(1)

# print("📏 ارتفاع‌های انتخاب‌شده برای ستون‌ها:", ", ".join(str(h) for h in heights))

# # ------------------------------------------------------------
# # ⚙️ تنظیمات اولیه تحلیل
# # ------------------------------------------------------------
# scaleFac = 10 * 9.81
# TFree = 0
# dataDirRoot = '../../'

# # ---------------------- پوشهٔ ورودی GM بر اساس حالت ----------------------
# if RUN_MODE == 'train':
#     GMFolder = os.path.join(dataDirRoot, 'Output', '1_IDA_Records_train')
# else:
#     GMFolder = os.path.join(dataDirRoot, 'Output', '1_IDA_Records_predict')

# # ---------------------- پوشهٔ خروجی تحلیل بر اساس حالت و خطی/غیرخطی ----------------------
# if RUN_MODE == 'train':
#     if IS_LINEAR:
#         dataDirBase = os.path.join(dataDirRoot, 'Output', '2_THA_train_linear')
#     else:
#         dataDirBase = os.path.join(dataDirRoot, 'Output', '2_THA_train_nonlinear')
# else:  # predict
#     if IS_LINEAR:
#         dataDirBase = os.path.join(dataDirRoot, 'Output', '2_THA_predict_linear')
#     else:
#         dataDirBase = os.path.join(dataDirRoot, 'Output', '2_THA_predict_nonlinear')

# showAnimationDeform = 0

# print(f"📥 پوشه رکوردهای ورودی: {GMFolder}")
# print(f"📂 پوشه پایه‌ی خروجی THA: {dataDirBase}\n")

# # ------------------------------------------------------------
# # ✳️ پیدا کردن همه رکوردهای .AT2
# # ------------------------------------------------------------
# if not os.path.isdir(GMFolder):
#     raise FileNotFoundError(f"❌ پوشه ورودی رکوردها پیدا نشد: {GMFolder}")

# all_records = [f for f in os.listdir(GMFolder) if f.endswith('.AT2')]
# print(f"🔍 تعداد رکوردهای پیدا شده: {len(all_records)}")

# # ------------------------------------------------------------
# # 🚀 اجرای تحلیل برای هر ارتفاع و هر رکورد
# # ------------------------------------------------------------
# for h_val in heights:
    # تنظیم ارتفاع ستون برای مدل (متغیر محیطی برای model_linear.py و model_nonlinear.py)
#     os.environ["H_COL"] = str(h_val)

    # ساخت نام پوشه برای این ارتفاع
#     if float(h_val).is_integer():
#         h_tag = f"H{int(h_val)}"          # مثال: H3
#     else:
#         h_tag = "H" + str(h_val).replace('.', 'p')   # مثال: H3p5

    # مسیر خروجی مخصوص این ارتفاع
#     dataDirOut = os.path.join(dataDirBase, h_tag)

    # لیست رکوردهای ناموفق مخصوص این ارتفاع
#     failed_records = []

#     print(f"🏗️ شروع تحلیل برای ارتفاع ستون {h_val} متر در پوشه: {dataDirOut}")

    # 🧹 پاکسازی پوشه خروجی مخصوص این ارتفاع
#     if os.path.exists(dataDirOut):
#         print(f"🧹 حذف محتوای قبلی پوشه: {dataDirOut}")
#         shutil.rmtree(dataDirOut)
#     os.makedirs(dataDirOut, exist_ok=True)

    # اجرای تحلیل برای هر رکورد
#     for i, rec_file in enumerate(all_records, start=1):
#         try:
#             record_name = os.path.splitext(rec_file)[0]  # مثلاً: RSN4_..._x1_0
#             inFileName = os.path.join(GMFolder, rec_file)
#             GMPath = os.path.join(GMFolder, record_name + ".txt")

            # مسیر خروجی مخصوص این رکورد (زیرپوشه‌ای داخل پوشه ارتفاع)
#             dataDir_rec = os.path.join(dataDirOut, record_name)
#             os.makedirs(dataDir_rec, exist_ok=True)

#             # ⚡ Recorderها از این متغیر برای مسیر خروجی استفاده می‌کنند
#             dataDir = dataDir_rec

            # 🔹 اجرای مدل (خطی یا غیرخطی) و میرایی
#             if IS_LINEAR:
#                 exec(io.open("model_linear.py", "r", encoding="utf-8").read())
#             else:
#                 exec(open("model_nonlinear.py").read())

#             exec(open("defineDamping.py").read())
#             exec(open("defineRecorders.py").read())

            # 🔹 خواندن رکورد زلزله و اعمال به سری زمانی
#             transformed_path = os.path.join(GMFolder, "transformed")
#             os.makedirs(transformed_path, exist_ok=True)

#             dtInput, numPoints = ReadRecord(inFileName, GMPath)
#             seriesTag = 2
#             timeSeries('Path', seriesTag, '-dt', dtInput, '-filePath', GMPath, '-factor', scaleFac)
#             GMDir = 1
#             pattern('UniformExcitation', 2, GMDir, '-accel', seriesTag)

            # 🔹 اجرای تحلیل دینامیکی
#             Tmax = numPoints * dtInput + TFree
#             dtAnalysis = dtInput

#             mode_str = "train" if RUN_MODE == "train" else "predict"
#             lin_str = "خطی" if IS_LINEAR else "غیرخطی"
#             print(f"⚙️ ارتفاع H = {h_val} m | اجرای رکورد {i}/{len(all_records)}: {record_name}  ({mode_str}, {lin_str})")

#             doDynamicAnalysis(Tmax, dtInput)
#             wipe()

#             print(f"✅ ارتفاع H = {h_val} m | رکورد {record_name} با موفقیت اجرا شد. ({mode_str}, {lin_str})\n")

#         except Exception as e:
#             print(f'❌ ارتفاع H = {h_val} m | خطا در رکورد {rec_file}: {e}')
#             failed_records.append(rec_file)

    # ✏️ ذخیره لیست رکوردهای ناموفق برای این ارتفاع
#     failed_suffix = f"_{RUN_MODE}_{'linear' if IS_LINEAR else 'nonlinear'}_{h_tag}"
#     failed_file_name = f"failed_records{failed_suffix}.txt"

#     with open(failed_file_name, "w", encoding="utf-8") as f:
#         for rec in failed_records:
#             f.write(f"{rec}\n")

#     print(f"📄 لیست رکوردهای ناموفق برای ارتفاع {h_val} متر در فایل {failed_file_name} ذخیره شد.")

# print(f"🏁 اجرای همه رکوردها برای همه ارتفاع‌ها تمام شد. حالت اجرا: {RUN_MODE} | مدل: {'خطی' if IS_LINEAR else 'غیرخطی'}")




# -*- coding: utf-8 -*-
import sys, io

# ✅ جلوگیری از خطای UnicodeEncodeError در محیط‌های مختلف (Spyder، CMD، Run-Codes)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')

import os
import shutil

# ============================================================
# ✅ بخش جدید: سازگاری با جابه‌جایی فایل بین:
#   Model\Time History Analysis (THA)  ↔  Model
# ============================================================

# مسیر فولدر همین فایل (هرجا باشد)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# اگر اسکریپت را به Model منتقل کنی، فایل‌های THA معمولاً اینجا هستند:
THA_DIR = os.path.join(BASE_DIR, "Time History Analysis (THA)")

# برای اینکه importهای زیر (ReadRecord, doDynamicAnalysis, ...) همیشه کار کنند:
# - اگر فایل داخل THA باشد: BASE_DIR همان THA_DIR واقعی است
# - اگر فایل داخل Model باشد: THA_DIR وجود دارد
if os.path.isdir(THA_DIR) and THA_DIR not in sys.path:
    sys.path.insert(0, THA_DIR)

# خود BASE_DIR را هم در sys.path بگذار (ایمن)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

def find_project_root(start_dir: str, max_up: int = 6) -> str:
    """
    ریشه پروژه را پیدا می‌کند: جایی که پوشه Output وجود دارد.
    """
    cur = os.path.abspath(start_dir)
    for _ in range(max_up):
        if os.path.isdir(os.path.join(cur, "Output")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    # اگر پیدا نشد، همان رفتار قبلی را تا حد ممکن حفظ می‌کنیم
    # (اما احتمالاً پروژه ساختار دیگری دارد)
    return os.path.abspath(start_dir)

PROJECT_ROOT = find_project_root(BASE_DIR)

def locate_dep(filename: str) -> str:
    """
    فایل‌های وابسته را در این مسیرها پیدا می‌کند:
      1) کنار همین اسکریپت
      2) داخل Time History Analysis (THA) (اگر اسکریپت داخل Model باشد)
    """
    c1 = os.path.join(BASE_DIR, filename)
    if os.path.exists(c1):
        return c1
    c2 = os.path.join(THA_DIR, filename)
    if os.path.exists(c2):
        return c2
    # اگر پیدا نشد، همان اسم را برمی‌گردانیم تا پیام خطای طبیعی بدهد
    return filename

# ============================================================
# حالا importها (بعد از sys.path تنظیم شده)
# ============================================================
from openseespy.opensees import *
from ReadRecord import ReadRecord
from analyzeAndAnimate import analyzeAndAnimateTHA
import vfo.vfo as vfo
import opsvis as opsv
from doDynamicAnalysis import doDynamicAnalysis

# ------------------------------------------------------------
# 🔧 انتخاب حالت train / predict
# ------------------------------------------------------------
choice = input("برای train عدد 0 و برای predict عدد 1 را وارد کن: ").strip()
if choice == "0":
    RUN_MODE = "train"
elif choice == "1":
    RUN_MODE = "predict"
else:
    print("❌ فقط عدد 0 یا 1 مجاز است.")
    sys.exit(1)

# ------------------------------------------------------------
# 🔧 انتخاب مدل خطی / غیرخطی
# ------------------------------------------------------------
lin_choice = input("مدل خطی باشد یا غیرخطی؟ برای مدل خطی عدد 1 و برای غیرخطی عدد 0 را وارد کن: ").strip()
IS_LINEAR = (lin_choice == "1")

print(f"📌 حالت اجرا: {RUN_MODE} | مدل: {'خطی' if IS_LINEAR else 'غیرخطی'}")

# همچنین به مدل خبر می‌دهیم (در صورت نیاز)
os.environ["RUN_MODE"] = RUN_MODE
os.environ["IS_LINEAR"] = "1" if IS_LINEAR else "0"

# ------------------------------------------------------------
# 📏 دریافت ارتفاع ستون (یک یا چند مقدار)
# ------------------------------------------------------------
heights_raw = input("ارتفاع ستون‌ها را وارد کن (مثلاً: 3 یا 3 4 5): ").strip()

if not heights_raw:
    print("⚠️ هیچ ارتفاعی وارد نشد؛ مقدار پیش‌فرض 3 متر در نظر گرفته می‌شود.")
    heights = [3.0]
else:
    heights = []
    for token in heights_raw.replace(',', ' ').split():
        try:
            h_val = float(token)
            heights.append(h_val)
        except ValueError:
            print(f"⚠️ مقدار «{token}» عدد معتبری نیست و نادیده گرفته می‌شود.")

    if not heights:
        print("❌ هیچ ارتفاع معتبری وارد نشد. اجرای برنامه متوقف شد.")
        sys.exit(1)

print("📏 ارتفاع‌های انتخاب‌شده برای ستون‌ها:", ", ".join(str(h) for h in heights))

# ------------------------------------------------------------
# ⚙️ تنظیمات اولیه تحلیل
# ------------------------------------------------------------
scaleFac = 10 * 9.81
TFree = 0

# ✅ قبلاً: dataDirRoot = '../../'
# ✅ الان: ریشه پروژه را خودکار پیدا می‌کنیم
dataDirRoot = PROJECT_ROOT

# ---------------------- پوشهٔ ورودی GM بر اساس حالت ----------------------
if RUN_MODE == 'train':
    GMFolder = os.path.join(dataDirRoot, 'Output', '1_IDA_Records_train')
else:
    GMFolder = os.path.join(dataDirRoot, 'Output', '1_IDA_Records_predict')

# ---------------------- پوشهٔ خروجی تحلیل بر اساس حالت و خطی/غیرخطی ----------------------
if RUN_MODE == 'train':
    if IS_LINEAR:
        dataDirBase = os.path.join(dataDirRoot, 'Output', '2_THA_train_linear')
    else:
        dataDirBase = os.path.join(dataDirRoot, 'Output', '2_THA_train_nonlinear')
else:  # predict
    if IS_LINEAR:
        dataDirBase = os.path.join(dataDirRoot, 'Output', '2_THA_predict_linear')
    else:
        dataDirBase = os.path.join(dataDirRoot, 'Output', '2_THA_predict_nonlinear')

showAnimationDeform = 0

print(f"📥 پوشه رکوردهای ورودی: {GMFolder}")
print(f"📂 پوشه پایه‌ی خروجی THA: {dataDirBase}\n")

# ------------------------------------------------------------
# ✳️ پیدا کردن همه رکوردهای .AT2
# ------------------------------------------------------------
if not os.path.isdir(GMFolder):
    raise FileNotFoundError(f"❌ پوشه ورودی رکوردها پیدا نشد: {GMFolder}")

all_records = [f for f in os.listdir(GMFolder) if f.endswith('.AT2')]
print(f"🔍 تعداد رکوردهای پیدا شده: {len(all_records)}")

# ------------------------------------------------------------
# 🚀 اجرای تحلیل برای هر ارتفاع و هر رکورد
# ------------------------------------------------------------
for h_val in heights:
    # تنظیم ارتفاع ستون برای مدل (متغیر محیطی برای model_linear.py و model_nonlinear.py)
    os.environ["H_COL"] = str(h_val)

    # ساخت نام پوشه برای این ارتفاع
    if float(h_val).is_integer():
        h_tag = f"H{int(h_val)}"          # مثال: H3
    else:
        h_tag = "H" + str(h_val).replace('.', 'p')   # مثال: H3p5

    # مسیر خروجی مخصوص این ارتفاع
    dataDirOut = os.path.join(dataDirBase, h_tag)

    # لیست رکوردهای ناموفق مخصوص این ارتفاع
    failed_records = []

    print(f"🏗️ شروع تحلیل برای ارتفاع ستون {h_val} متر در پوشه: {dataDirOut}")

    # 🧹 پاکسازی پوشه خروجی مخصوص این ارتفاع
    if os.path.exists(dataDirOut):
        print(f"🧹 حذف محتوای قبلی پوشه: {dataDirOut}")
        shutil.rmtree(dataDirOut)
    os.makedirs(dataDirOut, exist_ok=True)

    # اجرای تحلیل برای هر رکورد
    for i, rec_file in enumerate(all_records, start=1):
        try:
            record_name = os.path.splitext(rec_file)[0]  # مثلاً: RSN4_..._x1_0
            inFileName = os.path.join(GMFolder, rec_file)
            GMPath = os.path.join(GMFolder, record_name + ".txt")

            # مسیر خروجی مخصوص این رکورد (زیرپوشه‌ای داخل پوشه ارتفاع)
            dataDir_rec = os.path.join(dataDirOut, record_name)
            os.makedirs(dataDir_rec, exist_ok=True)

            # ⚡ Recorderها از این متغیر برای مسیر خروجی استفاده می‌کنند
            dataDir = dataDir_rec

            # 🔹 اجرای مدل (خطی یا غیرخطی) و میرایی
            if IS_LINEAR:
                exec(io.open(locate_dep("model_linear.py"), "r", encoding="utf-8").read())
            else:
                exec(open(locate_dep("model_nonlinear.py"), "r", encoding="utf-8").read())

            exec(open(locate_dep("defineDamping.py"), "r", encoding="utf-8").read())
            exec(open(locate_dep("defineRecorders.py"), "r", encoding="utf-8").read())

            # 🔹 خواندن رکورد زلزله و اعمال به سری زمانی
            transformed_path = os.path.join(GMFolder, "transformed")
            os.makedirs(transformed_path, exist_ok=True)

            dtInput, numPoints = ReadRecord(inFileName, GMPath)
            seriesTag = 2
            timeSeries('Path', seriesTag, '-dt', dtInput, '-filePath', GMPath, '-factor', scaleFac)
            GMDir = 1
            pattern('UniformExcitation', 2, GMDir, '-accel', seriesTag)

            # 🔹 اجرای تحلیل دینامیکی
            Tmax = numPoints * dtInput + TFree
            dtAnalysis = dtInput

            mode_str = "train" if RUN_MODE == "train" else "predict"
            lin_str = "خطی" if IS_LINEAR else "غیرخطی"
            print(f"⚙️ ارتفاع H = {h_val} m | اجرای رکورد {i}/{len(all_records)}: {record_name}  ({mode_str}, {lin_str})")

            doDynamicAnalysis(Tmax, dtInput)
            wipe()

            print(f"✅ ارتفاع H = {h_val} m | رکورد {record_name} با موفقیت اجرا شد. ({mode_str}, {lin_str})\n")

        except Exception as e:
            print(f'❌ ارتفاع H = {h_val} m | خطا در رکورد {rec_file}: {e}')
            failed_records.append(rec_file)

    # ✏️ ذخیره لیست رکوردهای ناموفق برای این ارتفاع
    failed_suffix = f"_{RUN_MODE}_{'linear' if IS_LINEAR else 'nonlinear'}_{h_tag}"
    failed_file_name = f"failed_records{failed_suffix}.txt"

    with open(failed_file_name, "w", encoding="utf-8") as f:
        for rec in failed_records:
            f.write(f"{rec}\n")

    print(f"📄 لیست رکوردهای ناموفق برای ارتفاع {h_val} متر در فایل {failed_file_name} ذخیره شد.")

print(f"🏁 اجرای همه رکوردها برای همه ارتفاع‌ها تمام شد. حالت اجرا: {RUN_MODE} | مدل: {'خطی' if IS_LINEAR else 'غیرخطی'}")
