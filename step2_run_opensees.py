"""
File name      : step2_run_opensees.py
Author         : pc22
Created on     : Sat Dec 27 13:11:43 2025
Last modified  : Wed Jan 08 2026
------------------------------------------------------------
Purpose:
    Parallelized, automated, and resumable Time History Analysis (THA)
    in OpenSeesPy for generating structural response datasets under
    multiple column heights and multiple earthquake records, in both
    training and prediction workflows, for both linear and nonlinear
    structural models, with an online global ETA for the entire batch.

    New in this version:
      - Multi-core / parallel execution across records using a
        process-based worker pool (ProcessPoolExecutor) to utilize
        multiple CPU cores and reduce total batch runtime.
------------------------------------------------------------
Description:
    This script is the main batch runner for dynamic time history
    analysis using OpenSeesPy, with built-in support for safe
    interruption/resume and progress time estimation.

    It supports two execution modes:
      1) train  : runs THA for a set of ground motion records intended
                 to produce training datasets.
      2) predict: runs THA for a separate set of records used for model
                 evaluation / comparison.

    It also supports two structural model types:
      - linear   : executes model_linear.py
      - nonlinear: executes model_nonlinear.py

    For each selected column height (H) and each earthquake record
    (.AT2), the script:
      - sets the height via environment variable (H_COL)
      - creates a dedicated output folder: H*/<record_name>/
      - builds the OpenSees model (linear/nonlinear)
      - defines damping and recorders
      - reads/transforms the ground motion record
      - applies the excitation using UniformExcitation
      - runs dynamic analysis via doDynamicAnalysis()
      - saves recorder outputs to the record-specific folder

    Parallel execution (multi-core):
      - Instead of running records sequentially, remaining (not-DONE)
        record-runs are dispatched to multiple independent processes
        using ProcessPoolExecutor.
      - Each process runs one (H, record) job in isolation (separate
        OpenSees state), improving stability and CPU utilization.

    Resume mechanism (robust long batch runs):
      - After successful execution of each record, a completion marker
        file (__DONE__.txt) is written to the corresponding record output
        folder.
      - Upon restart (after power loss, crash, or manual stop), records
        that already contain this marker are automatically skipped.
      - Records that were interrupted before completion are safely re-run,
        with a per-record cleanup to prevent mixing partial outputs with
        new results.

    Global ETA (entire project across all selected heights):
      - Before starting the run, the script scans all selected heights
        and counts how many record-runs remain (i.e., do not have
        __DONE__.txt).
      - During execution, it measures per-record wall time and updates an
        Exponential Moving Average (EMA).
      - After each successful record-run, it prints a concise global status:
           "🌍 کل پروژه: OK/Total | باقی‌مانده | ETA(ALL) → finish time"
        representing the estimated time remaining and the approximate
        completion timestamp for the entire batch (all heights).
------------------------------------------------------------
Inputs:
    - User inputs at runtime:
        * RUN_MODE: train (0) / predict (1)
        * Model type: linear (1) / nonlinear (0)
        * Column height(s): one or multiple values (e.g., 3 4 5)
        * Parallel workers: number of processes (optional; suggested default)

    - Earthquake records:
        Output/1_IDA_Records_train/*.AT2   (for train mode)
        Output/1_IDA_Records_predict/*.AT2 (for predict mode)

    - Dependent scripts (called via exec/import):
        * model_linear.py / model_nonlinear.py
        * defineDamping.py
        * defineRecorders.py
        * ReadRecord.py        (ReadRecord function)
        * doDynamicAnalysis.py (doDynamicAnalysis function)

    - Key environment variables set per job:
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

      (Exact files depend on defineRecorders.py, typically time-history
       outputs such as displacement, acceleration, internal forces, etc.)

    - Completion marker per successfully executed record:
        __DONE__.txt  (stored inside each record folder)

    - Failed record logs per height:
        failed_records_<mode>_<linear/nonlinear>_<Htag>.txt

    - Error diagnostics for failed records (optional):
        __ERROR__.txt inside the corresponding record folder

    - Console progress output:
        * Global ETA line after each successful run
        * End-of-run summary (OK/FAIL/SKIP + elapsed time)
------------------------------------------------------------
Changes since previous version:
    - Added parallel multi-process execution across (height, record)
      jobs using ProcessPoolExecutor to increase CPU utilization.
    - Preserved resume-safe execution using per-record completion markers
      (__DONE__.txt) and safe cleanup of partial outputs.
    - Kept global ETA across all selected heights using EMA of observed
      per-record wall time.
    - Added optional user input to select the number of parallel workers.
------------------------------------------------------------
Impact of changes:
    - Significantly reduces total runtime on multi-core CPUs by processing
      multiple records concurrently.
    - Maintains reliability for long batch runs (resume after interruption)
      without losing completed results.
    - Preserves reproducible folder hierarchy and traceability using DONE
      markers and per-record error logs.
------------------------------------------------------------
Status:
    Stable (Parallel batch processing / Resumable dataset generation with global ETA)
------------------------------------------------------------
Notes:
    - Parallel execution is process-based (not threads) to keep OpenSees
      state isolated per job and avoid cross-interference.
    - By default, output folders are preserved to support resume
      functionality. A clean re-run per height can be enforced via the
      runtime prompt (clean=1), which deletes the height folder and
      recomputes that height.
    - Scaling factor is set as: scaleFac = 10 * 9.81.
    - Ground motions are applied using UniformExcitation with a Path
      timeSeries generated from transformed record data.
    - Recorder definitions fully control which response quantities are saved.
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


# =============================================================================
# 
# 
# # -*- coding: utf-8 -*-
# import sys, io
# 
# # ✅ جلوگیری از خطای UnicodeEncodeError در محیط‌های مختلف (Spyder، CMD، Run-Codes)
# if hasattr(sys.stdout, "buffer"):
#     sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
# 
# import os
# import shutil
# 
# # ============================================================
# # ✅ بخش جدید: سازگاری با جابه‌جایی فایل بین:
# #   Model\Time History Analysis (THA)  ↔  Model
# # ============================================================
# 
# # مسیر فولدر همین فایل (هرجا باشد)
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 
# # اگر اسکریپت را به Model منتقل کنی، فایل‌های THA معمولاً اینجا هستند:
# THA_DIR = os.path.join(BASE_DIR, "Time History Analysis (THA)")
# 
# # برای اینکه importهای زیر (ReadRecord, doDynamicAnalysis, ...) همیشه کار کنند:
# # - اگر فایل داخل THA باشد: BASE_DIR همان THA_DIR واقعی است
# # - اگر فایل داخل Model باشد: THA_DIR وجود دارد
# if os.path.isdir(THA_DIR) and THA_DIR not in sys.path:
#     sys.path.insert(0, THA_DIR)
# 
# # خود BASE_DIR را هم در sys.path بگذار (ایمن)
# if BASE_DIR not in sys.path:
#     sys.path.insert(0, BASE_DIR)
# 
# def find_project_root(start_dir: str, max_up: int = 6) -> str:
#     """
#     ریشه پروژه را پیدا می‌کند: جایی که پوشه Output وجود دارد.
#     """
#     cur = os.path.abspath(start_dir)
#     for _ in range(max_up):
#         if os.path.isdir(os.path.join(cur, "Output")):
#             return cur
#         parent = os.path.dirname(cur)
#         if parent == cur:
#             break
#         cur = parent
#     # اگر پیدا نشد، همان رفتار قبلی را تا حد ممکن حفظ می‌کنیم
#     # (اما احتمالاً پروژه ساختار دیگری دارد)
#     return os.path.abspath(start_dir)
# 
# PROJECT_ROOT = find_project_root(BASE_DIR)
# 
# def locate_dep(filename: str) -> str:
#     """
#     فایل‌های وابسته را در این مسیرها پیدا می‌کند:
#       1) کنار همین اسکریپت
#       2) داخل Time History Analysis (THA) (اگر اسکریپت داخل Model باشد)
#     """
#     c1 = os.path.join(BASE_DIR, filename)
#     if os.path.exists(c1):
#         return c1
#     c2 = os.path.join(THA_DIR, filename)
#     if os.path.exists(c2):
#         return c2
#     # اگر پیدا نشد، همان اسم را برمی‌گردانیم تا پیام خطای طبیعی بدهد
#     return filename
# 
# # ============================================================
# # حالا importها (بعد از sys.path تنظیم شده)
# # ============================================================
# from openseespy.opensees import *
# from ReadRecord import ReadRecord
# from analyzeAndAnimate import analyzeAndAnimateTHA
# import vfo.vfo as vfo
# import opsvis as opsv
# from doDynamicAnalysis import doDynamicAnalysis
# 
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
# 
# # ------------------------------------------------------------
# # 🔧 انتخاب مدل خطی / غیرخطی
# # ------------------------------------------------------------
# lin_choice = input("مدل خطی باشد یا غیرخطی؟ برای مدل خطی عدد 1 و برای غیرخطی عدد 0 را وارد کن: ").strip()
# IS_LINEAR = (lin_choice == "1")
# 
# print(f"📌 حالت اجرا: {RUN_MODE} | مدل: {'خطی' if IS_LINEAR else 'غیرخطی'}")
# 
# # همچنین به مدل خبر می‌دهیم (در صورت نیاز)
# os.environ["RUN_MODE"] = RUN_MODE
# os.environ["IS_LINEAR"] = "1" if IS_LINEAR else "0"
# 
# # ------------------------------------------------------------
# # 📏 دریافت ارتفاع ستون (یک یا چند مقدار)
# # ------------------------------------------------------------
# heights_raw = input("ارتفاع ستون‌ها را وارد کن (مثلاً: 3 یا 3 4 5): ").strip()
# 
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
# 
#     if not heights:
#         print("❌ هیچ ارتفاع معتبری وارد نشد. اجرای برنامه متوقف شد.")
#         sys.exit(1)
# 
# print("📏 ارتفاع‌های انتخاب‌شده برای ستون‌ها:", ", ".join(str(h) for h in heights))
# 
# # ------------------------------------------------------------
# # ⚙️ تنظیمات اولیه تحلیل
# # ------------------------------------------------------------
# scaleFac = 10 * 9.81
# TFree = 0
# 
# # ✅ قبلاً: dataDirRoot = '../../'
# # ✅ الان: ریشه پروژه را خودکار پیدا می‌کنیم
# dataDirRoot = PROJECT_ROOT
# 
# # ---------------------- پوشهٔ ورودی GM بر اساس حالت ----------------------
# if RUN_MODE == 'train':
#     GMFolder = os.path.join(dataDirRoot, 'Output', '1_IDA_Records_train')
# else:
#     GMFolder = os.path.join(dataDirRoot, 'Output', '1_IDA_Records_predict')
# 
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
# 
# showAnimationDeform = 0
# 
# print(f"📥 پوشه رکوردهای ورودی: {GMFolder}")
# print(f"📂 پوشه پایه‌ی خروجی THA: {dataDirBase}\n")
# 
# # ------------------------------------------------------------
# # ✳️ پیدا کردن همه رکوردهای .AT2
# # ------------------------------------------------------------
# if not os.path.isdir(GMFolder):
#     raise FileNotFoundError(f"❌ پوشه ورودی رکوردها پیدا نشد: {GMFolder}")
# 
# all_records = [f for f in os.listdir(GMFolder) if f.endswith('.AT2')]
# print(f"🔍 تعداد رکوردهای پیدا شده: {len(all_records)}")
# 
# # ------------------------------------------------------------
# # 🚀 اجرای تحلیل برای هر ارتفاع و هر رکورد
# # ------------------------------------------------------------
# for h_val in heights:
#     # تنظیم ارتفاع ستون برای مدل (متغیر محیطی برای model_linear.py و model_nonlinear.py)
#     os.environ["H_COL"] = str(h_val)
# 
#     # ساخت نام پوشه برای این ارتفاع
#     if float(h_val).is_integer():
#         h_tag = f"H{int(h_val)}"          # مثال: H3
#     else:
#         h_tag = "H" + str(h_val).replace('.', 'p')   # مثال: H3p5
# 
#     # مسیر خروجی مخصوص این ارتفاع
#     dataDirOut = os.path.join(dataDirBase, h_tag)
# 
#     # لیست رکوردهای ناموفق مخصوص این ارتفاع
#     failed_records = []
# 
#     print(f"🏗️ شروع تحلیل برای ارتفاع ستون {h_val} متر در پوشه: {dataDirOut}")
# 
#     # 🧹 پاکسازی پوشه خروجی مخصوص این ارتفاع
#     if os.path.exists(dataDirOut):
#         print(f"🧹 حذف محتوای قبلی پوشه: {dataDirOut}")
#         shutil.rmtree(dataDirOut)
#     os.makedirs(dataDirOut, exist_ok=True)
# 
#     # اجرای تحلیل برای هر رکورد
#     for i, rec_file in enumerate(all_records, start=1):
#         try:
#             record_name = os.path.splitext(rec_file)[0]  # مثلاً: RSN4_..._x1_0
#             inFileName = os.path.join(GMFolder, rec_file)
#             GMPath = os.path.join(GMFolder, record_name + ".txt")
# 
#             # مسیر خروجی مخصوص این رکورد (زیرپوشه‌ای داخل پوشه ارتفاع)
#             dataDir_rec = os.path.join(dataDirOut, record_name)
#             os.makedirs(dataDir_rec, exist_ok=True)
# 
#             # ⚡ Recorderها از این متغیر برای مسیر خروجی استفاده می‌کنند
#             dataDir = dataDir_rec
# 
#             # 🔹 اجرای مدل (خطی یا غیرخطی) و میرایی
#             if IS_LINEAR:
#                 exec(io.open(locate_dep("model_linear.py"), "r", encoding="utf-8").read())
#             else:
#                 exec(open(locate_dep("model_nonlinear.py"), "r", encoding="utf-8").read())
# 
#             exec(open(locate_dep("defineDamping.py"), "r", encoding="utf-8").read())
#             exec(open(locate_dep("defineRecorders.py"), "r", encoding="utf-8").read())
# 
#             # 🔹 خواندن رکورد زلزله و اعمال به سری زمانی
#             transformed_path = os.path.join(GMFolder, "transformed")
#             os.makedirs(transformed_path, exist_ok=True)
# 
#             dtInput, numPoints = ReadRecord(inFileName, GMPath)
#             seriesTag = 2
#             timeSeries('Path', seriesTag, '-dt', dtInput, '-filePath', GMPath, '-factor', scaleFac)
#             GMDir = 1
#             pattern('UniformExcitation', 2, GMDir, '-accel', seriesTag)
# 
#             # 🔹 اجرای تحلیل دینامیکی
#             Tmax = numPoints * dtInput + TFree
#             dtAnalysis = dtInput
# 
#             mode_str = "train" if RUN_MODE == "train" else "predict"
#             lin_str = "خطی" if IS_LINEAR else "غیرخطی"
#             print(f"⚙️ ارتفاع H = {h_val} m | اجرای رکورد {i}/{len(all_records)}: {record_name}  ({mode_str}, {lin_str})")
# 
#             doDynamicAnalysis(Tmax, dtInput)
#             wipe()
# 
#             print(f"✅ ارتفاع H = {h_val} m | رکورد {record_name} با موفقیت اجرا شد. ({mode_str}, {lin_str})\n")
# 
#         except Exception as e:
#             print(f'❌ ارتفاع H = {h_val} m | خطا در رکورد {rec_file}: {e}')
#             failed_records.append(rec_file)
# 
#     # ✏️ ذخیره لیست رکوردهای ناموفق برای این ارتفاع
#     failed_suffix = f"_{RUN_MODE}_{'linear' if IS_LINEAR else 'nonlinear'}_{h_tag}"
#     failed_file_name = f"failed_records{failed_suffix}.txt"
# 
#     with open(failed_file_name, "w", encoding="utf-8") as f:
#         for rec in failed_records:
#             f.write(f"{rec}\n")
# 
#     print(f"📄 لیست رکوردهای ناموفق برای ارتفاع {h_val} متر در فایل {failed_file_name} ذخیره شد.")
# 
# print(f"🏁 اجرای همه رکوردها برای همه ارتفاع‌ها تمام شد. حالت اجرا: {RUN_MODE} | مدل: {'خطی' if IS_LINEAR else 'غیرخطی'}")
# 
# =============================================================================







# =============================================================================
# 
# 
# 
# import sys, io
# import os
# import shutil
# import time
# import traceback
# 
# # ✅ NEW (ETA helpers)
# from datetime import datetime, timedelta
# from collections import deque
# 
# # ✅ جلوگیری از خطای UnicodeEncodeError در محیط‌های مختلف (Spyder، CMD، Run-Codes)
# if hasattr(sys.stdout, "buffer"):
#     sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
# 
# # ============================================================
# # ✅ ETA formatting / EMA
# # ============================================================
# def fmt_seconds(sec: float) -> str:
#     sec = max(0.0, float(sec))
#     h = int(sec // 3600)
#     m = int((sec % 3600) // 60)
#     s = int(sec % 60)
#     return f"{h:02d}:{m:02d}:{s:02d}"
# 
# def ema_update(prev: float, x: float, alpha: float = 0.2) -> float:
#     return x if prev is None else (alpha * x + (1 - alpha) * prev)
# 
# # ============================================================
# # ✅ سازگاری با جابه‌جایی فایل بین:
# #   Model\Time History Analysis (THA)  ↔  Model
# # ============================================================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# THA_DIR = os.path.join(BASE_DIR, "Time History Analysis (THA)")
# 
# if os.path.isdir(THA_DIR) and THA_DIR not in sys.path:
#     sys.path.insert(0, THA_DIR)
# 
# if BASE_DIR not in sys.path:
#     sys.path.insert(0, BASE_DIR)
# 
# def find_project_root(start_dir: str, max_up: int = 6) -> str:
#     """Root = جایی که پوشه Output وجود دارد."""
#     cur = os.path.abspath(start_dir)
#     for _ in range(max_up):
#         if os.path.isdir(os.path.join(cur, "Output")):
#             return cur
#         parent = os.path.dirname(cur)
#         if parent == cur:
#             break
#         cur = parent
#     return os.path.abspath(start_dir)
# 
# PROJECT_ROOT = find_project_root(BASE_DIR)
# 
# def locate_dep(filename: str) -> str:
#     """
#     فایل‌های وابسته را در این مسیرها پیدا می‌کند:
#       1) کنار همین اسکریپت
#       2) داخل Time History Analysis (THA)
#     """
#     c1 = os.path.join(BASE_DIR, filename)
#     if os.path.exists(c1):
#         return c1
#     c2 = os.path.join(THA_DIR, filename)
#     if os.path.exists(c2):
#         return c2
#     return filename
# 
# # ============================================================
# # Importها بعد از تنظیم sys.path
# # ============================================================
# from openseespy.opensees import *
# from ReadRecord import ReadRecord
# from analyzeAndAnimate import analyzeAndAnimateTHA
# import vfo.vfo as vfo
# import opsvis as opsv
# from doDynamicAnalysis import doDynamicAnalysis
# 
# # ============================================================
# # ✅ تنظیمات RESUME
# # ============================================================
# # اگر True باشد، مثل قبل پوشه هر ارتفاع را پاک می‌کند (برای اجرای تمیز)
# CLEAN_START_PER_HEIGHT = False
# 
# # اسم فایل مارکر پایان موفق هر رکورد
# DONE_MARKER_NAME = "__DONE__.txt"
# 
# def done_marker_path(record_out_dir: str) -> str:
#     return os.path.join(record_out_dir, DONE_MARKER_NAME)
# 
# def is_record_done(record_out_dir: str) -> bool:
#     return os.path.isfile(done_marker_path(record_out_dir))
# 
# def write_done_marker(record_out_dir: str, extra_text: str = ""):
#     p = done_marker_path(record_out_dir)
#     with open(p, "w", encoding="utf-8") as f:
#         f.write("DONE\n")
#         f.write(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
#         if extra_text:
#             f.write(extra_text.strip() + "\n")
# 
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
# 
# # ------------------------------------------------------------
# # 🔧 انتخاب مدل خطی / غیرخطی
# # ------------------------------------------------------------
# lin_choice = input("مدل خطی باشد یا غیرخطی؟ برای مدل خطی عدد 1 و برای غیرخطی عدد 0 را وارد کن: ").strip()
# IS_LINEAR = (lin_choice == "1")
# 
# print(f"📌 حالت اجرا: {RUN_MODE} | مدل: {'خطی' if IS_LINEAR else 'غیرخطی'}")
# 
# os.environ["RUN_MODE"] = RUN_MODE
# os.environ["IS_LINEAR"] = "1" if IS_LINEAR else "0"
# 
# # ------------------------------------------------------------
# # 📏 دریافت ارتفاع ستون (یک یا چند مقدار)
# # ------------------------------------------------------------
# heights_raw = input("ارتفاع ستون‌ها را وارد کن (مثلاً: 3 یا 3 4 5): ").strip()
# 
# if not heights_raw:
#     print("⚠️ هیچ ارتفاعی وارد نشد؛ مقدار پیش‌فرض 3 متر در نظر گرفته می‌شود.")
#     heights = [3.0]
# else:
#     heights = []
#     for token in heights_raw.replace(',', ' ').split():
#         try:
#             heights.append(float(token))
#         except ValueError:
#             print(f"⚠️ مقدار «{token}» عدد معتبری نیست و نادیده گرفته می‌شود.")
#     if not heights:
#         print("❌ هیچ ارتفاع معتبری وارد نشد. اجرای برنامه متوقف شد.")
#         sys.exit(1)
# 
# print("📏 ارتفاع‌های انتخاب‌شده برای ستون‌ها:", ", ".join(str(h) for h in heights))
# 
# # ------------------------------------------------------------
# # ⚙️ تنظیمات اولیه تحلیل
# # ------------------------------------------------------------
# scaleFac = 10 * 9.81
# TFree = 0
# dataDirRoot = PROJECT_ROOT
# 
# # ---------------------- پوشهٔ ورودی GM بر اساس حالت ----------------------
# if RUN_MODE == 'train':
#     GMFolder = os.path.join(dataDirRoot, 'Output', '1_IDA_Records_train')
# else:
#     GMFolder = os.path.join(dataDirRoot, 'Output', '1_IDA_Records_predict')
# 
# # ---------------------- پوشهٔ خروجی THA بر اساس حالت و خطی/غیرخطی ----------------------
# if RUN_MODE == 'train':
#     dataDirBase = os.path.join(dataDirRoot, 'Output', '2_THA_train_linear' if IS_LINEAR else '2_THA_train_nonlinear')
# else:
#     dataDirBase = os.path.join(dataDirRoot, 'Output', '2_THA_predict_linear' if IS_LINEAR else '2_THA_predict_nonlinear')
# 
# print(f"📥 پوشه رکوردهای ورودی: {GMFolder}")
# print(f"📂 پوشه پایه‌ی خروجی THA: {dataDirBase}\n")
# 
# # ------------------------------------------------------------
# # ✳️ پیدا کردن همه رکوردهای .AT2
# # ------------------------------------------------------------
# if not os.path.isdir(GMFolder):
#     raise FileNotFoundError(f"❌ پوشه ورودی رکوردها پیدا نشد: {GMFolder}")
# 
# all_records = [f for f in os.listdir(GMFolder) if f.endswith('.AT2')]
# all_records.sort()
# print(f"🔍 تعداد رکوردهای پیدا شده: {len(all_records)}")
# 
# # ============================================================
# # ✅ NEW: GLOBAL ETA across ALL selected heights
# #   - Count remaining runs (not DONE) for all heights upfront
# #   - Track progress across heights while running
# # ============================================================
# EMA_ALPHA_GLOBAL = 0.2
# ema_total_global = None
# ema_model_global = None
# global_last100_total = deque(maxlen=100)
# global_last100_model = deque(maxlen=100)
# 
# global_total_to_run = 0
# for h_val in heights:
#     if float(h_val).is_integer():
#         h_tag_tmp = f"H{int(h_val)}"
#     else:
#         h_tag_tmp = "H" + str(h_val).replace('.', 'p')
# 
#     dataDirOut_tmp = os.path.join(dataDirBase, h_tag_tmp)
#     # اگر پوشه وجود ندارد، یعنی هیچ DONE ای هم نیست => همه رکوردها باقی‌مانده‌اند
#     for rf in all_records:
#         rn = os.path.splitext(rf)[0]
#         rec_dir = os.path.join(dataDirOut_tmp, rn)
#         if not os.path.isfile(os.path.join(rec_dir, DONE_MARKER_NAME)):
#             global_total_to_run += 1
# 
# global_executed = 0
# global_start_perf = time.perf_counter()
# 
# print("============================================================")
# print(f"🌍 کل پروژه (همه ارتفاع‌ها): کل رکوردها={len(all_records)} | تعداد ارتفاع‌ها={len(heights)}")
# print(f"🌍 قابل اجرا (بدون DONE) در همه ارتفاع‌ها: {global_total_to_run}")
# print("============================================================")
# 
# # ------------------------------------------------------------
# # 🚀 اجرای تحلیل برای هر ارتفاع و هر رکورد (با RESUME)
# # ------------------------------------------------------------
# for h_val in heights:
#     os.environ["H_COL"] = str(h_val)
# 
#     if float(h_val).is_integer():
#         h_tag = f"H{int(h_val)}"
#     else:
#         h_tag = "H" + str(h_val).replace('.', 'p')
# 
#     dataDirOut = os.path.join(dataDirBase, h_tag)
#     os.makedirs(dataDirOut, exist_ok=True)
# 
#     print(f"\n🏗️ ارتفاع ستون = {h_val} متر | مسیر خروجی ارتفاع: {dataDirOut}")
# 
#     # ✅ فقط اگر کاربر بخواهد از صفر برای این ارتفاع شروع کند
#     if CLEAN_START_PER_HEIGHT:
#         print(f"🧹 CLEAN_START_PER_HEIGHT=True → حذف کامل پوشه ارتفاع: {dataDirOut}")
#         shutil.rmtree(dataDirOut, ignore_errors=True)
#         os.makedirs(dataDirOut, exist_ok=True)
# 
#     failed_records = []
#     skipped = 0
#     executed = 0
# 
#     # ============================================================
#     # ✅ ETA stats per height + last100 reporting (per-height)
#     # ============================================================
#     ema_total = None
#     ema_model = None
#     last100_total = deque(maxlen=100)
#     last100_model = deque(maxlen=100)
# 
#     # تعداد رکوردهایی که واقعاً باید اجرا شوند (DONE ها حذف می‌شوند) برای همین ارتفاع
#     to_run_total = 0
#     for rf in all_records:
#         rn = os.path.splitext(rf)[0]
#         rec_dir = os.path.join(dataDirOut, rn)
#         if not os.path.isfile(os.path.join(rec_dir, DONE_MARKER_NAME)):
#             to_run_total += 1
# 
#     print(f"📌 این ارتفاع: کل رکوردها={len(all_records)} | قابل اجرا (بدون DONE)={to_run_total}")
#     height_start_perf = time.perf_counter()
# 
#     for i, rec_file in enumerate(all_records, start=1):
#         record_name = os.path.splitext(rec_file)[0]
#         inFileName = os.path.join(GMFolder, rec_file)
#         GMPath = os.path.join(GMFolder, record_name + ".txt")
# 
#         # پوشه خروجی رکورد
#         dataDir_rec = os.path.join(dataDirOut, record_name)
#         os.makedirs(dataDir_rec, exist_ok=True)
# 
#         # ✅ اگر قبلاً DONE شده، skip
#         if is_record_done(dataDir_rec):
#             skipped += 1
#             print(f"⏭️  SKIP (DONE) | {i}/{len(all_records)} | H={h_val} | {record_name}")
#             continue
# 
#         # timer for total record duration
#         t_rec0 = time.perf_counter()
# 
#         try:
#             # پاکسازی امن پوشه رکورد
#             for fname in os.listdir(dataDir_rec):
#                 fpath = os.path.join(dataDir_rec, fname)
#                 try:
#                     if os.path.isfile(fpath) or os.path.islink(fpath):
#                         os.remove(fpath)
#                     elif os.path.isdir(fpath):
#                         shutil.rmtree(fpath)
#                 except Exception:
#                     pass
# 
#             # ⚡ Recorderها از این متغیر برای مسیر خروجی استفاده می‌کنند
#             dataDir = dataDir_rec
# 
#             # همیشه قبل از ساخت مدل، wipe ایمن
#             wipe()
# 
#             # 🔹 اجرای مدل (خطی یا غیرخطی) و میرایی و رکوردرها
#             if IS_LINEAR:
#                 exec(io.open(locate_dep("model_linear.py"), "r", encoding="utf-8").read())
#             else:
#                 exec(open(locate_dep("model_nonlinear.py"), "r", encoding="utf-8").read())
# 
#             exec(open(locate_dep("defineDamping.py"), "r", encoding="utf-8").read())
#             exec(open(locate_dep("defineRecorders.py"), "r", encoding="utf-8").read())
# 
#             # 🔹 خواندن رکورد و تعریف تحریک
#             transformed_path = os.path.join(GMFolder, "transformed")
#             os.makedirs(transformed_path, exist_ok=True)
# 
#             dtInput, numPoints = ReadRecord(inFileName, GMPath)
#             seriesTag = 2
#             timeSeries('Path', seriesTag, '-dt', dtInput, '-filePath', GMPath, '-factor', scaleFac)
#             GMDir = 1
#             pattern('UniformExcitation', 2, GMDir, '-accel', seriesTag)
# 
#             Tmax = numPoints * dtInput + TFree
# 
#             mode_str = "train" if RUN_MODE == "train" else "predict"
#             lin_str = "خطی" if IS_LINEAR else "غیرخطی"
#             print(f"⚙️ RUN | {i}/{len(all_records)} | H={h_val} | {record_name}  ({mode_str}, {lin_str})")
# 
#             # timer for model/analysis duration only
#             t_model0 = time.perf_counter()
#             doDynamicAnalysis(Tmax, dtInput)
#             t_model1 = time.perf_counter()
#             model_sec = t_model1 - t_model0
# 
#             # total record time
#             t_rec1 = time.perf_counter()
#             total_sec = t_rec1 - t_rec0
# 
#             # ✅ مارکر DONE
#             write_done_marker(
#                 dataDir_rec,
#                 extra_text=f"mode={RUN_MODE}, model={'linear' if IS_LINEAR else 'nonlinear'}, H={h_val}, rec={record_name}"
#             )
# 
#             executed += 1
#             global_executed += 1
#             print(f"✅ DONE | H={h_val} | {record_name}\n")
# 
#             # wipe بعد از پایان رکورد
#             wipe()
# 
#             # ---------------- per-height stats ----------------
#             last100_total.append(total_sec)
#             last100_model.append(model_sec)
#             ema_total = ema_update(ema_total, total_sec, alpha=EMA_ALPHA_GLOBAL)
#             ema_model = ema_update(ema_model, model_sec, alpha=EMA_ALPHA_GLOBAL)
# 
#             remain_h = max(0, to_run_total - executed)
#             avg_total_h = ema_total if ema_total is not None else total_sec
#             eta_h_sec = remain_h * avg_total_h
#             finish_h = datetime.now() + timedelta(seconds=eta_h_sec)
#             elapsed_h = time.perf_counter() - height_start_perf
# 
#             # ---------------- global stats ----------------
#             global_last100_total.append(total_sec)
#             global_last100_model.append(model_sec)
#             ema_total_global = ema_update(ema_total_global, total_sec, alpha=EMA_ALPHA_GLOBAL)
#             ema_model_global = ema_update(ema_model_global, model_sec, alpha=EMA_ALPHA_GLOBAL)
# 
#             remain_g = max(0, global_total_to_run - global_executed)
#             avg_total_g = ema_total_global if ema_total_global is not None else total_sec
#             eta_g_sec = remain_g * avg_total_g
#             finish_g = datetime.now() + timedelta(seconds=eta_g_sec)
#             elapsed_g = time.perf_counter() - global_start_perf
# 
#             # ---------------- prints ----------------
#             # print(f"⏱️ زمان رکورد (کل): {fmt_seconds(total_sec)} | زمان مدل/تحلیل: {fmt_seconds(model_sec)}")
#             # print(f"📈 این ارتفاع: {executed}/{to_run_total} | باقی‌مانده: {remain_h} | ETA(H): {fmt_seconds(eta_h_sec)} → {finish_h.strftime('%Y-%m-%d %H:%M:%S')}")
#             print(f"🌍 کل پروژه: {global_executed}/{global_total_to_run} | باقی‌مانده: {remain_g} | ETA(ALL): {fmt_seconds(eta_g_sec)} → {finish_g.strftime('%Y-%m-%d %H:%M:%S')}")
#             # print(f"🕒 سپری‌شده: این ارتفاع={fmt_seconds(elapsed_h)} | کل پروژه={fmt_seconds(elapsed_g)}")
# 
#             # 100-run reports (only if we have 100 successes IN THIS SESSION)
#             if len(last100_total) == 100:
#                 print(f"📊 ۱۰۰ اجرای قبلی (این ارتفاع): کل={fmt_seconds(sum(last100_total))} | مدل/تحلیل={fmt_seconds(sum(last100_model))}")
#             if len(global_last100_total) == 100:
#                 print(f"📊 ۱۰۰ اجرای قبلی (کل پروژه): کل={fmt_seconds(sum(global_last100_total))} | مدل/تحلیل={fmt_seconds(sum(global_last100_model))}")
# 
#         except KeyboardInterrupt:
#             print("\n🛑 اجرای برنامه توسط کاربر متوقف شد (KeyboardInterrupt).")
#             print("✅ رکوردهای DONE شده حفظ می‌شوند؛ دفعه بعد ادامه می‌دهد.")
#             raise
# 
#         except Exception as e:
#             failed_records.append(rec_file)
#             print(f"❌ FAIL | H={h_val} | {rec_file}: {e}")
# 
#             # فایل خطا
#             try:
#                 err_path = os.path.join(dataDir_rec, "__ERROR__.txt")
#                 with open(err_path, "w", encoding="utf-8") as f:
#                     f.write(f"ERROR for record: {rec_file}\n")
#                     f.write(f"H={h_val}, mode={RUN_MODE}, model={'linear' if IS_LINEAR else 'nonlinear'}\n\n")
#                     f.write("Exception:\n")
#                     f.write(str(e) + "\n\n")
#                     f.write("Traceback:\n")
#                     f.write(traceback.format_exc())
#             except Exception:
#                 pass
# 
#             try:
#                 wipe()
#             except Exception:
#                 pass
# 
#     # ذخیره لیست رکوردهای ناموفق برای این ارتفاع
#     failed_suffix = f"_{RUN_MODE}_{'linear' if IS_LINEAR else 'nonlinear'}_{h_tag}"
#     failed_file_name = os.path.join(dataDirOut, f"failed_records{failed_suffix}.txt")
#     with open(failed_file_name, "w", encoding="utf-8") as f:
#         for rec in failed_records:
#             f.write(f"{rec}\n")
# 
#     print("------------------------------------------------------------")
#     print(f"📌 خلاصه ارتفاع {h_tag}:")
#     print(f"   ✅ اجرا شده جدید: {executed}")
#     print(f"   ⏭️  اسکیپ (قبلاً DONE): {skipped}")
#     print(f"   ❌ ناموفق: {len(failed_records)}")
#     print(f"📄 لاگ ناموفق‌ها: {failed_file_name}")
#     print("------------------------------------------------------------")
# 
# print(f"\n🏁 پایان اجرای همه ارتفاع‌ها. حالت: {RUN_MODE} | مدل: {'خطی' if IS_LINEAR else 'غیرخطی'}")
# 
# =============================================================================







# -*- coding: utf-8 -*-
import sys, io
import os
import shutil
import time
import traceback
from datetime import datetime, timedelta
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# ✅ جلوگیری از خطای UnicodeEncodeError در محیط‌های مختلف (Spyder، CMD، Run-Codes)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")

# ============================================================
# ✅ ETA formatting / EMA
# ============================================================
def fmt_seconds(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def ema_update(prev: float, x: float, alpha: float = 0.2) -> float:
    return x if prev is None else (alpha * x + (1 - alpha) * prev)

# ============================================================
# ✅ Root finder: جایی که پوشه Output وجود دارد
# ============================================================
def find_project_root(start_dir: str, max_up: int = 6) -> str:
    cur = os.path.abspath(start_dir)
    for _ in range(max_up):
        if os.path.isdir(os.path.join(cur, "Output")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(start_dir)

# ============================================================
# ✅ RESUME markers
# ============================================================
DONE_MARKER_NAME = "__DONE__.txt"

def done_marker_path(record_out_dir: str) -> str:
    return os.path.join(record_out_dir, DONE_MARKER_NAME)

def is_record_done(record_out_dir: str) -> bool:
    return os.path.isfile(done_marker_path(record_out_dir))

def write_done_marker(record_out_dir: str, extra_text: str = ""):
    p = done_marker_path(record_out_dir)
    with open(p, "w", encoding="utf-8") as f:
        f.write("DONE\n")
        f.write(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if extra_text:
            f.write(extra_text.strip() + "\n")

def safe_clean_dir(folder: str):
    """حذف امن محتوای پوشه (بدون حذف خود پوشه)."""
    if not os.path.isdir(folder):
        return
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        try:
            if os.path.isfile(fpath) or os.path.islink(fpath):
                os.remove(fpath)
            elif os.path.isdir(fpath):
                shutil.rmtree(fpath, ignore_errors=True)
        except Exception:
            pass

def height_tag(h_val: float) -> str:
    if float(h_val).is_integer():
        return f"H{int(h_val)}"
    return "H" + str(h_val).replace(".", "p")

# ============================================================
# ✅ Worker: اجرای یک رکورد برای یک ارتفاع (در یک پردازه مستقل)
# ============================================================
def run_one_job(job: dict) -> dict:
    """
    job keys:
      - h_val, rec_file, GMFolder, dataDirBase, PROJECT_ROOT, BASE_DIR, THA_DIR
      - RUN_MODE, IS_LINEAR
      - scaleFac, TFree
    """
    h_val      = job["h_val"]
    rec_file   = job["rec_file"]
    GMFolder   = job["GMFolder"]
    dataDirBase= job["dataDirBase"]
    RUN_MODE   = job["RUN_MODE"]
    IS_LINEAR  = job["IS_LINEAR"]
    scaleFac   = job["scaleFac"]
    TFree      = job["TFree"]
    BASE_DIR   = job["BASE_DIR"]
    THA_DIR    = job["THA_DIR"]

    # برای ماژول‌های وابسته (ReadRecord, doDynamicAnalysis, ...)
    if os.path.isdir(THA_DIR) and THA_DIR not in sys.path:
        sys.path.insert(0, THA_DIR)
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)

    def locate_dep(filename: str) -> str:
        c1 = os.path.join(BASE_DIR, filename)
        if os.path.exists(c1):
            return c1
        c2 = os.path.join(THA_DIR, filename)
        if os.path.exists(c2):
            return c2
        return filename

    # خروجی‌ها
    record_name = os.path.splitext(rec_file)[0]
    h_tag = height_tag(h_val)
    dataDirOut = os.path.join(dataDirBase, h_tag)
    dataDir_rec = os.path.join(dataDirOut, record_name)
    os.makedirs(dataDir_rec, exist_ok=True)

    # اگر DONE است، سریع برگرد
    if is_record_done(dataDir_rec):
        return {
            "status": "SKIP",
            "h_val": h_val,
            "rec_file": rec_file,
            "record_name": record_name,
            "dataDir_rec": dataDir_rec,
            "total_sec": 0.0,
            "model_sec": 0.0,
            "error": ""
        }

    # envها (مدل‌های شما از این‌ها استفاده می‌کنند)
    os.environ["RUN_MODE"] = RUN_MODE
    os.environ["IS_LINEAR"] = "1" if IS_LINEAR else "0"
    os.environ["H_COL"] = str(h_val)

    inFileName = os.path.join(GMFolder, rec_file)
    GMPath = os.path.join(GMFolder, record_name + ".txt")

    # تایمر کل رکورد
    t_rec0 = time.perf_counter()

    try:
        # پاکسازی امن پوشه رکورد
        safe_clean_dir(dataDir_rec)

        # Importهای OpenSees داخل پردازه (خیلی مهم)
        from openseespy.opensees import wipe, timeSeries, pattern
        from ReadRecord import ReadRecord
        from doDynamicAnalysis import doDynamicAnalysis

        # مدل‌ها به dataDir نیاز دارند (Recorderها)
        dataDir = dataDir_rec  # noqa: F841  (برای فایل‌های exec)

        # wipe ایمن
        wipe()

        # ساخت مدل
        if IS_LINEAR:
            exec(io.open(locate_dep("model_linear.py"), "r", encoding="utf-8").read(), globals(), locals())
        else:
            exec(io.open(locate_dep("model_nonlinear.py"), "r", encoding="utf-8").read(), globals(), locals())

        exec(io.open(locate_dep("defineDamping.py"), "r", encoding="utf-8").read(), globals(), locals())
        exec(io.open(locate_dep("defineRecorders.py"), "r", encoding="utf-8").read(), globals(), locals())

        # خواندن رکورد و تعریف تحریک
        transformed_path = os.path.join(GMFolder, "transformed")
        os.makedirs(transformed_path, exist_ok=True)

        dtInput, numPoints = ReadRecord(inFileName, GMPath)
        seriesTag = 2
        timeSeries("Path", seriesTag, "-dt", dtInput, "-filePath", GMPath, "-factor", scaleFac)
        GMDir = 1
        pattern("UniformExcitation", 2, GMDir, "-accel", seriesTag)

        Tmax = numPoints * dtInput + TFree

        # تایمر تحلیل (مدل)
        t_model0 = time.perf_counter()
        doDynamicAnalysis(Tmax, dtInput)
        t_model1 = time.perf_counter()
        model_sec = t_model1 - t_model0

        # زمان کل رکورد
        t_rec1 = time.perf_counter()
        total_sec = t_rec1 - t_rec0

        # مارکر DONE
        write_done_marker(
            dataDir_rec,
            extra_text=f"mode={RUN_MODE}, model={'linear' if IS_LINEAR else 'nonlinear'}, H={h_val}, rec={record_name}"
        )

        # wipe پایان رکورد
        wipe()

        return {
            "status": "OK",
            "h_val": h_val,
            "rec_file": rec_file,
            "record_name": record_name,
            "dataDir_rec": dataDir_rec,
            "total_sec": float(total_sec),
            "model_sec": float(model_sec),
            "error": ""
        }

    except KeyboardInterrupt:
        # در پردازه‌ها بهتر است همین را پاس بدهیم
        raise

    except Exception as e:
        # فایل خطا
        try:
            err_path = os.path.join(dataDir_rec, "__ERROR__.txt")
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(f"ERROR for record: {rec_file}\n")
                f.write(f"H={h_val}, mode={RUN_MODE}, model={'linear' if IS_LINEAR else 'nonlinear'}\n\n")
                f.write("Exception:\n")
                f.write(str(e) + "\n\n")
                f.write("Traceback:\n")
                f.write(traceback.format_exc())
        except Exception:
            pass

        return {
            "status": "FAIL",
            "h_val": h_val,
            "rec_file": rec_file,
            "record_name": record_name,
            "dataDir_rec": dataDir_rec,
            "total_sec": float(time.perf_counter() - t_rec0),
            "model_sec": 0.0,
            "error": f"{e}"
        }

# ============================================================
# ✅ MAIN
# ============================================================
def main():
    # سازگاری با جابه‌جایی فایل بین:
    #   Model\Time History Analysis (THA)  ↔  Model
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    THA_DIR  = os.path.join(BASE_DIR, "Time History Analysis (THA)")
    PROJECT_ROOT = find_project_root(BASE_DIR)

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

    # ------------------------------------------------------------
    # 📏 دریافت ارتفاع ستون (یک یا چند مقدار)
    # ------------------------------------------------------------
    heights_raw = input("ارتفاع ستون‌ها را وارد کن (مثلاً: 3 یا 3 4 5): ").strip()

    if not heights_raw:
        print("⚠️ هیچ ارتفاعی وارد نشد؛ مقدار پیش‌فرض 3 متر در نظر گرفته می‌شود.")
        heights = [3.0]
    else:
        heights = []
        for token in heights_raw.replace(",", " ").split():
            try:
                heights.append(float(token))
            except ValueError:
                print(f"⚠️ مقدار «{token}» عدد معتبری نیست و نادیده گرفته می‌شود.")
        if not heights:
            print("❌ هیچ ارتفاع معتبری وارد نشد. اجرای برنامه متوقف شد.")
            sys.exit(1)

    print("📏 ارتفاع‌های انتخاب‌شده:", ", ".join(str(h) for h in heights))

    # ------------------------------------------------------------
    # ⚙️ تنظیمات اولیه تحلیل
    # ------------------------------------------------------------
    scaleFac = 10 * 9.81
    TFree = 0
    dataDirRoot = PROJECT_ROOT

    # ---------------------- پوشهٔ ورودی GM ----------------------
    if RUN_MODE == "train":
        GMFolder = os.path.join(dataDirRoot, "Output", "1_IDA_Records_train")
    else:
        GMFolder = os.path.join(dataDirRoot, "Output", "1_IDA_Records_predict")

    # ---------------------- پوشهٔ خروجی THA ----------------------
    if RUN_MODE == "train":
        dataDirBase = os.path.join(
            dataDirRoot,
            "Output",
            "2_THA_train_linear" if IS_LINEAR else "2_THA_train_nonlinear"
        )
    else:
        dataDirBase = os.path.join(
            dataDirRoot,
            "Output",
            "2_THA_predict_linear" if IS_LINEAR else "2_THA_predict_nonlinear"
        )

    print(f"📥 پوشه رکوردهای ورودی: {GMFolder}")
    print(f"📂 پوشه پایه‌ی خروجی THA: {dataDirBase}")

    # ------------------------------------------------------------
    # ✳️ پیدا کردن همه رکوردهای .AT2
    # ------------------------------------------------------------
    if not os.path.isdir(GMFolder):
        raise FileNotFoundError(f"❌ پوشه ورودی رکوردها پیدا نشد: {GMFolder}")

    all_records = [f for f in os.listdir(GMFolder) if f.endswith(".AT2")]
    all_records.sort()
    print(f"🔍 تعداد رکوردهای پیدا شده: {len(all_records)}")

    # ------------------------------------------------------------
    # 🧹 اگر می‌خواهید از صفر شروع شود (اختیاری)
    # ------------------------------------------------------------
    CLEAN_START_PER_HEIGHT = False
    clean_choice = input("اگر می‌خواهی برای هر ارتفاع خروجی‌ها کامل پاک شود عدد 1 وگرنه 0: ").strip()
    if clean_choice == "1":
        CLEAN_START_PER_HEIGHT = True

    if CLEAN_START_PER_HEIGHT:
        print("🧹 حالت پاکسازی فعال شد: برای هر ارتفاع، پوشه خروجی آن ارتفاع پاک می‌شود.")

    # ------------------------------------------------------------
    # 👷 تعداد پردازه‌های موازی (بهینه برای سیستم شما: 6 پیشنهاد می‌شود)
    # ------------------------------------------------------------
    cpu_count = os.cpu_count() or 8
    default_workers = min(max(2, cpu_count - 2), 8)  # معمولاً 6 تا 8 خوب است
    w_in = input(f"تعداد پردازه‌های همزمان (پیشنهادی {default_workers}): ").strip()
    if not w_in:
        max_workers = default_workers
    else:
        try:
            max_workers = int(w_in)
            max_workers = max(1, max_workers)
        except ValueError:
            max_workers = default_workers

    print(f"🧠 CPU Threads: {cpu_count} | پردازه‌های همزمان: {max_workers}")
    print("------------------------------------------------------------")

    # ------------------------------------------------------------
    # ✅ اگر CLEAN_START_PER_HEIGHT=True، قبل از ساخت Jobها پاک کن
    # ------------------------------------------------------------
    if CLEAN_START_PER_HEIGHT:
        for h_val in heights:
            h_tag = height_tag(h_val)
            h_dir = os.path.join(dataDirBase, h_tag)
            if os.path.isdir(h_dir):
                print(f"🧹 حذف پوشه ارتفاع {h_tag}: {h_dir}")
                shutil.rmtree(h_dir, ignore_errors=True)
            os.makedirs(h_dir, exist_ok=True)

    # ------------------------------------------------------------
    # ✅ ساخت لیست Jobها (فقط آن‌هایی که DONE نیستند)
    # ------------------------------------------------------------
    jobs = []
    for h_val in heights:
        h_tag = height_tag(h_val)
        h_dir = os.path.join(dataDirBase, h_tag)
        os.makedirs(h_dir, exist_ok=True)

        for rec_file in all_records:
            record_name = os.path.splitext(rec_file)[0]
            rec_dir = os.path.join(h_dir, record_name)
            if not os.path.isfile(os.path.join(rec_dir, DONE_MARKER_NAME)):
                jobs.append({
                    "h_val": h_val,
                    "rec_file": rec_file,
                    "GMFolder": GMFolder,
                    "dataDirBase": dataDirBase,
                    "PROJECT_ROOT": PROJECT_ROOT,
                    "BASE_DIR": BASE_DIR,
                    "THA_DIR": THA_DIR,
                    "RUN_MODE": RUN_MODE,
                    "IS_LINEAR": IS_LINEAR,
                    "scaleFac": scaleFac,
                    "TFree": TFree
                })

    total_to_run = len(jobs)
    print("============================================================")
    print(f"🌍 کل پروژه: رکوردها={len(all_records)} | ارتفاع‌ها={len(heights)}")
    print(f"🌍 قابل اجرا (بدون DONE): {total_to_run}")
    print("============================================================")

    if total_to_run == 0:
        print("✅ همه رکوردها قبلاً DONE شده‌اند. چیزی برای اجرا باقی نمانده است.")
        return

    # ------------------------------------------------------------
    # ✅ ETA سراسری
    # ------------------------------------------------------------
    EMA_ALPHA_GLOBAL = 0.2
    ema_total_global = None
    ema_model_global = None
    last100_total = deque(maxlen=100)
    last100_model = deque(maxlen=100)

    executed_ok = 0
    executed_fail = 0
    executed_skip = 0

    global_start = time.perf_counter()

    # برای گزارش failed per height هم فایل می‌سازیم:
    failed_map = {height_tag(h): [] for h in heights}

    # ------------------------------------------------------------
    # 🚀 اجرای موازی
    # ------------------------------------------------------------
    print("🚀 شروع اجرای موازی...")
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_one_job, j) for j in jobs]

        for idx, fu in enumerate(as_completed(futures), start=1):
            res = fu.result()

            st = res["status"]
            h_val = res["h_val"]
            rec_file = res["rec_file"]
            rec_name = res["record_name"]

            if st == "OK":
                executed_ok += 1

                total_sec = res["total_sec"]
                model_sec = res["model_sec"]

                last100_total.append(total_sec)
                last100_model.append(model_sec)

                ema_total_global = ema_update(ema_total_global, total_sec, alpha=EMA_ALPHA_GLOBAL)
                ema_model_global = ema_update(ema_model_global, model_sec, alpha=EMA_ALPHA_GLOBAL)

                remain = max(0, total_to_run - (executed_ok + executed_fail))
                avg_total = ema_total_global if ema_total_global is not None else total_sec
                eta_sec = remain * avg_total
                finish_time = datetime.now() + timedelta(seconds=eta_sec)

                print(f"✅ DONE | {idx}/{total_to_run} | H={h_val} | {rec_name}")
                print(f"🌍 کل پروژه: OK={executed_ok} | FAIL={executed_fail} | باقی‌مانده≈{remain} | ETA(ALL): {fmt_seconds(eta_sec)} → {finish_time.strftime('%Y-%m-%d %H:%M:%S')}")
                if len(last100_total) == 100:
                    print(f"📊 ۱۰۰ اجرای قبلی: کل={fmt_seconds(sum(last100_total))} | مدل/تحلیل={fmt_seconds(sum(last100_model))}")

            elif st == "FAIL":
                executed_fail += 1
                ht = height_tag(h_val)
                failed_map[ht].append(rec_file)
                print(f"❌ FAIL | {idx}/{total_to_run} | H={h_val} | {rec_file} | {res.get('error','')}")

            elif st == "SKIP":
                executed_skip += 1
                print(f"⏭️  SKIP (DONE) | H={h_val} | {rec_name}")

    elapsed = time.perf_counter() - global_start
    print("============================================================")
    print("🏁 پایان اجرای موازی")
    print(f"⏱️ زمان سپری‌شده: {fmt_seconds(elapsed)}")
    print(f"✅ موفق: {executed_ok} | ❌ ناموفق: {executed_fail} | ⏭️ اسکیپ: {executed_skip}")
    print("============================================================")

    # ------------------------------------------------------------
    # ذخیره failed_records برای هر ارتفاع
    # ------------------------------------------------------------
    for h_val in heights:
        h_tag = height_tag(h_val)
        dataDirOut = os.path.join(dataDirBase, h_tag)
        failed_suffix = f"_{RUN_MODE}_{'linear' if IS_LINEAR else 'nonlinear'}_{h_tag}"
        failed_file_name = os.path.join(dataDirOut, f"failed_records{failed_suffix}.txt")
        with open(failed_file_name, "w", encoding="utf-8") as f:
            for rec in failed_map[h_tag]:
                f.write(f"{rec}\n")

        print(f"📌 ارتفاع {h_tag}: ❌ ناموفق={len(failed_map[h_tag])} | 📄 لاگ: {failed_file_name}")

if __name__ == "__main__":
    # برای ویندوز
    mp.freeze_support()
    main()




