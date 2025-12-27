










# -*- coding: utf-8 -*-
import os
import glob
import time
import numpy as np
from tqdm import tqdm  # pip install tqdm


def Fixed_Files(dataDir, mode, is_linear, heights=None, min_lines=None):
    """
    پردازش داده‌ها برای آموزش/پیش‌بینی مدل با طول‌های متغیر، به ازای چند ارتفاع ستون.
    - disp.txt ها را جمع کرده و ستون اول آن‌ها (مثلاً زمان) حذف می‌شود.
    - X (زلزله) و Y (پاسخ سازه) برای هر رکورد به یکدیگر هم‌تراز می‌شوند
      (طول مشترک = مین(len(X), len(Y))).
    - برای هر ارتفاع، خروجی‌ها در پوشه‌ی مخصوص همان ارتفاع ذخیره می‌شود و
      نام فایل‌های NumPy نیز شامل ارتفاع خواهد بود (X_data_H? , Y_data_H?).
    """

    start_time = time.time()
    dataDir = os.path.abspath(dataDir)

    # ---------------------------
    # انتخاب train/predict و linear/nonlinear
    # ---------------------------
    mode = mode.strip().lower()          # 'train' یا 'predict'
    if mode not in ["train", "predict"]:
        raise ValueError("mode باید 'train' یا 'predict' باشد.")

    lin_str = "linear" if is_linear else "nonlinear"

    # 🔹 IDA فقط به mode وابسته است (خطی/غیرخطی ندارد)
    gm_input_dir = os.path.join(dataDir, 'Output', f'1_IDA_Records_{mode}', 'zire ham')

    # 🔹 THA به mode و lin_str وابسته است (ریشه‌ی ارتفاع‌ها)
    tha_root_dir = os.path.join(dataDir, 'Output', f'2_THA_{mode}_{lin_str}')

    # 🔹 ریشه‌ی خروجی‌های Fixed
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
    # توابع کمکی
    # ---------------------------
    def clear_or_make(folder_path, pattern='*.txt'):
        """
        اگر پوشه وجود داشته باشد، فایل‌هایی که با pattern می‌خورند حذف می‌شوند.
        اگر وجود نداشته باشد، ساخته می‌شود.
        """
        if os.path.exists(folder_path):
            for f in glob.glob(os.path.join(folder_path, pattern)):
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"⚠️ حذف {f} ناموفق: {e}")
        else:
            os.makedirs(folder_path, exist_ok=True)

    # 0) ساخت Merge_disp-files و کپی disp.txt با حذف ستون اول
    def merge_disp_files(tha_dir, tha_merge_dir):
        os.makedirs(tha_merge_dir, exist_ok=True)
        disp_files = glob.glob(os.path.join(tha_dir, '**', 'disp.txt'), recursive=True)
        print(f"🔍 {len(disp_files)} فایل disp.txt در {tha_dir} پیدا شد. در حال ادغام...")

        for file_path in tqdm(disp_files, desc="📄 ادغام disp-files (حذف ستون اول)"):
            try:
                # نام پوشه رکورد (مثلاً RSN4_...)
                folder_name = os.path.basename(os.path.dirname(file_path))
                output_file = os.path.join(tha_merge_dir, f"{folder_name}.txt")

                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                processed = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 1:
                        # حذف ستون اول
                        processed.append(' '.join(parts[1:]) + '\n')

                with open(output_file, 'w', encoding='utf-8') as out_f:
                    out_f.writelines(processed)

            except Exception as e:
                print(f"⚠️ خطا در پردازش {file_path}: {e}")

        print(f"✅ خروجی disp‌ها در {tha_merge_dir} ذخیره شد.\n")

    # 1) کپی مستقیم GMها بدون کوتاه‌سازی
    def copy_gm_files(input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        files = glob.glob(os.path.join(input_dir, '*.txt'))
        print(f"🔍 {len(files)} فایل GM در {input_dir} پیدا شد. در حال کپی...")

        for file in tqdm(files, desc="📂 کپی GM (بدون کوتاه‌سازی)"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                file_name = os.path.basename(file).replace('_for_ML', '')
                with open(os.path.join(output_dir, file_name), 'w', encoding='utf-8') as f_out:
                    f_out.writelines(lines)
            except Exception as e:
                print(f"⚠️ خطا در کپی {file}: {e}")

    # 2) هم‌تراز کردن طول X/Y هر رکورد
    def align_pairs_to_min_len(gm_fixed_dir, tha_merge_dir, tha_fixed_dir):
        os.makedirs(tha_fixed_dir, exist_ok=True)
        for gm_file in tqdm(glob.glob(os.path.join(gm_fixed_dir, '*.txt')),
                            desc="🔄 هم‌تراز کردن X/Y رکوردها"):
            file_name = os.path.basename(gm_file)
            tha_file_path = os.path.join(tha_merge_dir, file_name)

            if not os.path.exists(tha_file_path):
                # اگر Y وجود ندارد، این رکورد را نادیده بگیر
                continue

            try:
                with open(gm_file, 'r', encoding='utf-8') as f:
                    gm_lines = f.readlines()
                with open(tha_file_path, 'r', encoding='utf-8') as f:
                    tha_lines = f.readlines()

                L = min(len(gm_lines), len(tha_lines))
                if L == 0:
                    # رکورد بی‌معنی، حذف GM
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
                print(f"⚠️ خطا در هم‌ترازسازی {file_name}: {e}")

    # 3) حذف فایل‌های خالی
    def remove_empty_files(gm_fixed_dir, tha_fixed_dir):
        for folder in [gm_fixed_dir, tha_fixed_dir]:
            for file in tqdm(glob.glob(os.path.join(folder, '*.txt')),
                             desc=f"🗑 حذف فایل‌های خالی {os.path.basename(folder)}"):
                try:
                    if os.stat(file).st_size == 0:
                        os.remove(file)
                except Exception as e:
                    print(f"⚠️ حذف {file} ناموفق: {e}")

    # 4) ذخیرهٔ دیکشنری NumPy
    def save_numpy_data(input_dir, output_file):
        data_dict = {}
        for file in tqdm(glob.glob(os.path.join(input_dir, '*.txt')),
                         desc=f"📥 ساخت {os.path.basename(output_file)}"):
            file_name = os.path.basename(file)
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    arr = []
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        arr.append(float(parts[0]))  # تک‌کاناله
                    data = np.array(arr, dtype=np.float32)
                if data.size > 0:
                    data_dict[file_name] = data
            except Exception as e:
                print(f"⚠️ خطا در خواندن {file}: {e}")

        if not data_dict:
            print(f"⚠️ هیچ داده‌ای برای ذخیره در {output_file} پیدا نشد.")
        else:
            np.save(output_file, data_dict)
            print(f"✅ {output_file} ذخیره شد. شامل {len(data_dict)} رکورد.\n")

    # اگر heights داده نشده، سعی کن از روی فولدرهای THA تشخیص بدهی
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
        # اگر باز هم خالی بود، حداقل یک ارتفاع پیش‌فرض
        heights = [3.0]

    print(f"📏 ارتفاع‌های در حال پردازش: {', '.join(str(h) for h in heights)}\n")

    # ---------------------------
    # اجرای پایپ‌لاین برای هر ارتفاع
    # ---------------------------
    for h in heights:
        # برچسب پوشه برای این ارتفاع
        if float(h).is_integer():
            h_tag = f"H{int(h)}"        # مثال: H3
        else:
            h_tag = "H" + str(h).replace('.', 'p')   # مثال: H3p5

        tha_dir = os.path.join(tha_root_dir, h_tag)
        if not os.path.isdir(tha_dir):
            print(f"⚠️ پوشه THA برای ارتفاع {h} متر پیدا نشد: {tha_dir}  → رد می‌شود.\n")
            continue

        tha_merge_dir = os.path.join(tha_dir, 'Merge_disp-files')
        gm_fixed_dir = os.path.join(gm_fixed_root, h_tag)
        tha_fixed_dir = os.path.join(tha_fixed_root, h_tag)

        print("--------------------------------------------------")
        print(f"🏗️ شروع پردازش برای ارتفاع H = {h} m")
        print(f"📥 THA dir       = {tha_dir}")
        print(f"📥 THA merge dir = {tha_merge_dir}")
        print(f"📤 GM Fixed dir  = {gm_fixed_dir}")
        print(f"📤 THA Fixed dir = {tha_fixed_dir}")
        print("--------------------------------------------------\n")

        # اجرای پایپ‌لاین برای این ارتفاع
        merge_disp_files(tha_dir, tha_merge_dir)
        clear_or_make(gm_fixed_dir)       # پاک‌کردن txt های قبلی GM
        clear_or_make(tha_fixed_dir)      # پاک‌کردن txt های قبلی THA
        copy_gm_files(gm_input_dir, gm_fixed_dir)
        align_pairs_to_min_len(gm_fixed_dir, tha_merge_dir, tha_fixed_dir)
        remove_empty_files(gm_fixed_dir, tha_fixed_dir)

        # نام فایل‌های خروجی npy شامل ارتفاع
        x_out = os.path.join(gm_fixed_dir, f'X_data_{h_tag}.npy')
        y_out = os.path.join(tha_fixed_dir, f'Y_data_{h_tag}.npy')

        save_numpy_data(gm_fixed_dir, x_out)
        save_numpy_data(tha_fixed_dir, y_out)

        # در انتها txt های موقتی را پاک کن تا فقط npy باقی بماند
        clear_or_make(gm_fixed_dir, pattern='*.txt')
        clear_or_make(tha_fixed_dir, pattern='*.txt')

    total_time = round(time.time() - start_time, 2)
    print(f"\n✅ پردازش کامل شد! ⏳ زمان کل اجرا: {total_time} ثانیه\n")


# ---------------------------
# اجرای مستقیم
# ---------------------------
if __name__ == "__main__":
    # فرض: این فایل در پوشه‌ای مثل Codes_github/Model/ قرار دارد
    # و data_directory همان پوشه‌ی اصلی پروژه است (یک سطح بالاتر)
    data_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    print("=== مرحله ۳: ساخت فایل‌های Fixed و X/Y برای LSTM ===")

    # ۱) train یا predict
    choice = input("برای train عدد 0 و برای predict عدد 1 را وارد کن: ").strip()
    if choice == "0":
        mode = "train"
    elif choice == "1":
        mode = "predict"
    else:
        print("❌ فقط 0 یا 1 مجاز است.")
        raise SystemExit

    # ۲) خطی یا غیرخطی
    choice_lin = input("مدل خطی است یا غیرخطی؟ برای خطی 1 و برای غیرخطی 0 را وارد کن: ").strip()
    if choice_lin == "1":
        is_linear = True
    elif choice_lin == "0":
        is_linear = False
    else:
        print("❌ فقط 0 یا 1 مجاز است.")
        raise SystemExit

    # ۳) ارتفاع ستون‌ها
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

    Fixed_Files(data_directory, mode=mode, is_linear=is_linear, heights=heights)












