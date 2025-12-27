# # -*- coding: utf-8 -*-
# """
# Created on Wed Nov 19 09:21:33 2025

# @author: pc22
# """

# # -*- coding: utf-8 -*-
# """
# تحلیل رکوردهای PEER در Input/PEER_Predict
# و رسم یک عکس بزرگ شامل همه‌ی پارامترهای مهم
# (بدون تولید CSV یا Excel)

# خروجی:
#     Codes_github/Output/EQ_Predict_Analysis/eq_parameters_summary.png
# """

# import os
# import re
# import numpy as np
# import matplotlib.pyplot as plt

# # ------------------------------
# # مسیرها
# # ------------------------------
# base_dir = os.path.dirname(os.path.abspath(__file__))        # .../Model
# root_dir = os.path.abspath(os.path.join(base_dir, os.pardir))

# input_dir = os.path.join(root_dir, "Input", "PEER_Predict")
# output_dir = os.path.join(root_dir, "Output", "EQ_Predict_Analysis")
# os.makedirs(output_dir, exist_ok=True)

# print("📂 Input Dir :", input_dir)
# print("📂 Output Dir:", output_dir)

# # ------------------------------
# # خواندن فایل PEER .at2
# # ------------------------------
# def read_peer_at2(filepath):
#     with open(filepath, "r") as f:
#         lines = f.readlines()

#     dt = None
#     start = 0

#     for i, line in enumerate(lines):
#         if "NPTS" in line.upper() and "DT" in line.upper():
#             m = re.search(r"DT\s*=\s*([0-9Ee\+\-\.]+)", line)
#             if m:
#                 dt = float(m.group(1))
#             start = i + 1
#             break

#     if dt is None:
#         raise ValueError(f"DT not found in {filepath}")

#     data = " ".join(lines[start:])
#     accel = np.array([float(x) for x in data.split()], dtype=float)
#     return accel, dt

# # ------------------------------
# # محاسبه سری‌ها
# # ------------------------------
# def compute_time_series(acc, dt):
#     N = len(acc)
#     t = np.arange(N) * dt

    # سرعت
#     v = np.zeros_like(acc)
#     v[1:] = np.cumsum((acc[1:] + acc[:-1]) * 0.5 * dt)
#     v -= np.linspace(v[0], v[-1], N)  # حذف روند

    # جابجایی
#     d = np.zeros_like(acc)
#     d[1:] = np.cumsum((v[1:] + v[:-1]) * 0.5 * dt)
#     d -= np.linspace(d[0], d[-1], N)

#     return v, d

# # ------------------------------
# # سایر پارامترهای مهم
# # ------------------------------
# def compute_arias(acc, dt, g=9.81):
#     return (np.pi / (2*g)) * np.sum(acc**2 * dt)

# def compute_cav(acc, dt):
#     return np.sum(np.abs(acc) * dt)

# def compute_duration_5_95(acc, dt, g=9.81):
#     Ia_t = (np.pi/(2*g)) * np.cumsum(acc**2 * dt)
#     Ia_total = Ia_t[-1]
#     if Ia_total == 0:
#         return 0
#     ratio = Ia_t / Ia_total
#     t = np.arange(len(acc)) * dt
#     t5 = t[np.argmax(ratio >= 0.05)]
#     t95 = t[np.argmax(ratio >= 0.95)]
#     return t95 - t5

# def compute_predominant_period(acc, dt):
#     N = len(acc)
#     fft_vals = np.fft.rfft(acc)
#     freqs = np.fft.rfftfreq(N, dt)
#     amp = np.abs(fft_vals)
#     amp[0] = 0
#     idx = np.argmax(amp)
#     fp = freqs[idx]
#     return np.inf if fp == 0 else 1/fp

# # ------------------------------
# # پردازش تمام at2 ها
# # ------------------------------
# files = sorted(f for f in os.listdir(input_dir)
#                if f.lower().endswith(".at2"))

# if not files:
#     raise FileNotFoundError("❌ هیچ فایل at2 پیدا نشد.")

# # ذخیره پارامترها (فقط در حافظه، نه در فایل)
# names = []
# PGA = []
# PGV = []
# PGD = []
# Arias = []
# CAV = []
# Dur_5_95 = []
# Tp = []
# Dur_total = []

# for fname in files:
#     print(f"🔎 Processing {fname}")
#     acc, dt = read_peer_at2(os.path.join(input_dir, fname))
#     v, d = compute_time_series(acc, dt)

#     names.append(fname.replace(".at2", ""))
#     PGA.append(np.max(np.abs(acc)))
#     PGV.append(np.max(np.abs(v)))
#     PGD.append(np.max(np.abs(d)))
#     Arias.append(compute_arias(acc, dt))
#     CAV.append(compute_cav(acc, dt))
#     Dur_5_95.append(compute_duration_5_95(acc, dt))
#     Tp.append(compute_predominant_period(acc, dt))
#     Dur_total.append(len(acc)*dt)

# # ------------------------------
# # رسم شکل بزرگ
# # ------------------------------
# labels = [n if len(n)<=15 else n[:12]+"..." for n in names]
# x = np.arange(len(names))

# fig, axes = plt.subplots(3, 3, figsize=(20, 12))
# axes = axes.flatten()

# params = [
#     (PGA, "PGA"),
#     (PGV, "PGV"),
#     (PGD, "PGD"),
#     (Arias, "Arias Intensity"),
#     (CAV, "CAV"),
#     (Dur_5_95, "Duration 5–95%"),
#     (Tp, "Predominant Period"),
#     (Dur_total, "Total Duration"),
#     ([dt]*len(names), "dt"),
# ]

# for ax, (val, title) in zip(axes, params):
#     ax.bar(x, val)
#     ax.set_title(title)
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels, rotation=90, fontsize=7)
#     ax.grid(True, axis="y", linestyle="--", linewidth=0.4)

# plt.tight_layout()
# out_path = os.path.join(output_dir, "eq_parameters_summary.png")
# plt.savefig(out_path, dpi=300, bbox_inches='tight')
# plt.show()

# print("🖼️ تصویر نهایی ساخته شد →", out_path)
# print("✅ پایان تحلیل")




# -*- coding: utf-8 -*-
import sys, io
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')

import os
import re
import numpy as np
import matplotlib.pyplot as plt

# ============================================================== #
# 📁 مسیرها
# ============================================================== #
base_dir = os.path.dirname(os.path.abspath(__file__))         # Codes_github/Model
root_dir = os.path.dirname(base_dir)                          # Codes_github
peer_dir = os.path.join(root_dir, "Input", "PEER_Predict")    # رکوردها (at2)
out_dir  = os.path.join(root_dir, "Output", "EQ_Per_Analysis")    # پوشه تحلیل
os.makedirs(out_dir, exist_ok=True)

print("📂 PEER dir:", peer_dir)
print("📂 Output dir:", out_dir)

# ============================================================== #
# 🧩 تابع خواندن فایل PEER و استخراج dt
# ============================================================== #
def read_peer_at2(path):
    """
    یک فایل PEER .at2 را می‌خواند و:
      - acc: آرایه شتاب
      - dt: گام زمانی
      - name: نام رکورد (بدون پسوند)
    را برمی‌گرداند.
    منطق:
      → خطی که NPTS و DT در آن است پیدا می‌شود،
      → dt از همان خط با regex خوانده می‌شود،
      → بقیه‌ی خطوط به‌عنوان داده عددی در نظر گرفته می‌شوند.
    """
    with open(path, "r") as f:
        lines = f.readlines()

    dt = None
    start_idx = None

    for i, line in enumerate(lines):
        upper = line.upper()
        if "NPTS" in upper and "DT" in upper:
            # مثال: NPTS= 32768, DT=   .0050 SEC,
            m = re.search(r"DT\s*=\s*([0-9.+\-Ee]+)", line, flags=re.IGNORECASE)
            if m:
                dt = float(m.group(1))
            start_idx = i + 1
            break

    if dt is None or start_idx is None:
        raise ValueError(f"❌ نتوانستم dt / شروع داده را در فایل {os.path.basename(path)} پیدا کنم.")

    # همه‌چیز بعد از خط NPTS/DT داده عددی در نظر گرفته می‌شود
    data_str = " ".join(lines[start_idx:])
    acc = np.fromstring(data_str, sep=" ", dtype=float)

    if acc.size == 0:
        raise ValueError(f"❌ بعد از هدر، داده عددی در فایل {os.path.basename(path)} پیدا نشد.")

    name = os.path.splitext(os.path.basename(path))[0]
    return acc, dt, name

# ============================================================== #
# 🎯 توابع محاسبه پارامترها از روی a(t)، dt
# ============================================================== #
def compute_parameters(acc, dt):
    """
    از شتاب (acc) و dt:
      - PGA, PGV, PGD
      - Arias intensity (نسبی)
      - CAV
      - مدت ۵–۹۵٪ انرژی
      - مدت کل
      - پریود غالب Tp
    را برمی‌گرداند.
    """
    n = len(acc)
    t = np.arange(n) * dt

    # 🔹 انتگرال‌گیری سرعت با قاعده ذوزنقه‌ای (بهتر از cumsum ساده)
    vel = np.zeros_like(acc)
    vel[1:] = np.cumsum(0.5 * (acc[1:] + acc[:-1]) * dt)

    # خط روند کوچک سرعت را حذف می‌کنیم (drift correction ساده)
    vel -= np.linspace(vel[0], vel[-1], n)

    # 🔹 جابجایی
    disp = np.zeros_like(acc)
    disp[1:] = np.cumsum(0.5 * (vel[1:] + vel[:-1]) * dt)
    disp -= np.linspace(disp[0], disp[-1], n)

    # 🔹 PGA, PGV, PGD
    PGA = np.max(np.abs(acc))
    PGV = np.max(np.abs(vel))
    PGD = np.max(np.abs(disp))

    # 🔹 Arias Intensity (به‌صورت نسبی کافی است، ضریب ثابت مهم نیست برای مقایسه)
    a_sq = acc ** 2
    IA = np.sum(a_sq * dt)

    # 🔹 CAV
    CAV = np.sum(np.abs(acc) * dt)

    # 🔹 مدت‌زمان ۵–۹۵٪ انرژی
    cum_E = np.cumsum(a_sq * dt)
    E_total = cum_E[-1] + 1e-12
    cum_norm = cum_E / E_total
    try:
        t5  = t[np.searchsorted(cum_norm, 0.05)]
        t95 = t[np.searchsorted(cum_norm, 0.95)]
        D_5_95 = t95 - t5
    except Exception:
        D_5_95 = np.nan

    # 🔹 مدت‌زمان کل
    D_total = t[-1] - t[0] if n > 0 else np.nan

    # 🔹 پریود غالب از طیف فوریه
    freqs = np.fft.rfftfreq(n, d=dt)
    spec  = np.abs(np.fft.rfft(acc))
    if len(freqs) > 0:
        spec[0] = 0.0  # حذف فرکانس صفر
    idx_peak = np.argmax(spec)
    f_peak = freqs[idx_peak] if idx_peak < len(freqs) else 0.0
    Tp = 1.0 / f_peak if f_peak > 0 else np.nan

    return {
        "PGA": PGA,
        "PGV": PGV,
        "PGD": PGD,
        "IA": IA,
        "CAV": CAV,
        "D_5_95": D_5_95,
        "D_total": D_total,
        "Tp": Tp,
        "dt": dt,
    }

# ============================================================== #
# 🔍 جمع‌آوری فایل‌ها و محاسبه پارامترها
# ============================================================== #
files = sorted([
    f for f in os.listdir(peer_dir)
    if f.lower().endswith(".at2")
])

if not files:
    raise FileNotFoundError("❌ هیچ فایل .at2 در پوشه PEER_Predict پیدا نشد.")

names = []
PGA_list = []
PGV_list = []
PGD_list = []
IA_list  = []
CAV_list = []
D595_list = []
Dtot_list = []
Tp_list  = []
dt_list  = []

print("✅ فایل‌های پیدا شده:")
for f in files:
    print("  -", f)

for fname in files:
    path = os.path.join(peer_dir, fname)
    try:
        acc, dt, name = read_peer_at2(path)
    except Exception as e:
        print(f"⚠️ خطا در خواندن {fname}: {e}")
        continue

    params = compute_parameters(acc, dt)

    names.append(name)
    PGA_list.append(params["PGA"])
    PGV_list.append(params["PGV"])
    PGD_list.append(params["PGD"])
    IA_list.append(params["IA"])
    CAV_list.append(params["CAV"])
    D595_list.append(params["D_5_95"])
    Dtot_list.append(params["D_total"])
    Tp_list.append(params["Tp"])
    dt_list.append(params["dt"])

# اگر هیچ رکوردی موفقیت‌آمیز نبود:
if len(names) == 0:
    print("❌ هیچ رکوردی با موفقیت خوانده و پردازش نشد. لطفاً یکی از فایل‌های at2 را باز کن و هدرش را برای من بفرست.")
    raise SystemExit

# تبدیل به آرایه برای راحتی
PGA_list = np.array(PGA_list)
PGV_list = np.array(PGV_list)
PGD_list = np.array(PGD_list)
IA_list  = np.array(IA_list)
CAV_list = np.array(CAV_list)
D595_list = np.array(D595_list)
Dtot_list = np.array(Dtot_list)
Tp_list  = np.array(Tp_list)
dt_list  = np.array(dt_list)

# ============================================================== #
# 📊 رسم همه پارامترها در یک شکل
# ============================================================== #
N = len(names)
x = np.arange(N)

fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.ravel()

def barplot(ax, values, title, ylabel):
    ax.bar(x, values, alpha=0.7)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(N)], fontsize=8)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

barplot(axes[0], PGA_list,  "PGA (max |a|)",         "PGA")
barplot(axes[1], PGV_list,  "PGV (max |v|)",         "PGV")
barplot(axes[2], PGD_list,  "PGD (max |d|)",         "PGD")
barplot(axes[3], IA_list,   "Arias Intensity (rel.)","IA")
barplot(axes[4], CAV_list,  "CAV",                   "CAV")
barplot(axes[5], D595_list, "Duration 5–95% energy", "Time (s)")
barplot(axes[6], Dtot_list, "Total Duration",        "Time (s)")
barplot(axes[7], Tp_list,   "Predominant Period Tp", "Tp (s)")
barplot(axes[8], dt_list,   "Δt for each record",    "dt (s)")

plt.tight_layout()
out_fig = os.path.join(out_dir, "EQ_parameters_comparison.png")
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
plt.show()

print("✅ تحلیل پارامترها تمام شد.")
print("📊 تصویر مقایسه‌ای ذخیره شد در:")
print("   ", out_fig)

# نگاشت اندیس ↔ نام رکورد
index_map_path = os.path.join(out_dir, "EQ_index_map.txt")
with open(index_map_path, "w", encoding="utf-8") as f:
    for i, (name, dt_val) in enumerate(zip(names, dt_list)):
        f.write(f"{i:03d}  {name}   dt={dt_val:.6f}\n")

print("📝 فایل نگاشت اندیس ↔ نام رکورد ذخیره شد در:")
print("   ", index_map_path)
