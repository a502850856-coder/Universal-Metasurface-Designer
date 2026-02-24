from __future__ import annotations
import os
import json
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gdspy

# =========================================================
# 用户参数
# =========================================================

# --- [!! 新增功能 !!] 3-bit 量化开关 ---
# True: 将相位离散化为8个等级，全图只使用8种半径的单元
# False: 使用单元库中所有可用的单元进行最高精度匹配
ENABLE_3BIT_QUANTIZATION = True
# -------------------------------------
# 可生成 15 种焦点类型:
#   point / line / vortex / bessel / off_axis / multi_focus /
#   astigmatic / airy / spiral_multi_focus / bessel_vortex /
#   longitudinal_multi_focus / optical_lattice / flat_top_gs / holographic_gs /
#   longitudinal_vortex
focus_type = "vortex"
lam_nm = 630      # 波长
lam_um = lam_nm * 1e-3
material = "GaN"
nx, ny = 80, 80        # 单元阵列尺寸
pixel_pitch = 240e-9   # 单元间距 (m)
f_um = 25.0            # 焦距

# --- 参数区 (保持不变) ---
xf_um = 10.0
yf_um = 5.0
f_x_um = 30.0
f_y_um = 40.0
airy_alpha = 15.0
vortex_charge = 2
bessel_k_r_over_k = 0.3
f_um_2 = 80.0
lattice_N_x = 5
lattice_N_y = 5
lattice_spacing_um = 4.0
gs_iterations = 100

# =========================================================
# 输出路径
# =========================================================
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
output_dir = os.path.join(desktop, "Metalens_Output_V3_3bit")
os.makedirs(output_dir, exist_ok=True)

# 文件名加上标记以便区分
quant_suffix = "_3bit" if ENABLE_3BIT_QUANTIZATION else "_full"
prefix = os.path.join(
    output_dir, f"{focus_type}_{int(lam_nm)}nm{quant_suffix}")

csv_path = prefix + "_placements.csv"
gds_path = prefix + ".gds"
phase_img_path = prefix + "_phase.png"
focus_img_path = prefix + "_focus.png"
meta_path = prefix + "_meta.json"
unit_lib_path = os.path.join(output_dir, "fake_unit_library.csv")

print("📁 输出目录:", output_dir)

# --- 单元库接口 ---
USE_REAL_LIBRARY = True
real_unit_lib_path = r"C:\Users\50285\Desktop\metalens_GDS_design\630nm_unit_library.csv"

# =========================================================
# 单元库加载
# =========================================================
if USE_REAL_LIBRARY:
    print(f"🔬 正在加载 [真实] 单元库: {real_unit_lib_path}")
    if not os.path.exists(real_unit_lib_path):
        print("❌ 错误：真实单元库文件未找到。")
        exit()
    try:
        df_lib = pd.read_csv(real_unit_lib_path)
        # 确保按相位排序，方便后续处理
        df_lib = df_lib.sort_values(by="phase_rad").reset_index(drop=True)

        phi_lib = df_lib["phase_rad"].values
        amp_lib = df_lib["amplitude"].values
        radius_lib = df_lib["radius_nm"].values

        N_units = len(df_lib)
        radius_min_nm = df_lib["radius_nm"].min()
        radius_max_nm = df_lib["radius_nm"].max()
        print(f"✅ 真实单元库加载成功 (共 {N_units} 个单元)")
    except Exception as e:
        print(f"❌ 错误：加载真实单元库时出错: {e}")
        exit()
else:
    print("🛠️ 正在生成 [虚拟] 单元库...")
    N_units = 64
    radius_min_nm, radius_max_nm = 40, 140
    phi_lib = np.linspace(0, 2*np.pi, N_units, endpoint=False)
    rng = np.random.default_rng(42)
    amp_lib = 0.85 + 0.15*rng.random(N_units)
    radius_lib = np.linspace(radius_min_nm, radius_max_nm, N_units)
    pd.DataFrame({"radius_nm": radius_lib, "amplitude": amp_lib,
                 "phase_rad": phi_lib}).to_csv(unit_lib_path, index=False)
    print("✅ 虚拟单元库已保存")

# =========================================================
# 空间网格
# =========================================================
pitch_um = pixel_pitch * 1e6
x = (np.arange(nx) - (nx - 1)/2)*pitch_um
y = (np.arange(ny) - (ny - 1)/2)*pitch_um
XX, YY = np.meshgrid(x, y, indexing='xy')
XX_m, YY_m = XX*1e-6, YY*1e-6
R_aper_m = (nx / 2.0) * pixel_pitch
R_aper_um = R_aper_m * 1e6
R_grid_m = np.sqrt(XX_m**2 + YY_m**2)
CIRCULAR_APERTURE_MASK = (R_grid_m <= R_aper_m)
k = 2*np.pi / (lam_um*1e-6)

# =========================================================
# 相位函数定义 (保持原样)
# =========================================================


def phase_point(XX_m, YY_m, f_um):
    f_m = f_um*1e-6
    R = np.sqrt(XX_m**2 + YY_m**2 + f_m**2)
    return np.mod(-k*(R - f_m), 2*np.pi)


def phase_line(XX_m, YY_m, f_um):
    f_m = f_um*1e-6
    X = np.abs(XX_m)
    R = np.sqrt(X**2 + f_m**2)
    return np.mod(-k*(R - f_m), 2*np.pi)


def phase_vortex(XX_m, YY_m, f_um, l=1):
    base = phase_point(XX_m, YY_m, f_um)
    theta = np.arctan2(YY_m, XX_m)
    return np.mod(base + l*theta, 2*np.pi)


def phase_bessel(XX_m, YY_m, k_r_over_k=0.3):
    k_r = k_r_over_k * k
    R = np.sqrt(XX_m**2 + YY_m**2)
    return np.mod(k_r*R, 2*np.pi)


def phase_off_axis(XX_m, YY_m, f_um, xf_um, yf_um):
    f_m = f_um * 1e-6
    xf_m = xf_um * 1e-6
    yf_m = yf_um * 1e-6
    R = np.sqrt((XX_m - xf_m)**2 + (YY_m - yf_m)**2 + f_m**2)
    R_center = np.sqrt(xf_m**2 + yf_m**2 + f_m**2)
    return np.mod(-k*(R - R_center), 2*np.pi)


def phase_multi_focus(XX_m, YY_m, f_um):
    phi_1 = phase_off_axis(XX_m, YY_m, f_um, xf_um=10.0, yf_um=5.0)
    phi_2 = phase_off_axis(XX_m, YY_m, f_um, xf_um=-10.0, yf_um=-5.0)
    complex_field = np.exp(1j * phi_1) + np.exp(1j * phi_2)
    return np.mod(np.angle(complex_field), 2*np.pi)


def phase_astigmatic(XX_m, YY_m, f_x_um, f_y_um):
    f_x_m = f_x_um*1e-6
    f_y_m = f_y_um*1e-6
    R_x = np.sqrt(XX_m**2 + f_x_m**2)
    phi_x = -k*(R_x - f_x_m)
    R_y = np.sqrt(YY_m**2 + f_y_m**2)
    phi_y = -k*(R_y - f_y_m)
    return np.mod(phi_x + phi_y, 2*np.pi)


def phase_airy(XX_m, YY_m, R_aper_m, airy_alpha):
    Xn = XX_m / R_aper_m
    Yn = YY_m / R_aper_m
    return np.mod(airy_alpha * (Xn**3 + Yn**3), 2*np.pi)


def phase_spiral_multi_focus(XX_m, YY_m, f_um, l=1):
    phi_1_base = phase_off_axis(XX_m, YY_m, f_um, xf_um=6.0, yf_um=6.0)
    phi_2_base = phase_off_axis(XX_m, YY_m, f_um, xf_um=-6.0, yf_um=-6.0)
    theta = np.arctan2(YY_m, XX_m)
    phi_1 = phi_1_base + l * theta
    phi_2 = phi_2_base - l * theta
    complex_field = np.exp(1j * phi_1) + np.exp(1j * phi_2)
    return np.mod(np.angle(complex_field), 2*np.pi)


def calculate_gs_phase(target_intensity_image, iterations, mask):
    if target_intensity_image.max() > 0:
        target_intensity_image /= target_intensity_image.max()
    target_amplitude = np.sqrt(target_intensity_image)
    lens_phase = np.random.rand(ny, nx) * 2 * np.pi
    for i in range(iterations):
        lens_field = mask * np.exp(1j * lens_phase)
        focal_field = np.fft.fftshift(
            np.fft.fft2(np.fft.ifftshift(lens_field)))
        focal_phase = np.angle(focal_field)
        focal_field_constrained = target_amplitude * np.exp(1j * focal_phase)
        lens_field_new = np.fft.fftshift(np.fft.ifft2(
            np.fft.ifftshift(focal_field_constrained)))
        lens_phase = np.angle(lens_field_new)
    return np.mod(lens_phase, 2*np.pi)


def phase_longitudinal_multi_focus(XX_m, YY_m, f1_um, f2_um):
    phi_1 = phase_point(XX_m, YY_m, f1_um)
    phi_2 = phase_point(XX_m, YY_m, f2_um)
    complex_field = np.exp(1j * phi_1) + np.exp(1j * phi_2)
    return np.mod(np.angle(complex_field), 2*np.pi)


def phase_optical_lattice(XX_m, YY_m, f_um, N_x, N_y, spacing_um):
    complex_field = np.zeros_like(XX_m, dtype=np.complex128)
    x_offsets = (np.arange(N_x) - (N_x - 1) / 2) * spacing_um
    y_offsets = (np.arange(N_y) - (N_y - 1) / 2) * spacing_um
    for xf in x_offsets:
        for yf in y_offsets:
            complex_field += np.exp(1j *
                                    phase_off_axis(XX_m, YY_m, f_um, xf, yf))
    return np.mod(np.angle(complex_field), 2*np.pi)


def phase_flat_top_gs(nx, ny, iterations, mask):
    target_img = np.zeros((ny, nx))
    cx, cy = nx // 2, ny // 2
    size = nx // 16
    target_img[cy - size: cy + size, cx - size: cx + size] = 1.0
    return calculate_gs_phase(target_img, iterations, mask)


def phase_holographic_gs(nx, ny, iterations, mask):
    target_img = np.zeros((ny, nx))
    cx, cy = nx // 2, ny // 2
    sx, sy = nx // 16, ny // 8
    ox = nx // 8
    target_img[cy-sy:cy+sy, cx-ox-sx:cx-ox+sx] = 1.0
    target_img[cy-sy:cy+sy, cx+ox-sx:cx+ox+sx] = 1.0
    return calculate_gs_phase(target_img, iterations, mask)


def phase_longitudinal_vortex(XX_m, YY_m, f1_um, f2_um, l=1):
    phi_1 = phase_vortex(XX_m, YY_m, f1_um, l)
    phi_2 = phase_vortex(XX_m, YY_m, f2_um, l)
    complex_field = np.exp(1j * phi_1) + np.exp(1j * phi_2)
    return np.mod(np.angle(complex_field), 2*np.pi)


# =========================================================
# 计算目标相位 PHI
# =========================================================
print(f"🚀 正在计算 {focus_type} 类型的相位...")
if focus_type == "point":
    PHI = phase_point(XX_m, YY_m, f_um)
elif focus_type == "line":
    PHI = phase_line(XX_m, YY_m, f_um)
elif focus_type == "vortex":
    PHI = phase_vortex(XX_m, YY_m, f_um, l=vortex_charge)
elif focus_type == "bessel":
    PHI = np.mod(phase_point(XX_m, YY_m, f_um) +
                 phase_bessel(XX_m, YY_m, bessel_k_r_over_k), 2*np.pi)
elif focus_type == "off_axis":
    PHI = phase_off_axis(XX_m, YY_m, f_um, xf_um, yf_um)
elif focus_type == "multi_focus":
    PHI = phase_multi_focus(XX_m, YY_m, f_um)
elif focus_type == "astigmatic":
    PHI = phase_astigmatic(XX_m, YY_m, f_x_um, f_y_um)
elif focus_type == "airy":
    PHI = phase_airy(XX_m, YY_m, R_aper_m, airy_alpha)
elif focus_type == "spiral_multi_focus":
    PHI = phase_spiral_multi_focus(XX_m, YY_m, f_um, l=vortex_charge)
elif focus_type == "bessel_vortex":
    PHI = np.mod(phase_point(XX_m, YY_m, f_um) + phase_bessel(XX_m, YY_m,
                 bessel_k_r_over_k) + vortex_charge * np.arctan2(YY_m, XX_m), 2*np.pi)
elif focus_type == "longitudinal_multi_focus":
    PHI = phase_longitudinal_multi_focus(XX_m, YY_m, f_um, f_um_2)
elif focus_type == "optical_lattice":
    PHI = phase_optical_lattice(
        XX_m, YY_m, f_um, lattice_N_x, lattice_N_y, lattice_spacing_um)
elif focus_type == "flat_top_gs":
    PHI = phase_flat_top_gs(nx, ny, gs_iterations, CIRCULAR_APERTURE_MASK)
elif focus_type == "holographic_gs":
    PHI = phase_holographic_gs(nx, ny, gs_iterations, CIRCULAR_APERTURE_MASK)
elif focus_type == "longitudinal_vortex":
    PHI = phase_longitudinal_vortex(XX_m, YY_m, f_um, f_um_2, l=vortex_charge)
else:
    raise ValueError(f"未知的 focus_type: {focus_type}")

print("✅ 目标相位 (PHI) 已生成。")

# =========================================================
# 相位映射与量化 (核心修改部分)
# =========================================================
selected_radius_nm = np.zeros_like(PHI)
selected_phase = np.zeros_like(PHI)
selected_amp = np.zeros_like(PHI)

if ENABLE_3BIT_QUANTIZATION:
    print("🔄 [3-Bit Quantization Mode Enabled]")
    print("   正在从单元库中预选 8 个标准单元...")

    # 1. 定义 8 个理想的相位等级 (0, 45, 90 ... 315 度)
    # 我们希望选出的单元相位最接近这些中心值
    ideal_levels = np.linspace(0, 2*np.pi, 8, endpoint=False)

    # 2. 从库中挑选最匹配这 8 个相位的单元
    quant_radii = []
    quant_real_phases = []
    quant_amps = []

    print("-" * 40)
    print("   Level | Ideal (deg) | Actual (deg) | Radius (nm)")
    print("-" * 40)

    for i, ideal_p in enumerate(ideal_levels):
        # 寻找库中与 ideal_p 差值最小的那个单元的索引
        # 注意处理 0 和 2pi 的循环问题
        diff = np.abs(phi_lib - ideal_p)
        # 处理 2pi 边界 (例如理想是0，库里有 6.28)
        diff = np.minimum(diff, np.abs(phi_lib - (ideal_p + 2*np.pi)))
        diff = np.minimum(diff, np.abs(phi_lib - (ideal_p - 2*np.pi)))

        idx = np.argmin(diff)

        quant_radii.append(radius_lib[idx])
        quant_real_phases.append(phi_lib[idx])
        quant_amps.append(amp_lib[idx])

        print(
            f"   Bit {i} | {np.degrees(ideal_p):5.1f}°      | {np.degrees(phi_lib[idx]):5.1f}°      | {radius_lib[idx]:.1f}")

    print("-" * 40)

    # 转为 numpy 数组以便索引
    quant_radii = np.array(quant_radii)
    quant_real_phases = np.array(quant_real_phases)
    quant_amps = np.array(quant_amps)

    # 3. 将全图目标相位 PHI 量化为 0-7 的整数索引
    # 公式： Index = round( phi / (2pi) * 8 ) % 8
    # 这样可以将相位划分到最近的 bin
    level_indices = np.round(PHI / (2*np.pi) * 8).astype(int) % 8

    # 4. 赋值
    selected_radius_nm = quant_radii[level_indices]
    selected_phase = quant_real_phases[level_indices]
    selected_amp = quant_amps[level_indices]

else:
    print("📍 [Full Library Mapping Mode]")
    print("   Searching full library for nearest neighbors...")
    flat_phi = PHI.flatten()
    tgt_complex = np.exp(1j * flat_phi)[:, None]
    lib_complex = np.exp(1j * phi_lib)[None, :]
    dist = np.abs(tgt_complex - lib_complex)
    sel_idx = np.argmin(dist, axis=1).reshape(PHI.shape)

    selected_radius_nm = radius_lib[sel_idx]
    selected_phase = phi_lib[sel_idx]
    selected_amp = amp_lib[sel_idx]

print("✅ 映射完成。")

# =========================================================
# 保存 CSV
# =========================================================
rows = []
for iy in range(ny):
    for ix in range(nx):
        if not CIRCULAR_APERTURE_MASK[iy, ix]:
            continue
        rows.append({
            "x_um": float(XX[iy, ix]),
            "y_um": float(YY[iy, ix]),
            "phase_target_rad": float(PHI[iy, ix]),
            "mapped_phase_rad": float(selected_phase[iy, ix]),
            "radius_nm": float(selected_radius_nm[iy, ix]),
            "amplitude": float(selected_amp[iy, ix]),
            "quantized_level": int(np.round(PHI[iy, ix] / (2*np.pi) * 8) % 8) if ENABLE_3BIT_QUANTIZATION else -1
        })
pd.DataFrame(rows).to_csv(csv_path, index=False)
print("✅ 已保存 CSV:", csv_path)

# =========================================================
# GDS 版图生成
# =========================================================
lib = gdspy.GdsLibrary(unit=1e-6, precision=1e-9)
cell = lib.new_cell(f"MULTI_{focus_type.upper()}")

print(f"📐 正在生成 GDS (共包含 {len(rows)} 个单元)...")
for r in rows:
    if r["radius_nm"] <= 0:
        continue
    circle = gdspy.Round((r["x_um"], r["y_um"]),
                         r["radius_nm"]/1000.0, number_of_points=64, layer=1)
    cell.add(circle)

lib.write_gds(gds_path)
print("✅ 已保存 GDS:", gds_path)

# =========================================================
# 绘图：相位与半径
# =========================================================
PHI_masked = np.where(CIRCULAR_APERTURE_MASK, PHI, np.nan)
radius_masked = np.where(CIRCULAR_APERTURE_MASK, selected_radius_nm, np.nan)

plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(PHI_masked, cmap="rainbow", origin="lower",
           extent=[x.min(), x.max(), y.min(), y.max()])
plt.title(f"Target Phase ({focus_type})")
plt.xlabel("x (µm)")
plt.ylabel("y (µm)")
plt.colorbar(label="Phase (rad)")

plt.subplot(1, 3, 2)
# 如果是量化的，使用不同的 colormap 方便看清层级
cmap_r = "Blues_r" if ENABLE_3BIT_QUANTIZATION else "viridis"
plt.imshow(radius_masked, cmap=cmap_r, origin="lower",
           extent=[x.min(), x.max(), y.min(), y.max()])
plt.title(
    f"Mapped Radius {'(3-bit)' if ENABLE_3BIT_QUANTIZATION else '(Cont.)'}")
plt.xlabel("x (µm)")
plt.colorbar(label="Radius (nm)")

# 增加一个显示实际相位的图，看看量化后的台阶效果
actual_phase_masked = np.where(CIRCULAR_APERTURE_MASK, selected_phase, np.nan)
plt.subplot(1, 3, 3)
plt.imshow(actual_phase_masked, cmap="rainbow", origin="lower",
           extent=[x.min(), x.max(), y.min(), y.max()])
plt.title("Quantized/Actual Phase")
plt.xlabel("x (µm)")
plt.colorbar(label="Phase (rad)")

plt.tight_layout()
plt.savefig(phase_img_path, dpi=200)
plt.close()
print("✅ 已保存分布图:", phase_img_path)

# =========================================================
# 焦场仿真
# =========================================================
print("🔬 正在计算焦平面光场...")
amp = selected_amp * np.exp(1j * selected_phase)  # 使用量化后的真实相位进行仿真
E = amp * CIRCULAR_APERTURE_MASK.astype(float)
E_fft = np.fft.fftshift(np.fft.fft2(E))
I = np.abs(E_fft)**2
if I.max() > 0:
    I /= I.max()

fx = np.fft.fftfreq(nx, d=pixel_pitch)
fy = np.fft.fftfreq(ny, d=pixel_pitch)
FX, FY = np.meshgrid(np.fft.fftshift(fx), np.fft.fftshift(fy))
Xf_um = FX * lam_um * f_um
Yf_um = FY * lam_um * f_um

plt.figure(figsize=(5, 4))
plt.imshow(I, cmap="inferno", origin="lower", extent=[
           Xf_um.min(), Xf_um.max(), Yf_um.min(), Yf_um.max()])
plt.title(f"{focus_type} Intensity (z={f_um}µm)")
plt.xlabel("x' (µm)")
plt.ylabel("y' (µm)")
plt.colorbar(label="Norm. Intensity")
plt.tight_layout()
plt.savefig(focus_img_path, dpi=250)
plt.close()

# =========================================================
# 元数据
# =========================================================
meta = {
    "focus_type": focus_type,
    "quantization_3bit": ENABLE_3BIT_QUANTIZATION,
    "outputs": {"gds": gds_path, "csv": csv_path}
}
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("\n🎉 完成！")
if ENABLE_3BIT_QUANTIZATION:
    print("模式: [3-Bit 8-Level Quantization]")
    print("注意观察终端输出的 'Bit Table' 确认选取的8个半径是否合理。")
else:
    print("模式: [Full Library Mapping]")
