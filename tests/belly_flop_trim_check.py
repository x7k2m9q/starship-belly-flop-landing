"""
Step 7A 配平查表关键点验证.
检查 α×Mach 网格上几个关键点的配平襟翼角是否物理合理.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from src.belly_flop.aero_model import (
    TrimTable, trim_flaps, aero_coefficients, DELTA_MAX,
    C_DELTA_FWD, C_DELTA_AFT,
)

print("=" * 60)
print("配平查表关键点验证")
print("=" * 60)

tt = TrimTable()
print(f"网格: alpha {tt.alpha_deg[0]}-{tt.alpha_deg[-1]}° (步长{tt.alpha_deg[1]-tt.alpha_deg[0]}°), "
      f"Mach {tt.mach[0]}-{tt.mach[-1]} (步长{tt.mach[1]-tt.mach[0]})")
print(f"网格大小: {tt.n_alpha} x {tt.n_mach}")

# 关键点1: α=85°(配平点), Mach=0.5 (亚声速)
# 期望: Cm(85°)=-Cma·sin(0)=0, δ_fwd=δ_aft=0
print("\n[关键点1] α=85°(配平点), Mach=0.5")
alpha = np.deg2rad(85.0)
df, da = trim_flaps(alpha, 0.5)
CD0, CDa, CLa, Cma = aero_coefficients(0.5)
Cm = -Cma * np.sin(alpha - np.deg2rad(85.0))
print(f"  Cm(85°,0.5) = {Cm:.6f} (期望~0)")
print(f"  δ_fwd = {np.degrees(df):.3f}°, δ_aft = {np.degrees(da):.3f}° (期望~0°)")
print(f"  状态: {'PASS' if abs(Cm) < 1e-6 and abs(df) < 1e-3 else 'CHECK'}")

# 关键点2: α=45°, Mach=0.5 (偏离配平点)
# Cm(45°)=-Cma·sin(45°-85°)=-Cma·sin(-40°)=+Cma·sin(40°)>0
# δ_fwd = -Cm/(2·Cδf) < 0, δ_aft = -Cm/(2·Cδa) < 0
print("\n[关键点2] α=45°(偏离配平), Mach=0.5")
alpha = np.deg2rad(45.0)
df, da = trim_flaps(alpha, 0.5)
CD0, CDa, CLa, Cma = aero_coefficients(0.5)
Cm = -Cma * np.sin(alpha - np.deg2rad(85.0))
df_expect = -Cm / (2.0 * C_DELTA_FWD)
da_expect = -Cm / (2.0 * C_DELTA_AFT)
print(f"  Cm(45°,0.5) = {Cm:.6f} (期望>0, 因为sin(-40°)<0, -Cma·负=正)")
print(f"  δ_fwd = {np.degrees(df):.3f}° (期望 {np.degrees(df_expect):.3f}°)")
print(f"  δ_aft = {np.degrees(da):.3f}° (期望 {np.degrees(da_expect):.3f}°)")
print(f"  |δ_fwd|<|δ_aft| (因为Cδf>Cδa, 等力矩分配): {'PASS' if abs(df) < abs(da) else 'FAIL'}")

# 关键点3: α=0°, Mach=2.0 (超声速, 远离配平点)
# Cm(0°)=-Cma·sin(-85°)=+Cma·sin(85°)≈+Cma (大正值)
# δ_fwd = -Cm/(2·Cδf) 大负值, 可能饱和到-30°
print("\n[关键点3] α=0°(远离配平), Mach=2.0 (超声速)")
alpha = np.deg2rad(0.0)
df, da = trim_flaps(alpha, 2.0)
CD0, CDa, CLa, Cma = aero_coefficients(2.0)
Cm = -Cma * np.sin(alpha - np.deg2rad(85.0))
df_raw = -Cm / (2.0 * C_DELTA_FWD)
da_raw = -Cm / (2.0 * C_DELTA_AFT)
print(f"  Cm(0°,2.0) = {Cm:.6f} (期望大正值)")
print(f"  δ_fwd_raw = {np.degrees(df_raw):.3f}° (未钳位)")
print(f"  δ_aft_raw = {np.degrees(da_raw):.3f}° (未钳位)")
print(f"  δ_fwd = {np.degrees(df):.3f}° (钳位后, 期望=-30°)")
print(f"  δ_aft = {np.degrees(da):.3f}° (钳位后, 期望=-30°)")
saturated = abs(np.degrees(df) - (-30.0)) < 0.1 and abs(np.degrees(da) - (-30.0)) < 0.1
print(f"  饱和检查: {'PASS (两端都饱和到-30°)' if saturated else 'CHECK'}")

# 关键点4: 双线性插值连续性
# 查询 α=82.5°(网格中点), Mach=0.6(网格中点), 应为周围4点平均
print("\n[关键点4] 双线性插值连续性 (α=82.5°, Mach=0.6)")
alpha = np.deg2rad(82.5)
df_mid, da_mid = trim_flaps(alpha, 0.6)
df_80_05, _ = trim_flaps(np.deg2rad(80.0), 0.5)
df_85_05, _ = trim_flaps(np.deg2rad(85.0), 0.5)
df_80_06, _ = trim_flaps(np.deg2rad(80.0), 0.6)
df_85_06, _ = trim_flaps(np.deg2rad(85.0), 0.6)
# 82.5°在80-85°中点, 0.6在0.4-0.6... 实际0.6是网格点
# α=82.5°: di=0.5, Mach=0.6: j_f=3.0, j0=2, dj=1.0
# 所以是 α=80°和85°在Mach=0.6的平均
df_expect = 0.5 * df_80_06 + 0.5 * df_85_06
print(f"  δ_fwd(82.5°,0.6) = {np.degrees(df_mid):.4f}°")
print(f"  期望(0.5·δ(80°,0.6)+0.5·δ(85°,0.6)) = {np.degrees(df_expect):.4f}°")
print(f"  插值连续性: {'PASS' if abs(df_mid - df_expect) < 1e-6 else 'FAIL'}")

# 关键点5: 配平表整体范围
print("\n[关键点5] 配平表整体范围")
print(f"  δ_fwd 范围: [{np.degrees(tt.delta_fwd_table.min()):.2f}°, {np.degrees(tt.delta_fwd_table.max()):.2f}°]")
print(f"  δ_aft 范围: [{np.degrees(tt.delta_aft_table.min()):.2f}°, {np.degrees(tt.delta_aft_table.max()):.2f}°]")
print(f"  钳位限制: ±{np.degrees(DELTA_MAX):.1f}°")
in_range = (tt.delta_fwd_table.min() >= -DELTA_MAX and tt.delta_fwd_table.max() <= DELTA_MAX and
            tt.delta_aft_table.min() >= -DELTA_MAX and tt.delta_aft_table.max() <= DELTA_MAX)
print(f"  范围检查: {'PASS (全部在±30°内)' if in_range else 'FAIL'}")

# 关键点6: α=85°行全为0 (配平点)
print("\n[关键点6] α=85°行 (配平点, 全Mach应δ≈0)")
idx_85 = list(tt.alpha_deg).index(85.0)
row_fwd = tt.delta_fwd_table[idx_85, :]
row_aft = tt.delta_aft_table[idx_85, :]
print(f"  δ_fwd(85°,all Mach): max|δ| = {np.degrees(np.max(np.abs(row_fwd))):.6f}°")
print(f"  δ_aft(85°,all Mach): max|δ| = {np.degrees(np.max(np.abs(row_aft))):.6f}°")
all_zero = np.max(np.abs(row_fwd)) < 1e-10 and np.max(np.abs(row_aft)) < 1e-10
print(f"  配平点零力矩: {'PASS' if all_zero else 'CHECK'}")

print("\n" + "=" * 60)
print("配平查表验证完成")
print("=" * 60)
