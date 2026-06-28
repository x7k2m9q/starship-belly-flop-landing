"""
星舰 6-DOF 气动模型 (Phase 7.0)
================================
3DOF→6DOF升级的核心: 添加侧向力(CY)和滚转/偏航力矩(Cl, Cn).

气动系数 (全包线):
  纵向 (3DOF已有):
    CD(α,M) = CD0(M) + CDα(M)·sin²(α)         # 阻力
    CL(α,M) = CLα(M)·sin(2α)·0.5               # 升力
    Cm(α,M) = -Cmα(M)·sin(α - 85°)             # 俯仰力矩 (配平在85°)
  侧向 (6DOF新增):
    CY(β)   = CYβ·sin(β)                        # 侧向力 (β=侧滑角)
    Cl(β)   = CLβ·sin(β)                        # 滚转力矩 (上反效应)
    Cn(β)   = CNβ·sin(β)                        # 偏航力矩 (方向稳定性)

  襟翼导数 (4片独立):
    前左(FL): 俯仰+, 偏航+, 滚转+
    前右(FR): 俯仰+, 偏航-, 滚转-
    后左(RL): 俯仰-, 偏航+, 滚转-
    后右(RR): 俯仰-, 偏航-, 滚转+

物理参数: 复用3DOF的星舰构型参数
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.atmosphere import atmosphere


# =====================================================================
# 物理常数 (复用3DOF, 星舰构型)
# =====================================================================
R_EARTH = 6371000.0
G0_SL = 9.80665
GAMMA_AIR = 1.4
R_AIR = 287.05287

DIAMETER = 9.0
L_REF = 50.0
S_REF = np.pi * (DIAMETER / 2.0) ** 2   # ≈ 63.617 m²
M_DRY = 100000.0
M_FUEL_INIT = 50000.0
M_TOTAL_INIT = M_DRY + M_FUEL_INIT

T_MAX = 4600e3
T_IDLE = 460e3
ISP = 380.0
G0_ISP = 9.80665

DELTA_MAX = np.deg2rad(30.0)

# 襟翼效率系数
C_DELTA_FWD = 0.8   # 前翼
C_DELTA_AFT = 0.6   # 后翼

# =====================================================================
# 侧向气动导数 (6DOF新增 — 工程估算)
# =====================================================================
# 基于星舰气动布局公开资料估算
# CYβ: 侧向力对侧滑角导数 (负值=侧滑产生反向侧力, 方向稳定性)
CYB = -1.2     # 侧向力导数 (rad^-1)
# CLβ: 滚转力矩对侧滑角导数 (上反效应, 负值=侧滑产生恢复滚转)
CLB = -0.3     # 滚转力矩导数 (rad^-1)
# CNβ: 偏航力矩对侧滑角导数 (正值=方向稳定, weathervane效应)
CNB = 0.15     # 偏航力矩导数 (rad^-1)

# 襟翼侧向导数 (差动偏转产生滚转/偏航)
C_DELTA_ROLL = 0.5    # 滚转力矩效率 (差动襟翼)
C_DELTA_YAW = 0.3     # 偏航力矩效率 (差动襟翼)

# 转动惯量基准
IXX_DRY = (1.0 / 12.0) * M_DRY * (DIAMETER / 2.0) ** 2 * 2.0  # 滚转(绕Xb)
IYY_DRY = (1.0 / 12.0) * M_DRY * L_REF ** 2                    # 俯仰(绕Yb)
IZZ_DRY = IYY_DRY                                               # 偏航(绕Zb) ≈ 俯仰
L_FUEL = L_REF / 3.0
IXX_FUEL_INIT = (1.0 / 12.0) * M_FUEL_INIT * (DIAMETER / 2.0) ** 2 * 2.0
IYY_FUEL_INIT = (1.0 / 12.0) * M_FUEL_INIT * L_FUEL ** 2
IZZ_FUEL_INIT = IYY_FUEL_INIT


def get_inertia_tensor(m_fuel):
    """
    当前转动惯量张量 (3x3对角阵).

    Ixx = 滚转 (绕Xb, 火箭纵轴)
    Iyy = 俯仰 (绕Yb)
    Izz = 偏航 (绕Zb) ≈ Iyy (轴对称)

    燃料消耗: Iyy/Izz随燃料减少而减小, Ixx变化较小.
    """
    frac = max(m_fuel / M_FUEL_INIT, 0.0)
    Ixx = IXX_DRY + frac * IXX_FUEL_INIT
    Iyy = IYY_DRY + frac * IYY_FUEL_INIT
    Izz = IZZ_DRY + frac * IZZ_FUEL_INIT
    return np.diag([Ixx, Iyy, Izz])


def get_mass(m_fuel):
    """当前总质量."""
    return M_DRY + max(m_fuel, 0.0)


def gravity(h):
    """随高度变化的重力."""
    return G0_SL * (R_EARTH / (R_EARTH + h)) ** 2


def atmosphere_6dof(h):
    """大气模型接口 (复用1976标准大气)."""
    rho, a_sound, p, T = atmosphere(h)
    return rho, a_sound, p, T


# =====================================================================
# Mach sigmoid 平滑 (复用3DOF)
# =====================================================================
def mach_sigmoid_weights(M):
    """三层 Mach 加权: 亚→跨→超."""
    arg_trans = np.clip(-20.0 * (M - 1.0), -500.0, 500.0)
    arg_super = np.clip(-20.0 * (M - 1.5), -500.0, 500.0)
    w_trans = 1.0 / (1.0 + np.exp(arg_trans))
    w_super = 1.0 / (1.0 + np.exp(arg_super))
    return w_trans, w_super


def aero_coefficients(M):
    """
    返回 (CD0, CDa, CLa, Cma) 随 Mach 平滑变化.
    复用3DOF参数.
    """
    w_trans, w_super = mach_sigmoid_weights(M)
    CD0 = 0.3 * (1 - w_trans) + 0.8 * w_trans * (1 - w_super) + 0.5 * w_super
    CDa = 1.5 * (1 - w_trans) + 2.5 * w_trans * (1 - w_super) + 2.0 * w_super
    CLa = 1.2 * (1 - w_trans) + 0.8 * w_trans * (1 - w_super) + 0.6 * w_super
    Cma = 0.15 * (1 - w_trans) + 0.25 * w_trans * (1 - w_super) + 0.20 * w_super
    return CD0, CDa, CLa, Cma


# =====================================================================
# 气流角计算 (6DOF)
# =====================================================================
def airflow_angles_6dof(vel_b):
    """
    从body系速度计算攻角α和侧滑角β.

    vel_b = [u, v, w] (body系)
      u: 前向速度 (沿Xb)
      v: 侧向速度 (沿Yb)
      w: 垂直速度 (沿Zb)

    攻角: α = atan2(w, u)  (w>0时α>0, 尾部迎风)
    侧滑角: β = asin(v / V)  (v>0时β>0, 右侧滑)

    返回: (alpha, beta, V)
    """
    u, v, w = vel_b
    V = np.sqrt(u * u + v * v + w * w)
    if V < 1e-6:
        return 0.0, 0.0, 0.0
    alpha = np.arctan2(w, u)
    beta = np.arcsin(np.clip(v / V, -1.0, 1.0))
    return alpha, beta, V


# =====================================================================
# 6DOF 气动力/力矩计算
# =====================================================================
def aero_forces_and_moments_6dof(vel_b, q_quat, h, delta_flaps):
    """
    6-DOF 气动力和力矩计算.

    参数:
      vel_b: body系速度 [u, v, w] (m/s)
      q_quat: 四元数 [w,x,y,z] (未使用, 气动力在body系计算)
      h: 高度 (m)
      delta_flaps: [d_FL, d_FR, d_RL, d_RR] 4片襟翼偏转 (rad)

    返回:
      F_aero_b: body系气动力 [Fx, Fy, Fz] (N)
      M_aero_b: body系气动力矩 [Mx, My, Mz] (N·m)

    坐标系 (body系):
      Fx: 沿Xb(头部方向), 阻力主要分量
      Fy: 沿Yb(右), 侧向力
      Fz: 沿Zb, 升力主要分量
      Mx: 滚转力矩 (绕Xb)
      My: 俯仰力矩 (绕Yb)
      Mz: 偏航力矩 (绕Zb)

    襟翼分配 (4片):
      FL(前左): 俯仰+, 偏航+, 滚转+
      FR(前右): 俯仰+, 偏航-, 滚转-
      RL(后左): 俯仰-, 偏航+, 滚转-
      RR(后右): 俯仰-, 偏航-, 滚转+

      俯仰: (FL+FR)·Cδf - (RL+RR)·Cδa  (前翼抬头+, 后翼低头-)
      偏航: (FL+RL)·Cδ_yaw - (FR+RR)·Cδ_yaw  (左侧偏航+, 右侧偏航-)
      滚转: (FL+RR)·Cδ_roll - (FR+RL)·Cδ_roll  (对角线差动)
    """
    u, v, w = vel_b
    V = np.sqrt(u * u + v * v + w * w)

    if V < 1e-6 or h < 0:
        return np.zeros(3), np.zeros(3)

    # 大气
    rho, a_sound, p, T = atmosphere_6dof(h)
    M = V / a_sound if a_sound > 0 else 0.0

    # 气流角
    alpha, beta, _ = airflow_angles_6dof(vel_b)

    # 气动系数
    CD0, CDa, CLa, Cma = aero_coefficients(M)
    CD = CD0 + CDa * np.sin(alpha) ** 2
    CL = CLa * np.sin(2.0 * alpha) * 0.5
    Cm = -Cma * np.sin(alpha - np.deg2rad(85.0))

    # 侧向气动 (6DOF新增)
    CY = CYB * np.sin(beta)
    Cl = CLB * np.sin(beta)   # 滚转力矩系数
    Cn = CNB * np.sin(beta)   # 偏航力矩系数

    # 动压
    Q = 0.5 * rho * V ** 2 * S_REF

    # ---- 气动力 (body系) ----
    # 阻力: 沿 -V_hat (body系)
    # 升力: 垂直V, 在Xb-Zb平面内
    # 侧力: 沿Yb

    # body系气动力:
    # Fx = -D·cos(α) + L·sin(α)  (前向, α=0时纯阻力)
    # Fz = -D·sin(α) - L·cos(α)  (沿Zb, α=0时升力沿-Zb)
    # Fy = Q·S·CY                 (侧向)
    Fx = Q * (-CD * np.cos(alpha) + CL * np.sin(alpha))
    Fy = Q * CY
    Fz = Q * (-CD * np.sin(alpha) - CL * np.cos(alpha))
    F_aero_b = np.array([Fx, Fy, Fz])

    # ---- 襟翼力矩 (4片差动) ----
    d_FL, d_FR, d_RL, d_RR = delta_flaps

    # 俯仰力矩: 前翼(FL+FR)·Cδf - 后翼(RL+RR)·Cδa
    # 正偏转=抬头(增大alpha)
    M_flap_pitch = Q * L_REF * (
        C_DELTA_FWD * (d_FL + d_FR) - C_DELTA_AFT * (d_RL + d_RR)
    )

    # 偏航力矩: 左侧(FL+RL) - 右侧(FR+RR), 差动偏转
    M_flap_yaw = Q * L_REF * C_DELTA_YAW * (
        (d_FL + d_RL) - (d_FR + d_RR)
    )

    # 滚转力矩: 对角线差动 (FL+RR) - (FR+RL)
    M_flap_roll = Q * L_REF * C_DELTA_ROLL * (
        (d_FL + d_RR) - (d_FR + d_RL)
    )

    # ---- 总气动力矩 (body系) ----
    # 俯仰: 气动 + 襟翼
    My = Q * L_REF * Cm + M_flap_pitch
    # 偏航: 气动 + 襟翼
    Mz = Q * L_REF * Cn + M_flap_yaw
    # 滚转: 气动 + 襟翼
    Mx = Q * L_REF * Cl + M_flap_roll

    M_aero_b = np.array([Mx, My, Mz])

    return F_aero_b, M_aero_b


# =====================================================================
# 6DOF 配平襟翼 (4片)
# =====================================================================
def trim_flaps_6dof(alpha, mach):
    """
    6DOF配平襟翼: 求解4片襟翼使M_total=0 (俯仰).

    简化: 前后翼等比偏转 (δ_fwd = δ_aft = δ), 侧向/滚转为0.
    与3DOF配平一致, 但输出4片.

    返回: [d_FL, d_FR, d_RL, d_RR] (rad)
    """
    CD0, CDa, CLa, Cma = aero_coefficients(mach)
    Cm = -Cma * np.sin(alpha - np.deg2rad(85.0))

    # 等力矩分配: δ_fwd = -Cm/(2·Cδf), δ_aft = -Cm/(2·Cδa)
    d_fwd = -Cm / (2.0 * C_DELTA_FWD)
    d_aft = -Cm / (2.0 * C_DELTA_AFT)

    d_fwd = np.clip(d_fwd, -DELTA_MAX, DELTA_MAX)
    d_aft = np.clip(d_aft, -DELTA_MAX, DELTA_MAX)

    # 4片: 前左=前右=δ_fwd, 后左=后右=δ_aft
    return np.array([d_fwd, d_fwd, d_aft, d_aft])
