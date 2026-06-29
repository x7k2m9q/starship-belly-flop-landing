"""
Belly-Flop 气动模型 (星舰简化构型).
====================================
严格遵循气动减速复刻方案1.0 Step 7A.

攻角定义 (搞反就全错):
    α = θ - γ,  γ = atan2(vx, vz)
    θ=0°  (火箭垂直, 头朝上), γ=0° (垂直下落) → α=0°  (顺气流, 最小阻力)
    θ=90° (火箭水平, 腹部朝下), γ=0° (垂直下落) → α=90° (腹部迎风, 最大阻力)

气动系数 (含 Mach sigmoid 平滑, 禁止 if-else 分段):
    CD(α,M) = CD0(M) + CDα(M)·sin²(α)
    CL(α,M) = CLα(M)·sin(2α)·0.5        # 0.5=大攻角失速衰减
    Cm(α)   = -Cmα(M)·sin(α - 85°)      # 自然配平在 α=85°

物理参数 (星舰简化, 严禁复用前项目常量):
    S_ref  = π·4.5² ≈ 63.6 m²           # 参考面积 (直径9m)
    L_ref  = 50 m                        # 参考长度 (星舰高度)
    m_dry  = 100000 kg                   # 干重
    m_fuel = 50000 kg                    # 燃料
    T_max  = 4600 kN                     # 最大推力 (3台猛禽)
    T_idle = 460 kN                      # 怠速推力 (10%)
    Cδf    = 0.8                         # 前翼效率系数
    Cδa    = 0.6                         # 后翼效率系数
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.atmosphere import atmosphere  # 复用1976标准大气 (允许)


# =====================================================================
# 物理常数 (星舰构型, 与前项目完全不同)
# =====================================================================
R_EARTH = 6371000.0          # m, 地球半径
G0_SL = 9.80665              # m/s², 海平面重力
GAMMA_AIR = 1.4              # 空气比热比
R_AIR = 287.05287            # J/(kg·K), 气体常数

# 星舰几何/质量参数
DIAMETER = 9.0               # m, 直径
L_REF = 50.0                 # m, 参考长度 (星舰高度)
S_REF = np.pi * (DIAMETER / 2.0) ** 2   # m², 参考面积 ≈ 63.617
M_DRY = 100000.0             # kg, 干重
M_FUEL_INIT = 50000.0        # kg, 初始燃料
M_TOTAL_INIT = M_DRY + M_FUEL_INIT  # 150000 kg

# 推力参数
T_MAX = 4600e3               # N, 最大推力 (3台猛禽)
T_IDLE = 460e3               # N, 怠速推力 (10%)
ISP = 380.0                  # s, 比冲
G0_ISP = 9.80665             # Isp参考重力

# 襟翼效率
C_DELTA_FWD = 0.8            # 前翼效率系数
C_DELTA_AFT = 0.6            # 后翼效率系数
DELTA_MAX = np.deg2rad(30.0) # 襟翼最大偏转角 ±30°

# 转动惯量基准 (随燃料消耗更新)
IYY_DRY = (1.0 / 12.0) * M_DRY * L_REF ** 2          # 干重转动惯量
L_FUEL = L_REF / 3.0                                   # 燃料分布在底部1/3
IYY_FUEL_INIT = (1.0 / 12.0) * M_FUEL_INIT * L_FUEL ** 2  # 初始燃料转动惯量


def get_Iyy(m_fuel):
    """当前转动惯量 (随燃料消耗更新). Iyy = Iyy_dry + (m_fuel/m_fuel_init)·Iyy_fuel."""
    return IYY_DRY + (m_fuel / M_FUEL_INIT) * IYY_FUEL_INIT


def get_mass(m_fuel):
    """当前总质量."""
    return M_DRY + m_fuel


def gravity(h):
    """随高度变化的重力. g = 9.80665·(R_earth/(R_earth+h))²."""
    return G0_SL * (R_EARTH / (R_EARTH + h)) ** 2


# =====================================================================
# Mach sigmoid 平滑 (禁止 if-else 分段)
# =====================================================================
def mach_sigmoid_weights(M):
    """
    返回 (w_trans, w_super) 三层 Mach 加权.
      w_trans = 1/(1+exp(-20*(M-1.0)))   # 亚→跨 过渡 (Mach=1)
      w_super = 1/(1+exp(-20*(M-1.5)))   # 跨→超 过渡 (Mach=1.5)
    数值截断防溢出.
    """
    # sigmoid 数值稳定截断
    arg_trans = np.clip(-20.0 * (M - 1.0), -500.0, 500.0)
    arg_super = np.clip(-20.0 * (M - 1.5), -500.0, 500.0)
    w_trans = 1.0 / (1.0 + np.exp(arg_trans))
    w_super = 1.0 / (1.0 + np.exp(arg_super))
    return w_trans, w_super


def aero_coefficients(M):
    """
    返回 (CD0, CDa, CLa, Cma) 随 Mach 平滑变化.
    亚声速 (M<0.8):  CD0=0.3, CDα=1.5, CLα=1.2, Cmα=0.15
    跨声速 (0.8<M<1.2): CD0=0.8, CDα=2.5, CLα=0.8, Cmα=0.25
    超声速 (M>1.2):  CD0=0.5, CDα=2.0, CLα=0.6, Cmα=0.20
    """
    w_trans, w_super = mach_sigmoid_weights(M)
    # 三层加权: 亚*(1-w_trans) + 跨*w_trans*(1-w_super) + 超*w_super
    CD0 = 0.3 * (1 - w_trans) + 0.8 * w_trans * (1 - w_super) + 0.5 * w_super
    CDa = 1.5 * (1 - w_trans) + 2.5 * w_trans * (1 - w_super) + 2.0 * w_super
    CLa = 1.2 * (1 - w_trans) + 0.8 * w_trans * (1 - w_super) + 0.6 * w_super
    Cma = 0.15 * (1 - w_trans) + 0.25 * w_trans * (1 - w_super) + 0.20 * w_super
    return CD0, CDa, CLa, Cma


# =====================================================================
# 气动力/力矩计算
# =====================================================================
def angle_of_attack(theta, vx, vz):
    """
    攻角 α = θ - γ, γ = atan2(vx, vz).
    严格遵循方案定义:
      θ=0°(垂直)+γ=0°(垂直下落) → α=0°(顺气流)
      θ=90°(水平)+γ=0°(垂直下落) → α=90°(腹部迎风)
    γ=atan2(vx,vz): vx>0,vz>0(斜向下) → γ∈(0,π/2), 速度方向与垂直轴夹角.
    """
    gamma = np.arctan2(vx, vz)
    alpha = theta - gamma
    # 归一化到 [-π, π]
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
    return alpha, gamma


def aero_forces_and_moments(vx, vz, theta, h, delta_fwd=0.0, delta_aft=0.0):
    """
    计算气动力 (阻力D, 升力L) 和气动力矩 M_aero, M_flap.

    返回: (D, L, Fx_aero, Fz_aero, M_aero, M_flap, M_total, Q, alpha, gamma, M, rho, a_sound)

    坐标系 (2D NED-like):
      x: 水平 (下程方向)
      h: 高度 (向上为正)
      vx: 水平速度
      vz: 垂直速度 (正向下)
      θ: 俯仰角 (0=垂直头朝上, π/2=水平腹部朝下)
      q: 俯仰角速度

    气动力坐标变换 (速度系→x-z frame):
      γ = atan2(vx, vz) 是从垂直轴(vz)到速度向量的夹角
      速度单位向量: V_hat = (sin(γ), cos(γ))  [在(x, z)系, z向下]
      阻力沿 -V_hat: D_vec = (-D·sin(γ), -D·cos(γ))
      升力垂直速度, 指向减小γ方向: L_vec = (L·cos(γ), -L·sin(γ))
      Fx_aero = -D·sin(γ) + L·cos(γ)    # 水平
      Fz_aero = -D·cos(γ) - L·sin(γ)    # 垂直(正向下)

      验证: γ=0(垂直下落) → Fx=L, Fz=-D (阻力向上✓, 升力水平✓)
      验证: γ=π/2(水平) → Fx=-D, Fz=-L (阻力水平✓, 升力向上✓)
    """
    # 大气
    rho, a_sound, p, T = atmosphere(h)

    V = np.sqrt(vx ** 2 + vz ** 2)
    if V < 1e-6:
        # 零速度, 无气动力
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, rho, a_sound)

    M = V / a_sound
    alpha, gamma = angle_of_attack(theta, vx, vz)

    # 气动系数
    CD0, CDa, CLa, Cma = aero_coefficients(M)
    CD = CD0 + CDa * np.sin(alpha) ** 2
    CL = CLa * np.sin(2.0 * alpha) * 0.5
    Cm = -Cma * np.sin(alpha - np.deg2rad(85.0))

    # 动压
    Q = 0.5 * rho * V ** 2 * S_REF

    # 气动力
    D = Q * CD
    L = Q * CL

    # 气动力坐标变换 (γ=atan2(vx,vz), 从垂直轴算)
    # 阻力: -D·V_hat = (-D·sin(γ), -D·cos(γ))
    # 升力: L·(cos(γ), -sin(γ))  [垂直速度, 指向减小γ方向]
    Fx_aero = -D * np.sin(gamma) + L * np.cos(gamma)
    Fz_aero = -D * np.cos(gamma) - L * np.sin(gamma)

    # 气动力矩 (Phase 9.0修复: Q已含S_REF, 不再重复乘)
    # Bug: 原代码 M_aero = Q * S_REF * L_REF * Cm, S_REF被乘了两次, 力矩放大27倍
    # 修复: M_aero = Q * L_REF * Cm (= 0.5*rho*V²*S_REF*L_REF*Cm)
    M_aero = Q * L_REF * Cm
    M_flap = Q * L_REF * (C_DELTA_FWD * delta_fwd + C_DELTA_AFT * delta_aft)
    M_total = M_aero + M_flap

    return (D, L, Fx_aero, Fz_aero, M_aero, M_flap, M_total, Q,
            alpha, gamma, M, rho, a_sound)


# =====================================================================
# 配平襟翼查表 (离线生成, 运行时双线性插值)
# =====================================================================
class TrimTable:
    """
    离线扫描 α(0-90°) × Mach(0-3) 网格, 求解 M_total=0 时的 [δ_fwd, δ_aft].
    运行时双线性插值. 严禁 scipy.optimize 在线求解.

    配平条件: M_total = Q·S·L·[Cm(α,M) + Cδf·δ_fwd + Cδa·δ_aft] = 0
    简化: 前后翼等比偏转 (δ_fwd = δ_aft = δ), 则
      Cδf·δ + Cδa·δ = -Cm  →  δ = -Cm / (Cδf + Cδa)
    实际: δ_fwd = -Cm/(2·Cδf), δ_aft = -Cm/(2·Cδa)  (等力矩分配)
    钳位: |δ_fwd| ≤ 30°, |δ_aft| ≤ 30°
    """
    def __init__(self):
        self.alpha_deg = np.arange(0.0, 91.0, 5.0)       # 0-90°, 步长5°
        self.mach = np.arange(0.0, 3.01, 0.2)            # 0-3, 步长0.2
        self.n_alpha = len(self.alpha_deg)
        self.n_mach = len(self.mach)
        self.delta_fwd_table = np.zeros((self.n_alpha, self.n_mach))
        self.delta_aft_table = np.zeros((self.n_alpha, self.n_mach))
        self._build_table()

    def _build_table(self):
        """离线生成配平查表."""
        for i, a_deg in enumerate(self.alpha_deg):
            alpha = np.deg2rad(a_deg)
            for j, m in enumerate(self.mach):
                CD0, CDa, CLa, Cma = aero_coefficients(m)
                Cm = -Cma * np.sin(alpha - np.deg2rad(85.0))
                # 等力矩分配: δ_fwd = -Cm/(2·Cδf), δ_aft = -Cm/(2·Cδa)
                d_fwd = -Cm / (2.0 * C_DELTA_FWD)
                d_aft = -Cm / (2.0 * C_DELTA_AFT)
                # 钳位
                self.delta_fwd_table[i, j] = np.clip(d_fwd, -DELTA_MAX, DELTA_MAX)
                self.delta_aft_table[i, j] = np.clip(d_aft, -DELTA_MAX, DELTA_MAX)

    def lookup(self, alpha, mach):
        """
        双线性插值查询配平襟翼角.
        alpha: rad, mach: 无量纲
        返回: (delta_fwd, delta_aft) rad
        """
        # alpha 转度数并钳位
        a_deg = np.degrees(alpha)
        a_deg = np.clip(a_deg, 0.0, 90.0)
        # mach 钳位
        m = np.clip(mach, 0.0, 3.0)

        # 查找索引
        i_f = a_deg / 5.0
        j_f = m / 0.2
        i0 = int(np.clip(np.floor(i_f), 0, self.n_alpha - 2))
        j0 = int(np.clip(np.floor(j_f), 0, self.n_mach - 2))
        di = i_f - i0
        dj = j_f - j0

        # 双线性插值
        d_fwd = (self.delta_fwd_table[i0, j0] * (1 - di) * (1 - dj) +
                 self.delta_fwd_table[i0 + 1, j0] * di * (1 - dj) +
                 self.delta_fwd_table[i0, j0 + 1] * (1 - di) * dj +
                 self.delta_fwd_table[i0 + 1, j0 + 1] * di * dj)
        d_aft = (self.delta_aft_table[i0, j0] * (1 - di) * (1 - dj) +
                 self.delta_aft_table[i0 + 1, j0] * di * (1 - dj) +
                 self.delta_aft_table[i0, j0 + 1] * (1 - di) * dj +
                 self.delta_aft_table[i0 + 1, j0 + 1] * di * dj)

        return d_fwd, d_aft


# 全局配平表 (首次访问时构建)
_TRIM_TABLE = None

def get_trim_table():
    """获取全局配平表 (懒加载)."""
    global _TRIM_TABLE
    if _TRIM_TABLE is None:
        _TRIM_TABLE = TrimTable()
    return _TRIM_TABLE


def trim_flaps(alpha, mach):
    """查询配平襟翼角 (便捷接口)."""
    return get_trim_table().lookup(alpha, mach)
