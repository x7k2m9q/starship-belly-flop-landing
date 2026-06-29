"""
Belly-Flop Step 7D: 翻转段 bang-bang + PD + 前馈补偿.
=====================================================
理论方案 9.0-Final Step 7D.

缺陷21: bang-bang切换时间试凑 → t_switch=sqrt(Iyy·Δθ/M_max) 解析
缺陷22: 翻转中纯解析模型 → 数值积分前馈+PD反馈+前馈力矩补偿

控制策略 (三项叠加, 全部转换为力矩再除以襟翼效率):
  1. bang-bang 期望轨迹: θ_ref(t) 梯形速度剖面
     - 0 < t < t_switch: 匀加速翻转 (θ̈_ref = -α_max)
     - t_switch < t < t_total: 匀减速翻转 (θ̈_ref = +α_max)
     - t_switch = t_target/2, α_max = |Δθ|/t_switch²

  2. 标称前馈力矩 (缺陷22核心: 数值积分前馈):
     - M_nominal = Iyy · θ̈_ref
     - 这是驱动火箭按bang-bang加速度运动所需的力矩
     - 缺失此项 → PD误差极小(bang-bang起步慢) → 火箭不翻转 → trim失衡 → 失控

  3. PD反馈力矩 (跟踪bang-bang轨迹):
     - M_pd = Iyy · (ωn²·e_θ + 2ζωn·e_q)
     - e_θ = θ_ref - θ (归一化到[-π,π])
     - e_q = θ̇_ref - q
     - ωn=3.0 rad/s, ζ=0.7 (带宽>2.36, 稳态误差<5°)

  4. 前馈力矩补偿 (补偿气动力矩残余):
     - trim在α_ref处配平, 实际α偏离α_ref时有残余
     - M_residual = M_aero(α) - M_aero(α_ref)
     - M_ff = -M_residual

  5. 总额外襟翼偏转:
     - M_extra = M_nominal + M_pd + M_ff
     - δ_extra = M_extra / (Q·S·L·(Cδf+Cδa))
     - 前后翼等偏: δ_fwd = δ_aft = δ_extra (最大力矩效率)

Kill Criteria: 翻转超时 > 8s
"""
import numpy as np
from .aero_model import (
    aero_coefficients, angle_of_attack, aero_forces_and_moments,
    atmosphere, get_mass, get_Iyy, gravity, trim_flaps,
    S_REF, L_REF, T_MAX, T_IDLE, ISP, G0_ISP,
    M_FUEL_INIT, M_DRY,
    C_DELTA_FWD, C_DELTA_AFT, DELTA_MAX,
)


# =====================================================================
# 翻转段参数
# =====================================================================
THETA_BELLY = np.deg2rad(85.0)    # 翻转起始角 (belly姿态)
THETA_LAND = np.deg2rad(0.0)      # 翻转终止角 (着陆姿态)
T_FLIP_MAX = 8.0                  # s, 翻转超时Kill阈值
T_FLIP_TARGET = 3.5               # s, 目标翻转时间 (工程判断: 3-5s, 太快PD跟踪不上)

# bang-bang参数
M_MARGIN = 0.8                    # 力矩裕度 (用80%最大力矩, 留余量)

# PD跟踪带宽 (非调参, 基于bang-bang加速度反推)
# 稳态误差 e_ss = α_max/ωn², 要求 e_ss < 5°=0.087rad
# ωn² > α_max/0.087 = 0.484/0.087 = 5.56 → ωn > 2.36
# 取 ωn = 3.0 留余量
WN_TRACK = 3.0                    # rad/s, PD自然频率
ZETA_TRACK = 0.7                  # 阻尼比

# 兼容旧接口
KP_FLIP = WN_TRACK ** 2           # = 9.0
KD_FLIP = 2.0 * ZETA_TRACK * WN_TRACK  # = 4.2


def compute_max_flip_torque(state, m_fuel):
    """
    计算翻转段最大可用力矩 (襟翼最大偏转).

    M_max = Q·S·L·(Cδf+Cδa)·δ_max

    返回: (M_max, Q, alpha, M)
    """
    x, h, vx, vz, theta, q = state
    V = np.sqrt(vx ** 2 + vz ** 2)

    if V < 1e-6:
        return 0.0, 0.0, 0.0, 0.0

    rho, a_sound, p, T_air = atmosphere(h)
    M = V / a_sound
    alpha, gamma = angle_of_attack(theta, vx, vz)

    Q = 0.5 * rho * V ** 2 * S_REF
    # Phase 9.0修复: Q已含S_REF, 不再重复乘 (与aero_model.py一致)
    # Bug: 原代码 M_max = Q * S_REF * L_REF * ..., S_REF被乘了两次, 力矩放大27倍
    # 修复: M_max = Q * L_REF * ... (= 0.5*rho*V²*S_REF*L_REF*...)
    M_max = Q * L_REF * (C_DELTA_FWD + C_DELTA_AFT) * DELTA_MAX

    return M_max, Q, alpha, M


def compute_t_switch(theta0, thetaf, state, m_fuel, t_target=T_FLIP_TARGET):
    """
    缺陷21: bang-bang切换时间解析公式.

    工程判断: 翻转时间太短(如0.25s)会导致PD跟踪不上、过冲失控.
    用目标翻转时间T_FLIP_TARGET(3.5s)反推所需力矩, 而非用最大力矩.

    对称bang-bang (加速t_switch, 减速t_switch):
      t_switch = t_target / 2
      alpha_max = |Δθ| / t_switch²
      M_needed = alpha_max · Iyy

    Kill检查: M_needed > M_max·M_MARGIN → 力矩不足, 延长翻转时间

    返回: (t_switch, t_total, alpha_max, M_max)
    """
    M_max, Q, alpha, M = compute_max_flip_torque(state, m_fuel)
    Iyy = get_Iyy(m_fuel)

    # 目标翻转时间 (工程判断: 3-5s, 太快PD跟踪不上)
    t_total = t_target
    t_switch = t_total / 2.0

    # 角度变化量
    delta_theta = abs(theta0 - thetaf)

    # 所需角加速度
    alpha_max = delta_theta / (t_switch ** 2) if t_switch > 0 else 0.0

    # Kill检查: 所需力矩是否超过最大可用力矩
    M_needed = alpha_max * Iyy
    if M_needed > M_max * M_MARGIN:
        # 力矩不足, 延长翻转时间
        t_switch = np.sqrt(delta_theta / (M_max * M_MARGIN / Iyy))
        t_total = 2.0 * t_switch
        alpha_max = delta_theta / (t_switch ** 2)

    return t_switch, t_total, alpha_max, M_max


def bangbang_theta_trajectory(t, theta0, thetaf, t_switch, t_total):
    """
    bang-bang期望轨迹 (梯形角速度剖面).

    0 < t < t_switch: 匀加速翻转
      θ(t) = θ0 - 0.5·α_max·t²
      θ̇(t) = -α_max·t

    t_switch < t < t_total: 匀减速翻转
      θ(t) = θ_mid - α_max·t_switch·(t-t_switch) + 0.5·α_max·(t-t_switch)²
      θ̇(t) = -α_max·t_switch + α_max·(t-t_switch)

    t > t_total: 保持θf
      θ(t) = θf
      θ̇(t) = 0
    """
    delta_theta = theta0 - thetaf  # 正值 (翻转减小θ)

    if t_total < 1e-6:
        return thetaf, 0.0

    alpha_max = delta_theta / (t_switch ** 2) if t_switch > 0 else 0.0

    if t < 0:
        return theta0, 0.0
    elif t < t_switch:
        # 加速阶段
        theta = theta0 - 0.5 * alpha_max * t ** 2
        theta_dot = -alpha_max * t
    elif t < t_total:
        # 减速阶段
        t_dec = t - t_switch
        theta_mid = theta0 - 0.5 * alpha_max * t_switch ** 2
        theta = theta_mid - alpha_max * t_switch * t_dec + 0.5 * alpha_max * t_dec ** 2
        theta_dot = -alpha_max * t_switch + alpha_max * t_dec
    else:
        # 翻转完成
        theta = thetaf
        theta_dot = 0.0

    return theta, theta_dot


def bangbang_theta_acceleration(t, t_switch, t_total, alpha_max):
    """bang-bang参考角加速度 (用于标称前馈)."""
    if t < 0 or t >= t_total:
        return 0.0
    elif t < t_switch:
        return -alpha_max  # 加速阶段: θ减小
    else:
        return alpha_max   # 减速阶段: θ̈反向


def compute_feedforward_torque(state, theta_ref, m_fuel):
    """
    缺陷22: 前馈力矩补偿 (补偿气动力矩残余).

    trim在α_ref处配平, 实际α偏离α_ref时有残余力矩.
    M_residual = M_aero(α) - M_aero(α_ref)
    前馈补偿: M_ff = -M_residual

    返回: M_ff (N·m)
    """
    x, h, vx, vz, theta, q = state
    V = np.sqrt(vx ** 2 + vz ** 2)

    if V < 1e-6:
        return 0.0

    rho, a_sound, p, T_air = atmosphere(h)
    M = V / a_sound
    alpha, gamma = angle_of_attack(theta, vx, vz)

    # α_ref对应的力矩
    alpha_ref = theta_ref - gamma
    alpha_ref = (alpha_ref + np.pi) % (2 * np.pi) - np.pi

    CD0, CDa, CLa, Cma = aero_coefficients(M)
    Cm_actual = -Cma * np.sin(alpha - np.deg2rad(85.0))
    Cm_ref = -Cma * np.sin(alpha_ref - np.deg2rad(85.0))

    # 残余力矩
    Q = 0.5 * rho * V ** 2 * S_REF
    # Phase 9.0修复: Q已含S_REF, 不再重复乘 (与aero_model.py一致)
    M_residual = Q * L_REF * (Cm_actual - Cm_ref)

    # 前馈补偿 (抵消残余)
    M_ff = -M_residual

    return M_ff


class FlipController:
    """
    Step 7D 翻转段控制器: bang-bang + 标称前馈 + PD + 前馈力矩补偿.

    使用方法:
      1. 翻转开始时调用 plan() 规划bang-bang轨迹
      2. 每步调用 control() 获取控制量
      3. 检查 is_complete() 判断翻转是否完成
    """

    def __init__(self, theta0=THETA_BELLY, thetaf=THETA_LAND):
        self.theta0 = theta0
        self.thetaf = thetaf
        self.t_switch = 0.0
        self.t_total = 0.0
        self.alpha_max = 0.0
        self.M_max = 0.0
        self.planned = False
        self.flip_t = 0.0  # 翻转计时

    def plan(self, state, m_fuel):
        """
        规划bang-bang翻转轨迹 (缺陷21: 解析t_switch).
        """
        self.t_switch, self.t_total, self.alpha_max, self.M_max = compute_t_switch(
            self.theta0, self.thetaf, state[:6], m_fuel)
        self.planned = True
        self.flip_t = 0.0

        return {
            't_switch': self.t_switch,
            't_total': self.t_total,
            'alpha_max': self.alpha_max,
            'M_max': self.M_max,
            'kill_timeout': self.t_total > T_FLIP_MAX,
        }

    def control(self, state, dt):
        """
        计算翻转段控制量.

        三项力矩叠加:
          M_nominal = Iyy · θ̈_ref        (标称前馈, 驱动bang-bang加速度)
          M_pd = Iyy · (ωn²·e_θ + 2ζωn·e_q)  (PD跟踪)
          M_ff = -M_residual              (前馈补偿气动残余)

        总额外襟翼:
          δ_extra = (M_nominal + M_pd + M_ff) / (Q·S·L·(Cδf+Cδa))
          前后翼等偏: δ_fwd = δ_aft = δ_extra

        返回: (T, theta_cmd, delta_extra_fwd, delta_extra_aft)
        """
        if not self.planned:
            raise RuntimeError("FlipController: 必须先调用 plan()")

        x, h, vx, vz, theta, q, m_fuel = state
        self.flip_t += dt

        # bang-bang期望轨迹
        theta_ref, theta_ref_dot = bangbang_theta_trajectory(
            self.flip_t, self.theta0, self.thetaf, self.t_switch, self.t_total)

        # bang-bang参考加速度 (标称前馈)
        theta_ref_ddot = bangbang_theta_acceleration(
            self.flip_t, self.t_switch, self.t_total, self.alpha_max)

        # 推力: 翻转段保持T_idle (缺陷12: 防T=0使TVC无效)
        T = T_IDLE

        V = np.sqrt(vx ** 2 + vz ** 2)
        if V > 1e-6:
            rho, a_sound, p, T_air = atmosphere(h)
            M_mach = V / a_sound
            Q = 0.5 * rho * V ** 2 * S_REF
            Iyy = get_Iyy(m_fuel)

            # ============ 1. 标称前馈力矩 (缺陷22: 数值积分前馈) ============
            # M_nominal = Iyy · θ̈_ref
            # 这是驱动火箭按bang-bang加速度运动所需的力矩
            M_nominal = Iyy * theta_ref_ddot

            # ============ 2. PD反馈力矩 (跟踪bang-bang轨迹) ============
            # M_pd = Iyy · (ωn²·e_θ + 2ζωn·e_q)
            e_theta = theta_ref - theta
            # 归一化到[-π,π]防wrap
            e_theta = (e_theta + np.pi) % (2 * np.pi) - np.pi
            e_q = theta_ref_dot - q
            M_pd = Iyy * (WN_TRACK ** 2 * e_theta + 2.0 * ZETA_TRACK * WN_TRACK * e_q)

            # ============ 3. 前馈力矩补偿 (缺陷22: 补偿气动残余) ============
            M_ff = compute_feedforward_torque(state[:6], theta_ref, m_fuel)

            # ============ 总额外襟翼偏转 ============
            M_extra = M_nominal + M_pd + M_ff
            # Phase 9.0修复: Q已含S_REF, 不再重复乘 (与aero_model.py一致)
            denom = Q * L_REF * (C_DELTA_FWD + C_DELTA_AFT)
            delta_extra = M_extra / denom if denom > 0 else 0.0
            delta_extra = np.clip(delta_extra, -DELTA_MAX, DELTA_MAX)
        else:
            delta_extra = 0.0

        # 前后翼等偏 (最大力矩效率: M_flap = Q·S·L·(Cδf+Cδa)·δ)
        delta_extra_fwd = delta_extra
        delta_extra_aft = delta_extra

        # theta_cmd传给dynamics: 用theta_ref使trim在α_ref处配平
        # trim(α_ref)抵消M_aero(α_ref), delta_extra提供净控制力矩
        theta_cmd = theta_ref

        return T, theta_cmd, delta_extra_fwd, delta_extra_aft

    def is_complete(self, state):
        """
        翻转完成判断: θ接近θf且q接近0.
        theta归一化到[-π,π]防多圈wrap误判.
        """
        x, h, vx, vz, theta, q, m_fuel = state

        # 归一化theta到[-π,π]
        theta_norm = (theta + np.pi) % (2 * np.pi) - np.pi
        thetaf_norm = (self.thetaf + np.pi) % (2 * np.pi) - np.pi
        theta_err = abs(theta_norm - thetaf_norm)
        # 取最短角度差
        theta_err = min(theta_err, 2 * np.pi - theta_err)

        q_small = abs(q) < 0.05  # rad/s

        # 时间超限或角度收敛
        if self.flip_t > self.t_total + 2.0:  # 给2s余量收敛
            return True
        if theta_err < np.deg2rad(5.0) and q_small:
            return True
        return False

    def is_timeout(self):
        """Kill检查: 翻转超时 > 8s."""
        return self.flip_t > T_FLIP_MAX
