"""
Belly-Flop 2D 动力学 (为 9.0 凸化做准备).
============================================
状态: X = [x, h, vx, vz, θ, q] (6维, 控制)
      + m_fuel (第7维, 仅积分用)
控制: U = [T, θ_cmd] (2维)

坐标系:
  x: 水平 (下程方向)
  h: 高度 (向上为正)
  vx: 水平速度
  vz: 垂直速度 (正向下)
  θ: 俯仰角 (0=垂直头朝上, π/2=水平腹部朝下)
  q: 俯仰角速度

θ_cmd 跟踪动力学 (二阶):
  θ̈ = ωn²·(θ_cmd - θ) - 2ζωn·q, ωn=2rad/s, ζ=0.7
  在开环验证中, 通过配平襟翼在 α_cmd 处配平实现:
    - 襟翼在 α_cmd = θ_cmd - γ 处配平 (M_flap = -M_aero(α_cmd))
    - 实际 α 偏离 α_cmd 时, M_total = M_aero(α) - M_aero(α_cmd) ≠ 0
    - Cm 的负斜率提供恢复力矩, 气动耦合提供阻尼

完整动力学:
  V = sqrt(vx² + vz²)
  γ = atan2(vx, vz)
  α = θ - γ
  M = V / a_sound(h)
  Q = 0.5·ρ(h)·V²·S_ref
  D = Q·CD(α,M),  L = Q·CL(α,M)
  # 气动力坐标变换 (γ=atan2(vx,vz), 从垂直轴算)
  Fx_aero = -D·sin(γ) + L·cos(γ)
  Fz_aero = -D·cos(γ) - L·sin(γ)
  ax = (Fx_aero + T·sin(θ)) / m
  az = (Fz_aero - T·cos(θ)) / m + g(h)
  M_aero = Q·S·L·Cm(α,M)
  M_flap = Q·S·L·(Cδf·δ_fwd + Cδa·δ_aft)
  dq/dt = (M_aero + M_flap) / Iyy(m_fuel)
  dθ/dt = q
  dm/dt = -T / (Isp·g0)

积分: RK4, dt=0.01s
"""
import numpy as np
from .aero_model import (
    aero_forces_and_moments, aero_coefficients, angle_of_attack,
    get_Iyy, get_mass, gravity, trim_flaps,
    S_REF, L_REF, T_MAX, T_IDLE, ISP, G0_ISP,
    M_FUEL_INIT, C_DELTA_FWD, C_DELTA_AFT, DELTA_MAX,
)


# θ_cmd 跟踪动力学参数 (PD 带宽)
OMEGA_N = 2.0      # rad/s, 自然频率
ZETA = 0.7          # 阻尼比


def state_derivative(state, T, theta_cmd, delta_extra_fwd=0.0, delta_extra_aft=0.0):
    """
    计算状态导数.

    state: [x, h, vx, vz, θ, q, m_fuel] (7维)
    T: 推力 (N)
    theta_cmd: 期望俯仰角 (rad)
    delta_extra_fwd: 额外前翼偏转 (rad, PD阻尼用, 默认0=纯配平)
    delta_extra_aft: 额后后翼偏转 (rad, PD阻尼用, 默认0=纯配平)

    返回: dstate/dt (7维)
    """
    x, h, vx, vz, theta, q, m_fuel = state

    # 质量/惯量
    m = get_mass(m_fuel)
    Iyy = get_Iyy(m_fuel)
    g = gravity(h)

    # 气流角
    V = np.sqrt(vx ** 2 + vz ** 2)
    alpha, gamma = angle_of_attack(theta, vx, vz)

    # 大气/Mach
    from .aero_model import atmosphere
    rho, a_sound, p, T_air = atmosphere(h)
    M = V / a_sound if a_sound > 0 else 0.0

    # 配平襟翼: 在 α_cmd = θ_cmd - γ 处配平
    alpha_cmd = theta_cmd - gamma
    # 归一化 α_cmd 到 [-π, π]
    alpha_cmd = (alpha_cmd + np.pi) % (2 * np.pi) - np.pi
    delta_trim_fwd, delta_trim_aft = trim_flaps(alpha_cmd, M)

    # 总襟翼 = 配平 + PD阻尼 (缺陷7B: 主动阻尼解决缺陷3弱阻尼)
    delta_fwd = np.clip(delta_trim_fwd + delta_extra_fwd, -DELTA_MAX, DELTA_MAX)
    delta_aft = np.clip(delta_trim_aft + delta_extra_aft, -DELTA_MAX, DELTA_MAX)

    # 气动力/力矩 (在实际 α 处)
    (D, L, Fx_aero, Fz_aero, M_aero, M_flap, M_total, Q,
     alpha_act, gamma_act, M_act, rho_act, a_act) = aero_forces_and_moments(
        vx, vz, theta, h, delta_fwd, delta_aft)

    # 状态导数
    dx_dt = vx
    dh_dt = -vz   # vz 正向下, h 正向上
    dvx_dt = (Fx_aero + T * np.sin(theta)) / m
    dvz_dt = (Fz_aero - T * np.cos(theta)) / m + g
    dtheta_dt = q
    dq_dt = M_total / Iyy
    dm_dt = -T / (ISP * G0_ISP)

    return np.array([dx_dt, dh_dt, dvx_dt, dvz_dt, dtheta_dt, dq_dt, dm_dt])


def rk4_step(state, T, theta_cmd, dt, delta_extra_fwd=0.0, delta_extra_aft=0.0):
    """
    RK4 单步积分.

    state: [x, h, vx, vz, θ, q, m_fuel] (7维)
    T: 推力 (N)
    theta_cmd: 期望俯仰角 (rad)
    dt: 步长 (s)
    delta_extra_fwd/aft: 额外襟翼偏转 (rad, PD阻尼用)

    返回: 新状态 (7维)
    """
    k1 = state_derivative(state, T, theta_cmd, delta_extra_fwd, delta_extra_aft)
    k2 = state_derivative(state + 0.5 * dt * k1, T, theta_cmd, delta_extra_fwd, delta_extra_aft)
    k3 = state_derivative(state + 0.5 * dt * k2, T, theta_cmd, delta_extra_fwd, delta_extra_aft)
    k4 = state_derivative(state + dt * k3, T, theta_cmd, delta_extra_fwd, delta_extra_aft)
    new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    # 燃料非负
    new_state[6] = max(new_state[6], 0.0)
    return new_state


def simulate(initial_state, T_schedule, theta_cmd_schedule, dt, t_end,
             record_interval=1):
    """
    开环仿真.

    initial_state: [x, h, vx, vz, θ, q, m_fuel] (7维)
    T_schedule: callable(t) -> 推力 (N), 或常数
    theta_cmd_schedule: callable(t) -> θ_cmd (rad), 或常数
    dt: 步长 (s)
    t_end: 结束时间 (s)
    record_interval: 每N步记录一次

    返回: history dict, 包含时间序列
    """
    # 处理常数控制
    if not callable(T_schedule):
        T_val = T_schedule
        T_schedule = lambda t: T_val
    if not callable(theta_cmd_schedule):
        theta_cmd_val = theta_cmd_schedule
        theta_cmd_schedule = lambda t: theta_cmd_val

    n_steps = int(t_end / dt)
    state = np.array(initial_state, dtype=float)

    # 记录
    times = []
    states = []
    controls = []
    aeros = []  # 气动信息

    for i in range(n_steps + 1):
        t = i * dt
        T = T_schedule(t)
        theta_cmd = theta_cmd_schedule(t)

        # 记录 (每 record_interval 步)
        if i % record_interval == 0 or i == n_steps:
            x, h, vx, vz, theta, q, m_fuel = state
            V = np.sqrt(vx ** 2 + vz ** 2)
            alpha, gamma = angle_of_attack(theta, vx, vz)
            rho, a_sound, p, T_air = atmosphere(h)
            M = V / a_sound if a_sound > 0 else 0.0
            Q = 0.5 * rho * V ** 2 * S_REF

            # 配平襟翼
            alpha_cmd = theta_cmd - gamma
            alpha_cmd = (alpha_cmd + np.pi) % (2 * np.pi) - np.pi
            delta_fwd, delta_aft = trim_flaps(alpha_cmd, M)

            # 气动力
            (D, L, Fx_aero, Fz_aero, M_aero, M_flap, M_total, Q_act,
             _, _, _, _, _) = aero_forces_and_moments(
                vx, vz, theta, h, delta_fwd, delta_aft)

            times.append(t)
            states.append(state.copy())
            controls.append([T, theta_cmd, delta_fwd, delta_aft])
            aeros.append({
                'V': V, 'alpha': alpha, 'gamma': gamma, 'Mach': M,
                'Q': Q, 'D': D, 'L': L, 'Fx_aero': Fx_aero, 'Fz_aero': Fz_aero,
                'M_aero': M_aero, 'M_flap': M_flap, 'M_total': M_total,
                'rho': rho, 'a_sound': a_sound,
                'alpha_cmd': alpha_cmd, 'delta_fwd': delta_fwd, 'delta_aft': delta_aft,
            })

        # 积分一步
        if i < n_steps:
            state = rk4_step(state, T, theta_cmd, dt)

            # 安全检查: 高度非负
            if state[1] < 0:
                state[1] = 0.0
                break

    # 转为 numpy 数组
    times = np.array(times)
    states = np.array(states)
    controls = np.array(controls)

    return {
        't': times,
        'x': states[:, 0],
        'h': states[:, 1],
        'vx': states[:, 2],
        'vz': states[:, 3],
        'theta': states[:, 4],
        'q': states[:, 5],
        'm_fuel': states[:, 6],
        'T': controls[:, 0],
        'theta_cmd': controls[:, 1],
        'delta_fwd': controls[:, 2],
        'delta_aft': controls[:, 3],
        'aero': aeros,
    }


# 便捷函数: 从 aero_model 导入 atmosphere (避免循环引用)
from .aero_model import atmosphere


def simulate_closed_loop(initial_state, controller, dt, t_end,
                         record_interval=10, h_ground=0.0):
    """
    闭环仿真 (Step 7B+).

    initial_state: [x, h, vx, vz, θ, q, m_fuel] (7维)
    controller: BellyFlopController 实例, 接口:
        controller.update(state, dt) -> (T, theta_cmd, d_extra_fwd, d_extra_aft, phase, kill, info)
    dt: 步长 (s)
    t_end: 最大仿真时长 (s)
    record_interval: 每N步记录一次
    h_ground: 地面高度 (m), h<=h_ground 触地停止

    返回: history dict (同 simulate, 额外含 phase, kill 标志)
    """
    n_steps = int(t_end / dt)
    state = np.array(initial_state, dtype=float)

    times = []
    states = []
    controls = []
    aeros = []
    phases = []
    kill_flag = False
    kill_reason = ''
    landing_success = False

    for i in range(n_steps + 1):
        t = i * dt

        # 控制器更新
        T, theta_cmd, d_extra_fwd, d_extra_aft, phase, kill, info = controller.update(state, dt)

        # 记录
        if i % record_interval == 0 or i == n_steps or kill or state[1] <= h_ground:
            x, h, vx, vz, theta, q, m_fuel = state
            V = np.sqrt(vx ** 2 + vz ** 2)
            alpha, gamma = angle_of_attack(theta, vx, vz)
            rho, a_sound, p, T_air = atmosphere(h)
            M = V / a_sound if a_sound > 0 else 0.0
            Q = 0.5 * rho * V ** 2 * S_REF

            # 配平襟翼 (记录用)
            alpha_cmd = theta_cmd - gamma
            alpha_cmd = (alpha_cmd + np.pi) % (2 * np.pi) - np.pi
            delta_trim_fwd, delta_trim_aft = trim_flaps(alpha_cmd, M)
            delta_fwd_total = np.clip(delta_trim_fwd + d_extra_fwd, -DELTA_MAX, DELTA_MAX)
            delta_aft_total = np.clip(delta_trim_aft + d_extra_aft, -DELTA_MAX, DELTA_MAX)

            (D, L, Fx_aero, Fz_aero, M_aero, M_flap, M_total, Q_act,
             _, _, _, _, _) = aero_forces_and_moments(
                vx, vz, theta, h, delta_fwd_total, delta_aft_total)

            times.append(t)
            states.append(state.copy())
            controls.append([T, theta_cmd, delta_fwd_total, delta_aft_total,
                             d_extra_fwd, d_extra_aft])
            aeros.append({
                'V': V, 'alpha': alpha, 'gamma': gamma, 'Mach': M,
                'Q': Q, 'D': D, 'L': L, 'Fx_aero': Fx_aero, 'Fz_aero': Fz_aero,
                'M_aero': M_aero, 'M_flap': M_flap, 'M_total': M_total,
                'rho': rho, 'a_sound': a_sound,
                'alpha_cmd': alpha_cmd,
                'delta_trim_fwd': delta_trim_fwd, 'delta_trim_aft': delta_trim_aft,
            })
            phases.append(phase)

        # Kill 检查
        if kill:
            kill_flag = True
            kill_reason = info.get('kill_reason', 'unknown')
            break

        # 触地检查
        if state[1] <= h_ground:
            # 着陆成功判定: vz < 10 m/s 且 |vx| < 5 m/s 且 |theta| < 15°
            vz_land = state[3]
            vx_land = state[2]
            theta_land = np.degrees(state[4])
            if abs(vz_land) < 10.0 and abs(vx_land) < 5.0 and abs(theta_land) < 15.0:
                landing_success = True
            else:
                kill_flag = True
                kill_reason = f'hard_landing (vz={vz_land:.1f}, vx={vx_land:.1f}, theta={theta_land:.1f}°)'
            break

        # 积分一步
        if i < n_steps:
            state = rk4_step(state, T, theta_cmd, dt, d_extra_fwd, d_extra_aft)

    times = np.array(times)
    states = np.array(states)
    controls = np.array(controls)

    return {
        't': times,
        'x': states[:, 0],
        'h': states[:, 1],
        'vx': states[:, 2],
        'vz': states[:, 3],
        'theta': states[:, 4],
        'q': states[:, 5],
        'm_fuel': states[:, 6],
        'T': controls[:, 0],
        'theta_cmd': controls[:, 1],
        'delta_fwd': controls[:, 2],
        'delta_aft': controls[:, 3],
        'delta_extra_fwd': controls[:, 4],
        'delta_extra_aft': controls[:, 5],
        'aero': aeros,
        'phase': phases,
        'kill': kill_flag,
        'kill_reason': kill_reason,
        'landing_success': landing_success,
    }
