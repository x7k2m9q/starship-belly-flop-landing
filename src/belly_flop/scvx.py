"""
Successive Convexification (SCvx) 框架 — Belly-Flop 制导.
=============================================================
理论方案 9.0-Final Step 7C 分步凸化:

Step 7C-1: 固定 α 的 SCvx (验证框架)
  - α = 80° 恒定, ∂aero/∂α = 0
  - 状态 X = [x, h, vx, vz] (4维, θ=α+γ 不独立)
  - 控制 U = [T] (1维)
  - tgo = 1.2·sqrt(h²+x²)/V (缺陷11)
  - T 下限 T_idle (缺陷12: 防T=0使TVC无效)
  - Kill: 不收敛 → 基本框架有问题

Step 7C-2: θ_cmd 控制量 + 线性气动 (验证可控性)
  - 状态 X = [x, h, vx, vz, θ, q] (6维)
  - 控制 U = [T, θ_cmd] (2维)
  - 二阶跟踪动力学, T·sin(θ) Taylor, CD 线性化

Step 7C-3: 完整 sin²(α) 凸化 (9.0 最终目标)
  - Q·CD 一阶 Taylor, γ 偏导, 缺陷3论文讨论

SCvx 算法:
  1. 初始化参考轨迹 (X_bar, U_bar)
  2. 在参考点线性化: A_d, B_d, C_d (离散化)
  3. 构造 SOCP: min ||X-X_ref||² + λ||U-U_ref||²
     s.t. 线性动力学 + 初始条件 + 控制约束 + 信赖域 + 终端约束
  4. 求解 SOCP → (X_opt, U_opt)
  5. 更新参考: X_bar = X_opt, U_bar = U_opt
  6. 收敛检查: ||X_opt - X_prev|| < tol
  7. 熔断: 超过 max_iter 或代价增大 → Kill
"""
import numpy as np
import cvxpy as cp
from .aero_model import (
    aero_coefficients, angle_of_attack, atmosphere,
    get_mass, get_Iyy, gravity,
    S_REF, L_REF, T_MAX, T_IDLE, ISP, G0_ISP,
    M_FUEL_INIT, M_DRY,
    R_EARTH, G0_SL,
)


# =====================================================================
# Step 7C-1: 固定 α 的 SCvx
# =====================================================================
ALPHA_FIXED_7C1 = np.deg2rad(80.0)  # 固定攻角 80°


def dynamics_7c1(state, T, m_fuel, alpha_fixed=ALPHA_FIXED_7C1):
    """
    固定 α 动力学 (连续时间).

    state: [x, h, vx, vz] (4维)
    T: 推力 (N)
    m_fuel: 燃料质量 (kg, 用于计算总质量)
    alpha_fixed: 固定攻角 (rad)

    返回: dstate/dt (4维)
    """
    x, h, vx, vz = state
    m = get_mass(m_fuel)
    g = gravity(h)

    V = np.sqrt(vx ** 2 + vz ** 2)
    if V < 1e-6:
        return np.array([0.0, 0.0, 0.0, g])

    gamma = np.arctan2(vx, vz)
    theta = alpha_fixed + gamma  # α 固定 → θ = α + γ

    # 大气/Mach
    rho, a_sound, p, T_air = atmosphere(h)
    M = V / a_sound if a_sound > 0 else 0.0

    # 气动系数 (α 固定, 只随 M 变化)
    CD0, CDa, CLa, Cma = aero_coefficients(M)
    CD = CD0 + CDa * np.sin(alpha_fixed) ** 2
    CL = CLa * np.sin(2.0 * alpha_fixed) * 0.5

    # 动压
    Q = 0.5 * rho * V ** 2 * S_REF
    D = Q * CD
    L = Q * CL

    # 气动力坐标变换 (γ=atan2(vx,vz), 从垂直轴算)
    Fx_aero = -D * np.sin(gamma) + L * np.cos(gamma)
    Fz_aero = -D * np.cos(gamma) - L * np.sin(gamma)

    # 推力方向 (θ = α + γ)
    ax = (Fx_aero + T * np.sin(theta)) / m
    az = (Fz_aero - T * np.cos(theta)) / m + g

    return np.array([vx, -vz, ax, az])


def jacobian_7c1(state, T, m_fuel, alpha_fixed=ALPHA_FIXED_7C1):
    """
    数值 Jacobian (中心差分).
    A = ∂f/∂X (4×4), B = ∂f/∂U (4×1)
    """
    eps = 1e-6
    f0 = dynamics_7c1(state, T, m_fuel, alpha_fixed)

    # A = ∂f/∂X
    A = np.zeros((4, 4))
    for i in range(4):
        s_plus = state.copy()
        s_plus[i] += eps
        s_minus = state.copy()
        s_minus[i] -= eps
        f_plus = dynamics_7c1(s_plus, T, m_fuel, alpha_fixed)
        f_minus = dynamics_7c1(s_minus, T, m_fuel, alpha_fixed)
        A[:, i] = (f_plus - f_minus) / (2 * eps)

    # B = ∂f/∂T
    f_plus = dynamics_7c1(state, T + eps, m_fuel, alpha_fixed)
    f_minus = dynamics_7c1(state, T - eps, m_fuel, alpha_fixed)
    B = (f_plus - f_minus).reshape(4, 1) / (2 * eps)

    return A, B


def discretize_7c1(state, T, dt, m_fuel, alpha_fixed=ALPHA_FIXED_7C1):
    """
    离散化 (Euler): X[k+1] = A_d·X[k] + B_d·U[k] + C_d.
    A_d = I + dt·A, B_d = dt·B, C_d = dt·(f(X̄,Ū) - A·X̄ - B·Ū)
    """
    A, B = jacobian_7c1(state, T, m_fuel, alpha_fixed)
    f0 = dynamics_7c1(state, T, m_fuel, alpha_fixed)

    A_d = np.eye(4) + dt * A
    B_d = dt * B
    C_d = dt * (f0 - A @ state - B.flatten() * T)

    return A_d, B_d, C_d


class SCvxSolver7C1:
    """
    Step 7C-1 SCvx 求解器: 固定 α=80°, 优化推力 T 和轨迹.

    参数:
      X0: 初始状态 [x, h, vx, vz]
      X_term: 终端目标 [x, h, vx, vz]
      N: 时域步数
      dt: 离散步长 (s)
      m_fuel_init: 初始燃料 (kg)
      max_iter: 最大 SCvx 迭代次数
      trust_radius: 信赖域半径
      conv_tol: 收敛容差
    """

    def __init__(self, X0, X_term, N, dt,
                 m_fuel_init=M_FUEL_INIT, max_iter=20,
                 trust_radius_x=30.0, trust_radius_v=30.0,
                 conv_tol=1e-3, verbose=False):
        self.X0 = np.array(X0, dtype=float)
        self.X_term = np.array(X_term, dtype=float)
        self.N = N
        self.dt = dt
        self.m_fuel = m_fuel_init
        self.max_iter = max_iter
        self.trust_x = trust_radius_x
        self.trust_v = trust_radius_v
        self.conv_tol = conv_tol
        self.verbose = verbose

        # 代价权重
        self.Q = np.diag([1.0, 10.0, 1.0, 1.0])       # 状态代价
        self.R = np.array([[0.01]])                     # 控制代价
        self.Qf = np.diag([10.0, 100.0, 10.0, 10.0])   # 终端代价

        # 终端约束: 软约束 (大Qf惩罚, SCvx标准做法, 避免硬约束导致infeasible)
        # 硬约束作为可选, 默认关闭
        self.terminal_hard_constraint = False
        self.h_term_tol = 1000.0     # m, 硬约束容差 (可选)
        self.v_term_tol = 50.0       # m/s, 硬约束容差 (可选)

        # 收敛历史
        self.cost_history = []
        self.converged = False
        self.iter_count = 0

    def _init_trajectory(self):
        """初始化参考轨迹 (用T_idle动力学仿真, 非直线插值)."""
        X_ref = np.zeros((4, self.N + 1))
        U_ref = np.zeros((1, self.N))
        state = self.X0.copy()
        m_fuel = self.m_fuel

        X_ref[:, 0] = state
        for k in range(self.N):
            U_ref[0, k] = T_IDLE
            # RK4积分生成参考轨迹
            k1 = dynamics_7c1(state, T_IDLE, m_fuel)
            k2 = dynamics_7c1(state + 0.5 * self.dt * k1, T_IDLE, m_fuel)
            k3 = dynamics_7c1(state + 0.5 * self.dt * k2, T_IDLE, m_fuel)
            k4 = dynamics_7c1(state + self.dt * k3, T_IDLE, m_fuel)
            state = state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            m_fuel -= T_IDLE * self.dt / (ISP * G0_ISP)
            m_fuel = max(m_fuel, 0.0)
            if state[1] < 0:
                state[1] = 0.0
            X_ref[:, k + 1] = state

        return X_ref, U_ref

    def _linearize(self, X_ref, U_ref):
        """在参考轨迹上线性化, 返回 A_d[k], B_d[k], C_d[k]."""
        A_list = []
        B_list = []
        C_list = []
        for k in range(self.N):
            # 燃料消耗估计 (简化: 匀速消耗)
            m_fuel_k = self.m_fuel * (1 - k / self.N * 0.3)  # 保守估计30%消耗
            A_d, B_d, C_d = discretize_7c1(
                X_ref[:, k], U_ref[0, k], self.dt, m_fuel_k)
            A_list.append(A_d)
            B_list.append(B_d)
            C_list.append(C_d)
        return A_list, B_list, C_list

    def _solve_scp(self, X_ref, U_ref, A_list, B_list, C_list):
        """求解 SOCP 子问题."""
        N = self.N
        X = cp.Variable((4, N + 1))
        U = cp.Variable((1, N))

        # 代价
        cost = 0
        for k in range(N):
            cost += cp.quad_form(X[:, k] - X_ref[:, k], self.Q)
            cost += cp.quad_form(U[:, k] - U_ref[:, k], self.R)
        cost += cp.quad_form(X[:, N] - self.X_term, self.Qf)

        # 约束
        constraints = [X[:, 0] == self.X0]

        for k in range(N):
            # 动力学
            constraints.append(
                X[:, k + 1] == A_list[k] @ X[:, k] + B_list[k] @ U[:, k] + C_list[k])
            # 控制约束 (缺陷12: T下限T_idle)
            constraints.append(U[0, k] >= T_IDLE)
            constraints.append(U[0, k] <= T_MAX)
            # 信赖域 (位置+速度分开)
            constraints.append(cp.norm(X[:2, k] - X_ref[:2, k], 'inf') <= self.trust_x)
            constraints.append(cp.norm(X[2:, k] - X_ref[2:, k], 'inf') <= self.trust_v)

        # 终端约束 (可选硬约束, 默认软约束)
        if self.terminal_hard_constraint:
            constraints.append(cp.norm(X[1, N] - self.X_term[1], 2) <= self.h_term_tol)
            constraints.append(cp.norm(X[2:, N] - self.X_term[2:], 2) <= self.v_term_tol)

        # 求解 (CLARABEL精度更好, SCS有时inaccurate)
        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception as e:
            return None, None, False, str(e)

        if prob.status not in ('optimal', 'optimal_inaccurate'):
            return None, None, False, f'status={prob.status}'

        return X.value, U.value, True, ''

    def solve(self):
        """
        SCvx 主循环.

        返回: (X_opt, U_opt, converged, info)
        """
        X_ref, U_ref = self._init_trajectory()

        for iteration in range(self.max_iter):
            self.iter_count = iteration + 1

            # 线性化
            A_list, B_list, C_list = self._linearize(X_ref, U_ref)

            # 求解 SOCP
            X_opt, U_opt, success, msg = self._solve_scp(
                X_ref, U_ref, A_list, B_list, C_list)

            if not success:
                if self.verbose:
                    print(f'  iter {iteration+1}: SOCP失败 - {msg}')
                return X_ref, U_ref, False, {'reason': f'socp_failed: {msg}',
                                              'iter': iteration + 1}

            # 代价
            cost = 0
            for k in range(self.N):
                cost += float((X_opt[:, k] - X_ref[:, k]) @ self.Q @ (X_opt[:, k] - X_ref[:, k]))
                cost += float((U_opt[:, k] - U_ref[:, k]) @ self.R @ (U_opt[:, k] - U_ref[:, k]))
            cost += float((X_opt[:, self.N] - self.X_term) @ self.Qf @ (X_opt[:, self.N] - self.X_term))
            self.cost_history.append(cost)

            # 收敛检查
            dx = np.max(np.abs(X_opt - X_ref))
            du = np.max(np.abs(U_opt - U_ref))

            if self.verbose:
                print(f'  iter {iteration+1}: cost={cost:.4e}, dx={dx:.4f}, du={du:.4f}')

            if dx < self.conv_tol and du < self.conv_tol:
                self.converged = True
                if self.verbose:
                    print(f'  收敛 @ iter {iteration+1}')
                return X_opt, U_opt, True, {'iter': iteration + 1, 'cost': cost,
                                            'cost_history': self.cost_history}

            # 代价增大检查 (熔断)
            if iteration > 0 and cost > 2.0 * self.cost_history[-2]:
                if self.verbose:
                    print(f'  熔断: 代价增大 cost={cost:.4e} > 2×prev={self.cost_history[-2]:.4e}')
                return X_opt, U_opt, False, {'reason': 'cost_diverge',
                                              'iter': iteration + 1, 'cost': cost}

            # 更新参考
            X_ref = X_opt
            U_ref = U_opt

        # 超过最大迭代
        if self.verbose:
            print(f'  熔断: 超过max_iter={self.max_iter}')
        return X_ref, U_ref, False, {'reason': 'max_iter_exceeded',
                                      'iter': self.max_iter, 'cost': self.cost_history[-1]}


def simulate_scvx_trajectory(X0, U_opt, dt, m_fuel_init=M_FUEL_INIT,
                              alpha_fixed=ALPHA_FIXED_7C1):
    """
    用非线性动力学仿真 SCvx 输出的控制序列 (验证跟踪性能).

    返回: history dict
    """
    N = len(U_opt)
    state = np.array(X0, dtype=float)
    m_fuel = m_fuel_init

    times = np.zeros(N + 1)
    states = np.zeros((4, N + 1))
    thrusts = np.zeros(N)
    machs = np.zeros(N + 1)

    states[:, 0] = state

    for k in range(N):
        T = U_opt[k]
        # RK4 积分
        k1 = dynamics_7c1(state, T, m_fuel, alpha_fixed)
        k2 = dynamics_7c1(state + 0.5 * dt * k1, T, m_fuel, alpha_fixed)
        k3 = dynamics_7c1(state + 0.5 * dt * k2, T, m_fuel, alpha_fixed)
        k4 = dynamics_7c1(state + dt * k3, T, m_fuel, alpha_fixed)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # 燃料消耗
        m_fuel -= T * dt / (ISP * G0_ISP)
        m_fuel = max(m_fuel, 0.0)

        # 高度非负
        if state[1] < 0:
            state[1] = 0.0

        times[k + 1] = (k + 1) * dt
        states[:, k + 1] = state
        thrusts[k] = T

        # Mach
        V = np.sqrt(state[2] ** 2 + state[3] ** 2)
        rho, a_sound, p, T_air = atmosphere(state[1])
        machs[k + 1] = V / a_sound if a_sound > 0 else 0.0

    V0 = np.sqrt(states[2, 0] ** 2 + states[3, 0] ** 2)
    machs[0] = V0 / atmosphere(states[1, 0])[1]

    return {
        't': times, 'x': states[0], 'h': states[1],
        'vx': states[2], 'vz': states[3],
        'T': thrusts, 'Mach': machs,
        'm_fuel_final': m_fuel,
    }


# =====================================================================
# Step 7C-2: θ_cmd 控制量 + 线性气动 (验证可控性)
# =====================================================================
# 状态: X = [x, h, vx, vz, θ, q] (6维)
# 控制: U = [T, θ_cmd] (2维)
# 二阶跟踪动力学: θ̈ = ωn²·(θ_cmd-θ) - 2ζωn·q
# T·sin(θ) 双线性: 数值Jacobian自动Taylor展开
#   ∂(T·sin(θ))/∂T = sin(θ̄), ∂(T·sin(θ))/∂θ = T̄·cos(θ̄)
# CD线性化: sin²(α) ≈ sin²(ᾱ) + sin(2ᾱ)·Δα (数值Jacobian自动处理)
# 信赖域: |Δθ|<10°, |Δvx|<30, |Δvz|<30, |ΔT|<0.3·T_max

OMEGA_N_TRACK = 2.0    # rad/s, θ跟踪自然频率
ZETA_TRACK = 0.7       # 阻尼比
THETA_CMD_MAX = np.deg2rad(85.0)   # θ_cmd上限 (belly姿态)
THETA_CMD_MIN = np.deg2rad(-10.0)  # θ_cmd下限 (允许轻微过冲)


def dynamics_7c2(state, U, m_fuel):
    """
    6维状态动力学 (连续时间).

    state: [x, h, vx, vz, θ, q] (6维)
    U: [T, θ_cmd] (2维)
    m_fuel: 燃料质量 (kg)

    返回: dstate/dt (6维)

    注: T·sin(θ)双线性和sin²(α)非线性由数值Jacobian自动线性化.
        缺陷13(二阶跟踪): θ̈ = ωn²·(θ_cmd-θ) - 2ζωn·q
        缺陷14(T·sin(θ) Taylor): 数值Jacobian ∂(T·sin(θ))/∂T=sin(θ̄), ∂/∂θ=T̄·cos(θ̄)
        缺陷15(CD线性化): sin²(α)≈sin²(ᾱ)+sin(2ᾱ)·Δα, 数值Jacobian自动处理
    """
    x, h, vx, vz, theta, q = state
    T, theta_cmd = U
    m = get_mass(m_fuel)
    g = gravity(h)

    V = np.sqrt(vx ** 2 + vz ** 2)

    # 气动力 (完整非线性, 数值Jacobian会自动线性化)
    if V < 1e-6:
        Fx_aero = 0.0
        Fz_aero = 0.0
    else:
        from .aero_model import aero_forces_and_moments
        (D, L, Fx_aero, Fz_aero, M_aero, M_flap, M_total, Q,
         alpha, gamma, M, rho, a_sound) = aero_forces_and_moments(
            vx, vz, theta, h, 0.0, 0.0)  # 配平襟翼=0, SCvx不优化襟翼

    # 推力方向 (T·sin(θ), T·cos(θ) — 双线性, 数值Jacobian自动Taylor展开)
    ax = (Fx_aero + T * np.sin(theta)) / m
    az = (Fz_aero - T * np.cos(theta)) / m + g

    # θ二阶跟踪动力学 (缺陷13: 防SCvx规划1步跳80°)
    dq_dt = OMEGA_N_TRACK ** 2 * (theta_cmd - theta) - 2 * ZETA_TRACK * OMEGA_N_TRACK * q

    return np.array([vx, -vz, ax, az, q, dq_dt])


def jacobian_7c2(state, U, m_fuel):
    """
    数值Jacobian (中心差分).
    A = ∂f/∂X (6×6), B = ∂f/∂U (6×2)

    自动处理:
      - T·sin(θ)双线性 (缺陷14): ∂(T·sin(θ))/∂T=sin(θ̄), ∂/∂θ=T̄·cos(θ̄)
      - sin²(α)非线性 (缺陷15): ∂CD/∂α=CDα·sin(2ᾱ)
      - γ=atan2偏导 (缺陷18): ∂γ/∂vx=vz/(vx²+vz²), ∂γ/∂vz=-vx/(vx²+vz²)
    """
    eps = 1e-6
    f0 = dynamics_7c2(state, U, m_fuel)

    # A = ∂f/∂X (6×6)
    A = np.zeros((6, 6))
    for i in range(6):
        s_plus = state.copy()
        s_plus[i] += eps
        s_minus = state.copy()
        s_minus[i] -= eps
        f_plus = dynamics_7c2(s_plus, U, m_fuel)
        f_minus = dynamics_7c2(s_minus, U, m_fuel)
        A[:, i] = (f_plus - f_minus) / (2 * eps)

    # B = ∂f/∂U (6×2)
    B = np.zeros((6, 2))
    for j in range(2):
        u_plus = U.copy()
        u_plus[j] += eps
        u_minus = U.copy()
        u_minus[j] -= eps
        f_plus = dynamics_7c2(state, u_plus, m_fuel)
        f_minus = dynamics_7c2(state, u_minus, m_fuel)
        B[:, j] = (f_plus - f_minus) / (2 * eps)

    return A, B


def discretize_7c2(state, U, dt, m_fuel):
    """离散化 (Euler): X[k+1] = A_d·X[k] + B_d·U[k] + C_d."""
    A, B = jacobian_7c2(state, U, m_fuel)
    f0 = dynamics_7c2(state, U, m_fuel)

    A_d = np.eye(6) + dt * A
    B_d = dt * B
    C_d = dt * (f0 - A @ state - B @ U)

    return A_d, B_d, C_d


class SCvxSolver7C2:
    """
    Step 7C-2 SCvx 求解器: θ_cmd控制量, 线性气动, 验证可控性.

    状态: X = [x, h, vx, vz, θ, q] (6维)
    控制: U = [T, θ_cmd] (2维)
    """

    def __init__(self, X0, X_term, N, dt,
                 m_fuel_init=M_FUEL_INIT, max_iter=30,
                 trust_theta=np.deg2rad(10.0), trust_v=30.0,
                 trust_T=0.3 * T_MAX, trust_pos=200.0,
                 conv_tol=1e-3, verbose=False):
        self.X0 = np.array(X0, dtype=float)
        self.X_term = np.array(X_term, dtype=float)
        self.N = N
        self.dt = dt
        self.m_fuel = m_fuel_init
        self.max_iter = max_iter
        self.trust_theta = trust_theta
        self.trust_v = trust_v
        self.trust_T = trust_T
        self.trust_pos = trust_pos
        self.conv_tol = conv_tol
        self.verbose = verbose

        # 代价权重 (6维状态, 2维控制)
        self.Q = np.diag([1.0, 10.0, 1.0, 1.0, 5.0, 1.0])
        self.R = np.diag([0.01, 0.1])
        self.Qf = np.diag([100.0, 1000.0, 100.0, 100.0, 50.0, 10.0])

        self.cost_history = []
        self.converged = False
        self.iter_count = 0

    def _init_trajectory(self):
        """初始化参考轨迹 (T_idle + θ_cmd=80°动力学仿真)."""
        X_ref = np.zeros((6, self.N + 1))
        U_ref = np.zeros((2, self.N))
        state = self.X0.copy()
        m_fuel = self.m_fuel

        X_ref[:, 0] = state
        theta_cmd_init = np.deg2rad(80.0)
        for k in range(self.N):
            U_ref[0, k] = T_IDLE
            U_ref[1, k] = theta_cmd_init
            # RK4积分
            k1 = dynamics_7c2(state, U_ref[:, k], m_fuel)
            k2 = dynamics_7c2(state + 0.5 * self.dt * k1, U_ref[:, k], m_fuel)
            k3 = dynamics_7c2(state + 0.5 * self.dt * k2, U_ref[:, k], m_fuel)
            k4 = dynamics_7c2(state + self.dt * k3, U_ref[:, k], m_fuel)
            state = state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            m_fuel -= T_IDLE * self.dt / (ISP * G0_ISP)
            m_fuel = max(m_fuel, 0.0)
            if state[1] < 0:
                state[1] = 0.0
            X_ref[:, k + 1] = state

        return X_ref, U_ref

    def _linearize(self, X_ref, U_ref):
        """在参考轨迹上线性化."""
        A_list, B_list, C_list = [], [], []
        for k in range(self.N):
            m_fuel_k = self.m_fuel * (1 - k / self.N * 0.3)
            A_d, B_d, C_d = discretize_7c2(X_ref[:, k], U_ref[:, k], self.dt, m_fuel_k)
            A_list.append(A_d)
            B_list.append(B_d)
            C_list.append(C_d)
        return A_list, B_list, C_list

    def _solve_scp(self, X_ref, U_ref, A_list, B_list, C_list):
        """求解 SOCP 子问题."""
        N = self.N
        X = cp.Variable((6, N + 1))
        U = cp.Variable((2, N))

        # 代价
        cost = 0
        for k in range(N):
            cost += cp.quad_form(X[:, k] - X_ref[:, k], cp.psd_wrap(self.Q))
            cost += cp.quad_form(U[:, k] - U_ref[:, k], cp.psd_wrap(self.R))
        cost += cp.quad_form(X[:, N] - self.X_term, cp.psd_wrap(self.Qf))

        # 约束
        constraints = [X[:, 0] == self.X0]

        for k in range(N):
            # 动力学
            constraints.append(
                X[:, k + 1] == A_list[k] @ X[:, k] + B_list[k] @ U[:, k] + C_list[k])
            # 控制约束 (缺陷12: T>=T_idle)
            constraints.append(U[0, k] >= T_IDLE)
            constraints.append(U[0, k] <= T_MAX)
            constraints.append(U[1, k] >= THETA_CMD_MIN)
            constraints.append(U[1, k] <= THETA_CMD_MAX)
            # 信赖域 (缺陷16: |Δθ|<10°, |Δvx|<30, |Δvz|<30, |ΔT|<0.3·T_max)
            constraints.append(cp.norm(X[4, k] - X_ref[4, k], 'inf') <= self.trust_theta)
            constraints.append(cp.norm(X[2:4, k] - X_ref[2:4, k], 'inf') <= self.trust_v)
            constraints.append(cp.norm(U[0, k] - U_ref[0, k], 'inf') <= self.trust_T)
            constraints.append(cp.norm(X[:2, k] - X_ref[:2, k], 'inf') <= self.trust_pos)
            # θ_cmd变化率约束 (缺陷13: 防θ_cmd 1步跳80°, PD跟踪不了)
            if k > 0:
                constraints.append(cp.norm(U[1, k] - U[1, k-1], 'inf') <= np.deg2rad(5.0))

        # 求解 (CLARABEL精度更好, SCS有时inaccurate)
        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception as e:
            return None, None, False, str(e)

        if prob.status not in ('optimal', 'optimal_inaccurate'):
            return None, None, False, f'status={prob.status}'

        return X.value, U.value, True, ''

    def solve(self):
        """SCvx 主循环."""
        X_ref, U_ref = self._init_trajectory()

        for iteration in range(self.max_iter):
            self.iter_count = iteration + 1
            A_list, B_list, C_list = self._linearize(X_ref, U_ref)
            X_opt, U_opt, success, msg = self._solve_scp(
                X_ref, U_ref, A_list, B_list, C_list)

            if not success:
                if self.verbose:
                    print(f'  iter {iteration+1}: SOCP失败 - {msg}')
                return X_ref, U_ref, False, {'reason': f'socp_failed: {msg}',
                                              'iter': iteration + 1}

            # 代价
            cost = 0
            for k in range(self.N):
                cost += float((X_opt[:, k] - X_ref[:, k]) @ self.Q @ (X_opt[:, k] - X_ref[:, k]))
                cost += float((U_opt[:, k] - U_ref[:, k]) @ self.R @ (U_opt[:, k] - U_ref[:, k]))
            cost += float((X_opt[:, self.N] - self.X_term) @ self.Qf @ (X_opt[:, self.N] - self.X_term))
            self.cost_history.append(cost)

            dx = np.max(np.abs(X_opt - X_ref))
            du = np.max(np.abs(U_opt - U_ref))

            if self.verbose:
                print(f'  iter {iteration+1}: cost={cost:.4e}, dx={dx:.4f}, du={du:.4f}')

            if dx < self.conv_tol and du < self.conv_tol:
                self.converged = True
                if self.verbose:
                    print(f'  收敛 @ iter {iteration+1}')
                return X_opt, U_opt, True, {'iter': iteration + 1, 'cost': cost,
                                            'cost_history': self.cost_history}

            if iteration > 0 and cost > 2.0 * self.cost_history[-2]:
                if self.verbose:
                    print(f'  熔断: 代价增大')
                return X_opt, U_opt, False, {'reason': 'cost_diverge',
                                              'iter': iteration + 1, 'cost': cost}

            X_ref = X_opt
            U_ref = U_opt

        if self.verbose:
            print(f'  熔断: 超过max_iter={self.max_iter}')
        return X_ref, U_ref, False, {'reason': 'max_iter_exceeded',
                                      'iter': self.max_iter, 'cost': self.cost_history[-1]}


def simulate_scvx_trajectory_7c2(X0, U_opt, dt, m_fuel_init=M_FUEL_INIT):
    """用非线性6维动力学仿真 SCvx 输出的控制序列."""
    N = len(U_opt)
    state = np.array(X0, dtype=float)
    m_fuel = m_fuel_init

    times = np.zeros(N + 1)
    states = np.zeros((6, N + 1))
    thrusts = np.zeros(N)
    theta_cmds = np.zeros(N)
    machs = np.zeros(N + 1)
    alphas = np.zeros(N + 1)

    states[:, 0] = state

    for k in range(N):
        U = U_opt[k]
        k1 = dynamics_7c2(state, U, m_fuel)
        k2 = dynamics_7c2(state + 0.5 * dt * k1, U, m_fuel)
        k3 = dynamics_7c2(state + 0.5 * dt * k2, U, m_fuel)
        k4 = dynamics_7c2(state + dt * k3, U, m_fuel)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        m_fuel -= U[0] * dt / (ISP * G0_ISP)
        m_fuel = max(m_fuel, 0.0)
        if state[1] < 0:
            state[1] = 0.0

        times[k + 1] = (k + 1) * dt
        states[:, k + 1] = state
        thrusts[k] = U[0]
        theta_cmds[k] = U[1]

        V = np.sqrt(state[2] ** 2 + state[3] ** 2)
        rho, a_sound, p, T_air = atmosphere(state[1])
        machs[k + 1] = V / a_sound if a_sound > 0 else 0.0
        alpha, gamma = angle_of_attack(state[4], state[2], state[3])
        alphas[k + 1] = alpha

    V0 = np.sqrt(states[2, 0] ** 2 + states[3, 0] ** 2)
    machs[0] = V0 / atmosphere(states[1, 0])[1]
    alphas[0] = angle_of_attack(states[4, 0], states[2, 0], states[3, 0])[0]

    return {
        't': times, 'x': states[0], 'h': states[1],
        'vx': states[2], 'vz': states[3], 'theta': states[4], 'q': states[5],
        'T': thrusts, 'theta_cmd': theta_cmds,
        'Mach': machs, 'alpha': alphas,
        'm_fuel_final': m_fuel,
    }


# =====================================================================
# Step 7C-3: 完整 sin²(α) 凸化 (9.0 最终目标)
# =====================================================================
# 解析 Jacobian (与 7C-2 数值 Jacobian 对比验证精度)
# 显式仿射表达式 (cvxpy DCP 合规, 不依赖数值差分)
#
# 缺陷17: Q·CD 一阶 Taylor: D ≈ Q̄·CD_lin + CD̄·Q_lin - Q̄·CD̄
# 缺陷18: γ偏导: ∂γ/∂vx=vz/(vx²+vz²), ∂γ/∂vz=-vx/(vx²+vz²)
# 缺陷19: sin(2·85°)≈0.17 敏感度消失 (配平点附近, 论文讨论)
# 缺陷20: CL的cos(2α)展开·0.5失速因子
#
# 凸化数学推导 (9.0 核心):
#   α = θ - γ,  Δα = Δθ - (∂γ/∂vx·Δvx + ∂γ/∂vz·Δvz)
#   sin²(α) ≈ sin²(ᾱ) + sin(2ᾱ)·Δα                    ← 三角恒等式
#   CD ≈ CD0 + CDα·[sin²(ᾱ) + sin(2ᾱ)·Δα]              ← 线性仿射
#   CL ≈ CLα·[sin(2ᾱ) + cos(2ᾱ)·Δα]·0.5                ← 线性仿射
#   Q ≈ Q̄ + ρ·S·(vx̄·Δvx + vz̄·Δvz)                       ← 线性仿射
#   D = Q·CD ≈ Q̄·CD_lin + CD̄·Q_lin - Q̄·CD̄              ← 仿射!
#   T·sin(θ) ≈ T̄·sin(θ̄) + sin(θ̄)·ΔT + T̄·cos(θ̄)·Δθ     ← 仿射!
#   T·cos(θ) ≈ T̄·cos(θ̄) + cos(θ̄)·ΔT - T̄·sin(θ̄)·Δθ     ← 仿射!


def analytical_jacobian_7c3(state, U, m_fuel):
    """
    Step 7C-3 解析 Jacobian (与 7C-2 数值 Jacobian 对比验证).

    完整解析推导, 处理所有非线性:
      - γ=atan2 偏导 (缺陷18)
      - sin²(α) 三角恒等式 (缺陷15/19)
      - Q·CD 乘积 Taylor (缺陷17)
      - T·sin(θ) 双线性 (缺陷14)

    state: [x, h, vx, vz, θ, q] (6维)
    U: [T, θ_cmd] (2维)
    返回: A (6×6), B (6×2)
    """
    x, h, vx, vz, theta, q = state
    T, theta_cmd = U
    m = get_mass(m_fuel)
    g = gravity(h)

    V2 = vx * vx + vz * vz
    V = np.sqrt(V2)

    # 零速度特殊处理 (无气动力)
    if V < 1e-6:
        A = np.zeros((6, 6))
        A[0, 2] = 1.0           # dx/dt = vx
        A[1, 3] = -1.0          # dh/dt = -vz
        A[3, 1] = -2.0 * G0_SL * (R_EARTH / (R_EARTH + h) ** 3)  # ∂g/∂h
        A[4, 5] = 1.0           # dθ/dt = q
        A[5, 4] = -OMEGA_N_TRACK ** 2
        A[5, 5] = -2.0 * ZETA_TRACK * OMEGA_N_TRACK
        B = np.zeros((6, 2))
        B[5, 1] = OMEGA_N_TRACK ** 2
        return A, B

    # ---- γ = atan2(vx, vz) 偏导 (缺陷18) ----
    # ∂γ/∂vx = vz/(vx²+vz²), ∂γ/∂vz = -vx/(vx²+vz²)
    dgamma_dvx = vz / V2
    dgamma_dvz = -vx / V2

    # ---- α = θ - γ 偏导 ----
    # ∂α/∂θ = 1, ∂α/∂vx = -∂γ/∂vx, ∂α/∂vz = -∂γ/∂vz
    dalpha_dvx = -dgamma_dvx
    dalpha_dvz = -dgamma_dvz

    # ---- 大气与 Mach ----
    rho, a_sound, p, T_air = atmosphere(h)
    M = V / a_sound

    # ---- 气动系数 (Mach sigmoid) ----
    CD0, CDa, CLa, Cma = aero_coefficients(M)

    # ---- α, γ 当前值 ----
    gamma = np.arctan2(vx, vz)
    alpha = theta - gamma

    sin_alpha = np.sin(alpha)
    sin_2alpha = np.sin(2.0 * alpha)
    cos_2alpha = np.cos(2.0 * alpha)

    # ---- CD = CD0 + CDα·sin²(α) ----
    # ∂CD/∂α = CDα·sin(2α)  (三角恒等式: d(sin²α)/dα = sin(2α))
    CD = CD0 + CDa * sin_alpha ** 2
    dCD_dalpha = CDa * sin_2alpha
    dCD_dtheta = dCD_dalpha * 1.0           # ∂α/∂θ = 1
    dCD_dvx = dCD_dalpha * dalpha_dvx
    dCD_dvz = dCD_dalpha * dalpha_dvz

    # ---- CL = CLα·sin(2α)·0.5 (缺陷20: 0.5失速因子) ----
    # ∂CL/∂α = CLα·cos(2α)·2·0.5 = CLα·cos(2α)  (链式法则: d(sin(2α))/dα=2cos(2α), 再乘0.5)
    CL = CLa * sin_2alpha * 0.5
    dCL_dalpha = CLa * cos_2alpha  # 不是 *0.5, 因为 2*0.5=1
    dCL_dtheta = dCL_dalpha * 1.0
    dCL_dvx = dCL_dalpha * dalpha_dvx
    dCL_dvz = dCL_dalpha * dalpha_dvz

    # ---- Q = 0.5·ρ·V²·S (缺陷17: Q·CD Taylor) ----
    # ∂Q/∂vx = ρ·S·vx, ∂Q/∂vz = ρ·S·vz
    Q = 0.5 * rho * V2 * S_REF
    dQ_dvx = rho * S_REF * vx
    dQ_dvz = rho * S_REF * vz

    # ---- D = Q·CD, L = Q·CL (乘积 Taylor) ----
    # ∂D/∂vx = ∂Q/∂vx·CD + Q·∂CD/∂vx
    D = Q * CD
    L = Q * CL
    dD_dvx = dQ_dvx * CD + Q * dCD_dvx
    dD_dvz = dQ_dvz * CD + Q * dCD_dvz
    dD_dtheta = Q * dCD_dtheta
    dL_dvx = dQ_dvx * CL + Q * dCL_dvx
    dL_dvz = dQ_dvz * CL + Q * dCL_dvz
    dL_dtheta = Q * dCL_dtheta

    # ---- 气动力坐标变换 ----
    # Fx_aero = -D·sin(γ) + L·cos(γ)
    # Fz_aero = -D·cos(γ) - L·sin(γ)
    sin_gamma = np.sin(gamma)
    cos_gamma = np.cos(gamma)

    # ∂Fx/∂vx = -∂D/∂vx·sin(γ) - D·cos(γ)·∂γ/∂vx + ∂L/∂vx·cos(γ) - L·sin(γ)·∂γ/∂vx
    dFx_dvx = (-dD_dvx * sin_gamma - D * cos_gamma * dgamma_dvx
               + dL_dvx * cos_gamma - L * sin_gamma * dgamma_dvx)
    dFx_dvz = (-dD_dvz * sin_gamma - D * cos_gamma * dgamma_dvz
               + dL_dvz * cos_gamma - L * sin_gamma * dgamma_dvz)
    dFx_dtheta = -dD_dtheta * sin_gamma + dL_dtheta * cos_gamma

    # ∂Fz/∂vx = -∂D/∂vx·cos(γ) + D·sin(γ)·∂γ/∂vx - ∂L/∂vx·sin(γ) - L·cos(γ)·∂γ/∂vx
    dFz_dvx = (-dD_dvx * cos_gamma + D * sin_gamma * dgamma_dvx
               - dL_dvx * sin_gamma - L * cos_gamma * dgamma_dvx)
    dFz_dvz = (-dD_dvz * cos_gamma + D * sin_gamma * dgamma_dvz
               - dL_dvz * sin_gamma - L * cos_gamma * dgamma_dvz)
    dFz_dtheta = -dD_dtheta * cos_gamma - dL_dtheta * sin_gamma

    # ---- 推力方向 (T·sin(θ), T·cos(θ) 双线性, 缺陷14) ----
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    # ∂(T·sin(θ))/∂T = sin(θ), ∂(T·sin(θ))/∂θ = T·cos(θ)
    # ∂(T·cos(θ))/∂T = cos(θ), ∂(T·cos(θ))/∂θ = -T·sin(θ)

    # ---- 加速度 ----
    # ax = (Fx_aero + T·sin(θ)) / m
    # az = (Fz_aero - T·cos(θ)) / m + g
    # ∂ax/∂θ = (∂Fx/∂θ + T·cos(θ)) / m
    # ∂az/∂θ = (∂Fz/∂θ + T·sin(θ)) / m  (注意: -T·cos(θ) 对 θ 求导 = +T·sin(θ))

    # ---- 构造 A (6×6) ----
    # 状态: [x, h, vx, vz, θ, q]
    # dx/dt = vx
    # dh/dt = -vz
    # dvx/dt = ax
    # dvz/dt = az
    # dθ/dt = q
    # dq/dt = ωn²·(θ_cmd-θ) - 2ζωn·q
    A = np.zeros((6, 6))
    A[0, 2] = 1.0
    A[1, 3] = -1.0
    A[2, 2] = dFx_dvx / m
    A[2, 3] = dFx_dvz / m
    A[2, 4] = (dFx_dtheta + T * cos_theta) / m
    A[3, 2] = dFz_dvx / m
    A[3, 3] = dFz_dvz / m
    A[3, 4] = (dFz_dtheta + T * sin_theta) / m
    A[4, 5] = 1.0
    A[5, 4] = -OMEGA_N_TRACK ** 2
    A[5, 5] = -2.0 * ZETA_TRACK * OMEGA_N_TRACK

    # ---- 构造 B (6×2) ----
    # 控制: [T, θ_cmd]
    # ∂ax/∂T = sin(θ)/m, ∂az/∂T = -cos(θ)/m
    # ∂(dq/dt)/∂θ_cmd = ωn²
    B = np.zeros((6, 2))
    B[2, 0] = sin_theta / m
    B[3, 0] = -cos_theta / m
    B[5, 1] = OMEGA_N_TRACK ** 2

    return A, B


def verify_jacobian_7c3(state, U, m_fuel):
    """
    验证解析 Jacobian 与数值 Jacobian 的一致性.

    返回: (A_analytical, A_numerical, B_analytical, B_numerical, max_err)
    """
    A_analytical, B_analytical = analytical_jacobian_7c3(state, U, m_fuel)
    A_numerical, B_numerical = jacobian_7c2(state, U, m_fuel)

    err_A = np.max(np.abs(A_analytical - A_numerical))
    err_B = np.max(np.abs(B_analytical - B_numerical))
    max_err = max(err_A, err_B)

    return A_analytical, A_numerical, B_analytical, B_numerical, max_err


def discretize_7c3(state, U, dt, m_fuel):
    """离散化 (Euler): 用解析 Jacobian."""
    A, B = analytical_jacobian_7c3(state, U, m_fuel)
    f0 = dynamics_7c2(state, U, m_fuel)  # 动力学相同, 复用 7C-2

    A_d = np.eye(6) + dt * A
    B_d = dt * B
    C_d = dt * (f0 - A @ state - B @ U)

    return A_d, B_d, C_d


class SCvxSolver7C3:
    """
    Step 7C-3 SCvx 求解器: 完整 sin²(α) 凸化, 解析 Jacobian.

    与 7C-2 区别:
      1. 使用解析 Jacobian (精度更高, 无数值差分误差)
      2. 验证缺陷17/18/19/20 全部处理
      3. Kill: 10 次不收敛 → 退回 7C-2

    状态: X = [x, h, vx, vz, θ, q] (6维)
    控制: U = [T, θ_cmd] (2维)
    """

    def __init__(self, X0, X_term, N, dt,
                 m_fuel_init=M_FUEL_INIT, max_iter=10,
                 trust_theta=np.deg2rad(10.0), trust_v=30.0,
                 trust_T=0.3 * T_MAX, trust_pos=200.0,
                 conv_tol=1e-3, verbose=False):
        self.X0 = np.array(X0, dtype=float)
        self.X_term = np.array(X_term, dtype=float)
        self.N = N
        self.dt = dt
        self.m_fuel = m_fuel_init
        # 7C-3 Kill: 10 次不收敛
        self.max_iter = max_iter
        self.trust_theta = trust_theta
        self.trust_v = trust_v
        self.trust_T = trust_T
        self.trust_pos = trust_pos
        self.conv_tol = conv_tol
        self.verbose = verbose

        # 代价权重 (与 7C-2 一致)
        self.Q = np.diag([1.0, 10.0, 1.0, 1.0, 5.0, 1.0])
        self.R = np.diag([0.01, 0.1])
        self.Qf = np.diag([100.0, 1000.0, 100.0, 100.0, 50.0, 10.0])

        self.cost_history = []
        self.converged = False
        self.iter_count = 0
        self.fallback_to_7c2 = False  # Kill: 退回 7C-2

    def _init_trajectory(self):
        """初始化参考轨迹 (与 7C-2 相同)."""
        X_ref = np.zeros((6, self.N + 1))
        U_ref = np.zeros((2, self.N))
        state = self.X0.copy()
        m_fuel = self.m_fuel

        X_ref[:, 0] = state
        theta_cmd_init = np.deg2rad(80.0)
        for k in range(self.N):
            U_ref[0, k] = T_IDLE
            U_ref[1, k] = theta_cmd_init
            k1 = dynamics_7c2(state, U_ref[:, k], m_fuel)
            k2 = dynamics_7c2(state + 0.5 * self.dt * k1, U_ref[:, k], m_fuel)
            k3 = dynamics_7c2(state + 0.5 * self.dt * k2, U_ref[:, k], m_fuel)
            k4 = dynamics_7c2(state + self.dt * k3, U_ref[:, k], m_fuel)
            state = state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            m_fuel -= T_IDLE * self.dt / (ISP * G0_ISP)
            m_fuel = max(m_fuel, 0.0)
            if state[1] < 0:
                state[1] = 0.0
            X_ref[:, k + 1] = state

        return X_ref, U_ref

    def _linearize(self, X_ref, U_ref):
        """在参考轨迹上用解析 Jacobian 线性化."""
        A_list, B_list, C_list = [], [], []
        for k in range(self.N):
            m_fuel_k = self.m_fuel * (1 - k / self.N * 0.3)
            A_d, B_d, C_d = discretize_7c3(X_ref[:, k], U_ref[:, k], self.dt, m_fuel_k)
            A_list.append(A_d)
            B_list.append(B_d)
            C_list.append(C_d)
        return A_list, B_list, C_list

    def _solve_scp(self, X_ref, U_ref, A_list, B_list, C_list):
        """求解 SOCP 子问题 (与 7C-2 相同结构, 但 A/B/C 来自解析 Jacobian)."""
        N = self.N
        X = cp.Variable((6, N + 1))
        U = cp.Variable((2, N))

        cost = 0
        for k in range(N):
            cost += cp.quad_form(X[:, k] - X_ref[:, k], cp.psd_wrap(self.Q))
            cost += cp.quad_form(U[:, k] - U_ref[:, k], cp.psd_wrap(self.R))
        cost += cp.quad_form(X[:, N] - self.X_term, cp.psd_wrap(self.Qf))

        constraints = [X[:, 0] == self.X0]

        for k in range(N):
            constraints.append(
                X[:, k + 1] == A_list[k] @ X[:, k] + B_list[k] @ U[:, k] + C_list[k])
            constraints.append(U[0, k] >= T_IDLE)
            constraints.append(U[0, k] <= T_MAX)
            constraints.append(U[1, k] >= THETA_CMD_MIN)
            constraints.append(U[1, k] <= THETA_CMD_MAX)
            constraints.append(cp.norm(X[4, k] - X_ref[4, k], 'inf') <= self.trust_theta)
            constraints.append(cp.norm(X[2:4, k] - X_ref[2:4, k], 'inf') <= self.trust_v)
            constraints.append(cp.norm(U[0, k] - U_ref[0, k], 'inf') <= self.trust_T)
            constraints.append(cp.norm(X[:2, k] - X_ref[:2, k], 'inf') <= self.trust_pos)
            # 缺陷13: θ_cmd 变化率约束
            if k > 0:
                constraints.append(cp.norm(U[1, k] - U[1, k-1], 'inf') <= np.deg2rad(5.0))

        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception as e:
            return None, None, False, str(e)

        if prob.status not in ('optimal', 'optimal_inaccurate'):
            return None, None, False, f'status={prob.status}'

        return X.value, U.value, True, ''

    def solve(self):
        """
        SCvx 主循环.

        Kill Criteria: 10 次不收敛 → 退回 7C-2 (fallback_to_7c2=True)
        """
        X_ref, U_ref = self._init_trajectory()

        for iteration in range(self.max_iter):
            self.iter_count = iteration + 1
            A_list, B_list, C_list = self._linearize(X_ref, U_ref)
            X_opt, U_opt, success, msg = self._solve_scp(
                X_ref, U_ref, A_list, B_list, C_list)

            if not success:
                if self.verbose:
                    print(f'  iter {iteration+1}: SOCP失败 - {msg}')
                # 7C-3 Kill: 退回 7C-2
                self.fallback_to_7c2 = True
                return X_ref, U_ref, False, {'reason': f'socp_failed: {msg}',
                                              'iter': iteration + 1,
                                              'fallback_to_7c2': True}

            cost = 0
            for k in range(self.N):
                cost += float((X_opt[:, k] - X_ref[:, k]) @ self.Q @ (X_opt[:, k] - X_ref[:, k]))
                cost += float((U_opt[:, k] - U_ref[:, k]) @ self.R @ (U_opt[:, k] - U_ref[:, k]))
            cost += float((X_opt[:, self.N] - self.X_term) @ self.Qf @ (X_opt[:, self.N] - self.X_term))
            self.cost_history.append(cost)

            dx = np.max(np.abs(X_opt - X_ref))
            du = np.max(np.abs(U_opt - U_ref))

            if self.verbose:
                print(f'  iter {iteration+1}: cost={cost:.4e}, dx={dx:.4f}, du={du:.4f}')

            if dx < self.conv_tol and du < self.conv_tol:
                self.converged = True
                if self.verbose:
                    print(f'  收敛 @ iter {iteration+1}')
                return X_opt, U_opt, True, {'iter': iteration + 1, 'cost': cost,
                                            'cost_history': self.cost_history}

            if iteration > 0 and cost > 2.0 * self.cost_history[-2]:
                if self.verbose:
                    print(f'  熔断: 代价增大')
                self.fallback_to_7c2 = True
                return X_opt, U_opt, False, {'reason': 'cost_diverge',
                                              'iter': iteration + 1, 'cost': cost,
                                              'fallback_to_7c2': True}

            X_ref = X_opt
            U_ref = U_opt

        # 7C-3 Kill: 超过 max_iter=10 不收敛 → 退回 7C-2
        if self.verbose:
            print(f'  7C-3 Kill: 超过 max_iter={self.max_iter}, 退回 7C-2')
        self.fallback_to_7c2 = True
        return X_ref, U_ref, False, {'reason': 'max_iter_exceeded_7c3',
                                      'iter': self.max_iter,
                                      'cost': self.cost_history[-1] if self.cost_history else 0,
                                      'fallback_to_7c2': True}


def analyze_reef19_sensitivity():
    """
    缺陷19 论文讨论: sin(2·85°)≈0.17 敏感度消失.

    在配平点 α=85° 附近, CD 对 α 的偏导:
      ∂CD/∂α = CDα·sin(2α)
      在 α=85°: sin(170°) ≈ 0.1736
      在 α=45°: sin(90°) = 1.0 (最大敏感度)
      在 α=0°:  sin(0°) = 0 (无敏感度)

    这意味着 SCvx 在配平点附近会认为"改 θ 没用", 只调推力.
    这不是 bug, 是物理本质: 配平点就是力矩平衡点, 改变 α 不会显著改变阻力.

    论文讨论:
      1. 配平点附近敏感度消失是物理本质, 非凸化缺陷
      2. SCvx 在配平点附近会自动增加推力调节权重
      3. 实际飞行中, 翻转段(α: 80°→0°)会经过敏感度区域, 不影响全局
      4. 7C-2 的 θ_cmd 变化[4.2°, 85°] 证明 SCvx 仍会主动调 θ_cmd

    返回: 敏感度分析数据
    """
    alpha_deg = np.arange(0, 91, 1)
    alpha_rad = np.deg2rad(alpha_deg)
    sin_2alpha = np.sin(2 * alpha_rad)

    # 配平点 α=85° 的敏感度
    sens_trim = np.sin(2 * np.deg2rad(85.0))
    # 最大敏感度 α=45°
    sens_max = np.sin(2 * np.deg2rad(45.0))
    # 比值
    ratio = sens_trim / sens_max

    return {
        'alpha_deg': alpha_deg,
        'sin_2alpha': sin_2alpha,
        'sens_at_trim_85deg': sens_trim,
        'sens_at_max_45deg': sens_max,
        'ratio_trim_to_max': ratio,
        'discussion': (
            "缺陷19: 配平点 α=85° 附近 sin(2α)≈0.17, CD 对 α 敏感度仅为最大值的 17.4%. "
            "SCvx 在配平点附近会认为'改 θ 没用', 只调推力. "
            "这不是凸化缺陷, 是物理本质: 配平点就是力矩平衡点. "
            "实际飞行中翻转段(α:80°→0°)会经过敏感度区域, 不影响全局优化. "
            "7C-2 的 θ_cmd 变化[4.2°,85°]证明 SCvx 仍会主动调 θ_cmd."
        )
    }
