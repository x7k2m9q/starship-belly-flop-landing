"""
星舰 6-DOF 三阶段控制器 (Phase 7.0 战役三 — 问题23)
=====================================================
理论方案7.0 问题23: 状态机切换逻辑 BELLY→FLIP→LANDING

三阶段状态机 (6DOF版, 14维状态):
  BELLY:   θ_cmd=85°(腹部朝下气动减速), T=0, 四元数PD维持BELLY姿态
           切换条件: tgo≤15s AND V<250m/s AND h>3km
  FLIP:    bang-bang+PD+前馈, θ从85°→0°, T=T_idle(保持TVC有效)
           切换条件: θ<10° AND h>500m
  LANDING: θ_cmd=0°(垂直), T=bang-bang匀减速, 四元数PD维持垂直
           切换条件: h≤0 (着陆)

6DOF关键设计 (与3DOF的区别):
  1. 状态向量14维: [px,py,pz, vx,vy,vz, qw,qx,qy,qz, p,q,r, m_fuel]
  2. 姿态用四元数, 期望姿态通过euler_angle_to_quat转换
  3. 控制分配4片襟翼 (FL/FR/RL/RR), 物理推导禁pinv
  4. 非理想执行器: Bouc-Wen+死区补偿+速率限制+TVC延迟
  5. AttitudeController6DOF计算力矩, allocate_flaps分配襟翼

阶段切换平滑过渡 (问题7):
  - BELLY→FLIP: bang-bang轨迹本身平滑, 无需额外斜坡
  - FLIP→LANDING: 2秒斜坡过渡θ_cmd (防力矩跳变)
  - 偏置在BELLY→FLIP时2秒内减至0

Kill条件:
  - FLIP后 h<800m: 着陆段不可行
  - 翻转超时 > 8s
  - 能量检查: a_needed > 0.7·a_avail (推力不足撞地)

接口:
  update(state, dt) -> (T_cmd, delta_flaps_actual, tvc_gimbal_actual,
                        phase, kill, info)
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.belly_flop.attitude_control_6dof import AttitudeController6DOF
from src.belly_flop.control_allocation_6dof import allocate_flaps_normalized
from src.belly_flop.actuator_nonideal_6dof import (
    FlapActuatorSuite6DOF, TVCActuator6DOF
)
from src.belly_flop.dynamics_6dof import (
    euler_angle_to_quat, get_pitch_angle_from_quat, get_tilt_angle_from_quat,
)
from src.belly_flop.aero_model_6dof import (
    get_inertia_tensor, get_mass, gravity, atmosphere_6dof,
    S_REF, L_REF, T_MAX, T_IDLE, ISP, G0_ISP,
    M_DRY, M_FUEL_INIT, DELTA_MAX,
)


# =====================================================================
# 阶段切换阈值 (复用3DOF, 问题23)
# =====================================================================
TGO_FLIP_TRIGGER = 15.0       # s, belly→flip的tgo阈值
V_FLIP_TRIGGER = 250.0        # m/s, belly→flip的速度阈值
H_FLIP_MIN = 3000.0           # m, belly→flip的最低高度
THETA_LAND_TRIGGER = np.deg2rad(10.0)  # flip→landing的俯仰角阈值
H_LAND_MIN = 500.0            # m, flip→landing的最低高度
H_FLIP_KILL = 800.0           # m, 翻转后h<此值→Kill

# 翻转参数 (复用3DOF)
THETA_BELLY_DEG = 85.0        # belly阶段俯仰角(度)
THETA_LAND_DEG = 0.0          # landing阶段俯仰角(度)
T_FLIP_MAX = 8.0              # s, 翻转超时Kill阈值
T_FLIP_TARGET = 3.5           # s, 目标翻转时间
M_MARGIN = 0.8                # 力矩裕度

# 斜坡过渡
RAMP_TRANSITION = 2.0         # s, 阶段切换斜坡过渡时间

# 能量检查
ENERGY_KILL_RATIO = 0.7       # a_needed > 0.7·a_avail → Kill

# 着陆参数
VZ_LAND_TARGET = 3.0          # m/s, 着陆目标垂直速度
VZ_LAND_MAX = 10.0            # m/s, 着陆最大垂直速度


# =====================================================================
# bang-bang翻转轨迹 (6DOF版, 复用3DOF逻辑)
# =====================================================================
def compute_max_flip_torque_6dof(state):
    """计算6DOF翻转段最大可用力矩.

    M_max = Q·L·(Cδf+Cδa)·δ_max
    (Q已含S_REF, 与aero_model_6dof.py一致)
    """
    from src.belly_flop.aero_model_6dof import (
        C_DELTA_FWD, C_DELTA_AFT, atmosphere_6dof, S_REF
    )
    vel_n = state[3:6]
    V = np.linalg.norm(vel_n)
    h = -state[2]

    if V < 1e-6 or h < 0:
        return 0.0, 0.0

    rho, a_sound, p, T = atmosphere_6dof(h)
    Q = 0.5 * rho * V ** 2 * S_REF
    M_max = Q * L_REF * (C_DELTA_FWD + C_DELTA_AFT) * DELTA_MAX
    return M_max, Q


def compute_t_switch_6dof(theta0_deg, thetaf_deg, state, m_fuel,
                           t_target=T_FLIP_TARGET):
    """6DOF bang-bang切换时间解析公式 (暗礁21).

    返回: (t_switch, t_total, alpha_max, M_max)
    """
    M_max, Q = compute_max_flip_torque_6dof(state)
    I_tensor = get_inertia_tensor(m_fuel)
    Iyy = I_tensor[1, 1]  # 俯仰惯量

    t_total = t_target
    t_switch = t_total / 2.0

    delta_theta = abs(np.deg2rad(theta0_deg - thetaf_deg))
    alpha_max = delta_theta / (t_switch ** 2) if t_switch > 0 else 0.0

    M_needed = alpha_max * Iyy
    if M_needed > M_max * M_MARGIN:
        # 力矩不足, 延长翻转时间
        t_switch = np.sqrt(delta_theta / (M_max * M_MARGIN / Iyy))
        t_total = 2.0 * t_switch
        alpha_max = delta_theta / (t_switch ** 2)

    return t_switch, t_total, alpha_max, M_max


def bangbang_theta_trajectory_6dof(t, theta0_deg, thetaf_deg,
                                    t_switch, t_total):
    """6DOF bang-bang期望轨迹 (梯形角速度剖面).

    返回: (theta_ref_deg, theta_ref_dot_rad_s)
    """
    delta_theta = np.deg2rad(theta0_deg - thetaf_deg)  # 正值

    if t_total < 1e-6:
        return thetaf_deg, 0.0

    alpha_max = delta_theta / (t_switch ** 2) if t_switch > 0 else 0.0

    if t < 0:
        return theta0_deg, 0.0
    elif t < t_switch:
        # 加速阶段
        theta = np.deg2rad(theta0_deg) - 0.5 * alpha_max * t ** 2
        theta_dot = -alpha_max * t
    elif t < t_total:
        # 减速阶段
        t_dec = t - t_switch
        theta_mid = np.deg2rad(theta0_deg) - 0.5 * alpha_max * t_switch ** 2
        theta = theta_mid - alpha_max * t_switch * t_dec + 0.5 * alpha_max * t_dec ** 2
        theta_dot = -alpha_max * t_switch + alpha_max * t_dec
    else:
        theta = np.deg2rad(thetaf_deg)
        theta_dot = 0.0

    return np.rad2deg(theta), theta_dot


def bangbang_theta_acceleration_6dof(t, t_switch, t_total, alpha_max):
    """6DOF bang-bang参考角加速度 (标称前馈)."""
    if t < 0 or t >= t_total:
        return 0.0
    elif t < t_switch:
        return -alpha_max
    else:
        return alpha_max


# =====================================================================
# 6DOF三阶段控制器
# =====================================================================
class PhaseController6DOF:
    """6-DOF三阶段控制器: BELLY → FLIP → LANDING.

    集成:
      - AttitudeController6DOF: 四元数PD+陷波滤波
      - FlapActuatorSuite6DOF: 4片非理想襟翼(Bouc-Wen+死区补偿)
      - TVCActuator6DOF: TVC延迟+速率限制
      - bang-bang翻转轨迹规划

    接口:
      update(state, dt) -> (T_cmd, delta_flaps_actual, tvc_gimbal_actual,
                            phase, kill, info)

    state: 14维 [px,py,pz, vx,vy,vz, qw,qx,qy,qz, p,q,r, m_fuel]
    """

    def __init__(self, use_nonideal_actuator=True, use_notch=True,
                 wn=2 * np.pi * 0.5, zeta=0.9):
        """
        参数:
          use_nonideal_actuator: 是否使用非理想执行器
            True: Bouc-Wen+死区+速率限制+TVC延迟 (工程级)
            False: 理想执行器 (调试用)
          use_notch: 是否使用陷波滤波器
          wn: 姿态控制器自然频率 (rad/s)
          zeta: 阻尼比
        """
        # 姿态控制器
        self.attitude_ctrl = AttitudeController6DOF(
            wn=wn, zeta=zeta, sample_rate=100.0, use_notch=use_notch
        )

        # 非理想执行器
        self.use_nonideal_actuator = use_nonideal_actuator
        if use_nonideal_actuator:
            self.flap_actuator = FlapActuatorSuite6DOF(
                dead_zones=[np.deg2rad(0.5)] * 4,
                rate_limits=[np.deg2rad(30.0)] * 4,
                use_compensation=True,
                dither_freq=20.0,
                dither_amp=np.deg2rad(1.0)
            )
            self.tvc_actuator = TVCActuator6DOF(
                delay=0.08,
                rate_limit=np.deg2rad(20.0),
                gimbal_limit=np.deg2rad(10.0),
                dt=0.01
            )
        else:
            self.flap_actuator = None
            self.tvc_actuator = None

        # 阶段状态
        self.phase = 'BELLY'
        self.phase_t = 0.0
        self.total_t = 0.0

        # 斜坡过渡
        self.theta_cmd_current_deg = THETA_BELLY_DEG
        self.theta_cmd_target_deg = THETA_BELLY_DEG
        self.ramp_active = False
        self.ramp_start_deg = THETA_BELLY_DEG
        self.ramp_end_deg = THETA_BELLY_DEG
        self.ramp_t = 0.0

        # FLIP轨迹规划
        self.flip_planned = False
        self.flip_t = 0.0
        self.flip_t_switch = 0.0
        self.flip_t_total = 0.0
        self.flip_alpha_max = 0.0
        self.flip_M_max = 0.0
        self.flip_theta0_deg = THETA_BELLY_DEG
        self.flip_thetaf_deg = THETA_LAND_DEG

        # LANDING计时
        self.landing_t = 0.0

        # 诊断
        self.last_info = {}

    def reset(self):
        """重置控制器状态."""
        self.phase = 'BELLY'
        self.phase_t = 0.0
        self.total_t = 0.0
        self.theta_cmd_current_deg = THETA_BELLY_DEG
        self.theta_cmd_target_deg = THETA_BELLY_DEG
        self.ramp_active = False
        self.ramp_t = 0.0
        self.flip_planned = False
        self.flip_t = 0.0
        self.landing_t = 0.0
        self.attitude_ctrl.reset()
        if self.use_nonideal_actuator:
            self.flap_actuator.reset()
            self.tvc_actuator.reset()

    # ============ 辅助函数 ============
    def _compute_tgo(self, state):
        """tgo = (h - H_LAND_MIN) / |vz|."""
        h = -state[2]
        vz = state[5]
        if abs(vz) < 1.0:
            return 999.0
        h_to_land = h - H_LAND_MIN
        if h_to_land <= 0:
            return 0.0
        return h_to_land / abs(vz)

    def _energy_check(self, state):
        """能量检查 (暗礁9)."""
        vel_n = state[3:6]
        V = np.linalg.norm(vel_n)
        h = -state[2]
        m_fuel = state[13]
        m = get_mass(m_fuel)
        g = gravity(h)

        if h < 1.0:
            return False, ''

        a_needed = V ** 2 / (2.0 * h)
        a_avail = T_MAX / m - g
        if a_avail <= 0:
            return True, f'no_thrust_authority (a_avail={a_avail:.1f}<=0)'

        ratio = a_needed / a_avail
        if ratio > ENERGY_KILL_RATIO:
            return True, (f'insufficient_thrust (a_needed={a_needed:.1f}, '
                          f'a_avail={a_avail:.1f}, ratio={ratio:.2f}>{ENERGY_KILL_RATIO})')
        return False, ''

    def _ramp_update(self, dt):
        """2秒斜坡过渡 (暗礁7)."""
        if self.ramp_active:
            self.ramp_t += dt
            alpha = min(1.0, self.ramp_t / RAMP_TRANSITION)
            self.theta_cmd_current_deg = (
                self.ramp_start_deg + alpha * (self.ramp_end_deg - self.ramp_start_deg)
            )
            if alpha >= 1.0:
                self.ramp_active = False
                self.theta_cmd_current_deg = self.ramp_end_deg
        else:
            self.theta_cmd_current_deg = self.theta_cmd_target_deg

    def _start_ramp(self, new_target_deg):
        """启动斜坡过渡."""
        if abs(new_target_deg - self.theta_cmd_current_deg) < 1e-6:
            self.theta_cmd_current_deg = new_target_deg
            self.theta_cmd_target_deg = new_target_deg
            self.ramp_active = False
        else:
            self.ramp_start_deg = self.theta_cmd_current_deg
            self.ramp_end_deg = new_target_deg
            self.theta_cmd_target_deg = new_target_deg
            self.ramp_t = 0.0
            self.ramp_active = True

    def _get_pitch_angle(self, state):
        """从状态提取俯仰角(度)."""
        q = state[6:10]
        theta_rad = get_pitch_angle_from_quat(q)
        return np.rad2deg(theta_rad)

    def _compute_Q_dyn(self, state):
        """计算动压."""
        vel_n = state[3:6]
        V = np.linalg.norm(vel_n)
        h = -state[2]
        if V < 1e-6 or h < 0:
            return 0.0, 0.0
        rho, a_sound, p, T = atmosphere_6dof(h)
        Q = 0.5 * rho * V ** 2 * S_REF
        M = V / a_sound if a_sound > 0 else 0.0
        return Q, M

    # ============ 各阶段控制律 ============
    def _belly_control(self, state):
        """BELLY阶段: θ_cmd=85°, T=0."""
        return 0.0, THETA_BELLY_DEG

    def _flip_plan(self, state):
        """规划FLIP bang-bang轨迹."""
        m_fuel = state[13]
        theta_current = self._get_pitch_angle(state)
        self.flip_theta0_deg = theta_current  # 从当前theta开始
        self.flip_thetaf_deg = THETA_LAND_DEG

        self.flip_t_switch, self.flip_t_total, self.flip_alpha_max, self.flip_M_max = \
            compute_t_switch_6dof(
                self.flip_theta0_deg, self.flip_thetaf_deg,
                state, m_fuel, t_target=T_FLIP_TARGET
            )
        self.flip_planned = True
        self.flip_t = 0.0

    def _flip_control(self, state, dt):
        """FLIP阶段: bang-bang+PD+前馈, T=T_idle."""
        self.flip_t += dt

        # bang-bang期望轨迹
        theta_ref_deg, theta_ref_dot = bangbang_theta_trajectory_6dof(
            self.flip_t, self.flip_theta0_deg, self.flip_thetaf_deg,
            self.flip_t_switch, self.flip_t_total
        )

        # bang-bang参考加速度 (标称前馈)
        theta_ref_ddot = bangbang_theta_acceleration_6dof(
            self.flip_t, self.flip_t_switch, self.flip_t_total, self.flip_alpha_max
        )

        return T_IDLE, theta_ref_deg, theta_ref_dot, theta_ref_ddot

    def _landing_control(self, state):
        """LANDING阶段: θ_cmd=0°, T=bang-bang匀减速."""
        vel_n = state[3:6]
        vx = vel_n[0]
        vz = vel_n[2]
        h = -state[2]
        m_fuel = state[13]
        m = get_mass(m_fuel)
        g = gravity(h)

        # 上升时关机
        if vz < -1.0:
            return 0.0, THETA_LAND_DEG

        # 下降时匀减速剖面
        h_eff = max(h, 1.0)
        a_brake = (vz ** 2 - VZ_LAND_TARGET ** 2) / (2.0 * h_eff)
        T_needed = m * (g + a_brake)
        T = np.clip(T_needed, 0.0, T_MAX)

        # 水平速度阻尼: θ_cmd向vx反方向倾斜, 限幅±10°
        if abs(vz) > 1.0:
            theta_cmd_deg = np.rad2deg(np.clip(
                -0.5 * np.arctan2(vx, abs(vz)),
                -np.deg2rad(10.0), np.deg2rad(10.0)
            ))
        else:
            theta_cmd_deg = THETA_LAND_DEG

        return T, theta_cmd_deg

    def _is_flip_complete(self, state):
        """翻转完成判断: θ<10° AND |q|<0.05rad/s."""
        theta_deg = self._get_pitch_angle(state)
        omega_b = state[10:13]
        q_rate = omega_b[1]  # 俯仰角速度

        theta_err = abs(np.deg2rad(theta_deg - THETA_LAND_DEG))
        q_small = abs(q_rate) < 0.05

        if self.flip_t > self.flip_t_total + 2.0:
            return True
        if theta_err < THETA_LAND_TRIGGER and q_small:
            return True
        return False

    # ============ 主控制循环 ============
    def update(self, state, dt):
        """控制器更新.

        参数:
          state: 14维 [px,py,pz, vx,vy,vz, qw,qx,qy,qz, p,q,r, m_fuel]
          dt: 时间步长 (s)

        返回:
          T_cmd: 推力指令 (N)
          delta_flaps_actual: 4片襟翼实际偏转 [d_FL,d_FR,d_RL,d_RR] (rad)
          tvc_gimbal_actual: TVC实际偏转 [gimbal_y, gimbal_z] (rad)
          phase: 当前阶段 ('BELLY'/'FLIP'/'LANDING')
          kill: 是否触发Kill (True=终止)
          info: 诊断信息dict
        """
        self.total_t += dt
        self.phase_t += dt

        # 提取状态
        vel_n = state[3:6]
        V = np.linalg.norm(vel_n)
        h = -state[2]
        m_fuel = state[13]
        theta_deg = self._get_pitch_angle(state)
        Q_dyn, M_mach = self._compute_Q_dyn(state)

        info = {
            'phase': self.phase, 'V': V, 'h': h, 'Mach': M_mach,
            'theta_deg': theta_deg, 't': self.total_t, 'Q_dyn': Q_dyn
        }

        # ============ 阶段切换 ============
        if self.phase == 'BELLY':
            tgo = self._compute_tgo(state)
            if tgo <= TGO_FLIP_TRIGGER and V < V_FLIP_TRIGGER and h > H_FLIP_MIN:
                self.phase = 'FLIP'
                self.phase_t = 0.0
                self._flip_plan(state)
                info['phase_transition'] = 'BELLY→FLIP'
                info['flip_t_switch'] = self.flip_t_switch
                info['flip_t_total'] = self.flip_t_total

        elif self.phase == 'FLIP':
            # 翻转完成检查
            if self.flip_planned and self._is_flip_complete(state):
                self.phase = 'LANDING'
                self.phase_t = 0.0
                self.landing_t = 0.0
                self._start_ramp(THETA_LAND_DEG)
                info['phase_transition'] = 'FLIP→LANDING'

            # Kill: 翻转后 h < 800m
            if h < H_FLIP_KILL:
                return (0.0, np.zeros(4), np.zeros(2), self.phase, True,
                        {**info, 'kill_reason': f'flip_too_low (h={h:.0f}m<{H_FLIP_KILL}m)'})

            # Kill: 翻转超时
            if self.flip_t > T_FLIP_MAX:
                return (0.0, np.zeros(4), np.zeros(2), self.phase, True,
                        {**info, 'kill_reason': f'flip_timeout (t={self.flip_t:.1f}s>{T_FLIP_MAX}s)'})

        elif self.phase == 'LANDING':
            self.landing_t += dt

        # ============ 能量检查 ============
        if self.phase == 'FLIP' or (self.phase == 'LANDING' and h > 200.0):
            kill, reason = self._energy_check(state)
            if kill:
                return (0.0, np.zeros(4), np.zeros(2), self.phase, True,
                        {**info, 'kill_reason': reason})

        # ============ 各阶段控制律 ============
        omega_des_dot = np.zeros(3)  # 期望角加速度(前馈)

        if self.phase == 'BELLY':
            T_cmd, theta_cmd_target_deg = self._belly_control(state)
            if abs(theta_cmd_target_deg - self.theta_cmd_target_deg) > 1e-6 and not self.ramp_active:
                self._start_ramp(theta_cmd_target_deg)
            self._ramp_update(dt)
            theta_cmd_deg = self.theta_cmd_current_deg
            omega_des = np.zeros(3)

        elif self.phase == 'FLIP':
            T_cmd, theta_cmd_deg, theta_ref_dot, theta_ref_ddot = self._flip_control(state, dt)
            # 期望角速度: 只有俯仰分量
            omega_des = np.array([0.0, theta_ref_dot, 0.0])
            # 期望角加速度(前馈): 只有俯仰分量
            omega_des_dot = np.array([0.0, theta_ref_ddot, 0.0])
            self.theta_cmd_current_deg = theta_cmd_deg
            self.theta_cmd_target_deg = theta_cmd_deg
            self.ramp_active = False

        else:  # LANDING
            T_cmd, theta_cmd_target_deg = self._landing_control(state)
            if abs(theta_cmd_target_deg - self.theta_cmd_target_deg) > 1e-6 and not self.ramp_active:
                self._start_ramp(theta_cmd_target_deg)
            self._ramp_update(dt)
            theta_cmd_deg = self.theta_cmd_current_deg
            omega_des = np.zeros(3)

        # ============ 姿态控制器 ============
        q_actual = state[6:10]
        omega_actual = state[10:13]
        I_body = get_inertia_tensor(m_fuel)

        # 期望四元数
        q_des = euler_angle_to_quat(theta_cmd_deg)

        # 计算力矩指令
        M_cmd, delta_flaps_cmd, tvc_gimbal_cmd = self.attitude_ctrl.compute_torque(
            q_des, omega_des, q_actual, omega_actual,
            I_body, m_fuel, Q_dyn, omega_des_dot
        )

        # ============ 非理想执行器 ============
        if self.use_nonideal_actuator:
            # 4片襟翼非理想执行器
            delta_flaps_actual = self.flap_actuator.update(delta_flaps_cmd, dt)
            # TVC非理想执行器
            tvc_gimbal_actual = self.tvc_actuator.update(tvc_gimbal_cmd, dt)
        else:
            # 理想执行器 (调试用)
            delta_flaps_actual = delta_flaps_cmd
            tvc_gimbal_actual = tvc_gimbal_cmd

        # ============ 诊断信息 ============
        info['T_cmd'] = T_cmd
        info['theta_cmd_deg'] = theta_cmd_deg
        info['M_cmd'] = M_cmd.copy()
        info['delta_flaps_cmd'] = delta_flaps_cmd.copy()
        info['delta_flaps_actual'] = delta_flaps_actual.copy()
        info['tvc_gimbal_cmd'] = tvc_gimbal_cmd.copy()
        info['tvc_gimbal_actual'] = tvc_gimbal_actual.copy()
        info['q_des'] = q_des.copy()
        info['omega_des'] = omega_des.copy()
        info['tgo'] = self._compute_tgo(state) if self.phase == 'BELLY' else 0.0
        info['flip_t'] = self.flip_t if self.phase == 'FLIP' else 0.0

        self.last_info = info.copy()

        return T_cmd, delta_flaps_actual, tvc_gimbal_actual, self.phase, False, info


# =====================================================================
# 便捷函数: 全程闭环仿真
# =====================================================================
def run_full_mission_6dof(h_init=10000.0, vz_init=300.0, vx_init=50.0,
                          theta_pitch_deg=85.0, m_fuel=None,
                          t_end=120.0, dt=0.01,
                          use_nonideal_actuator=True,
                          use_mekf=False,
                          seed=42):
    """全程闭环仿真: BELLY → FLIP → LANDING.

    参数:
      h_init: 初始高度 (m)
      vz_init: 初始下降速度 (m/s)
      vx_init: 初始水平速度 (m/s)
      theta_pitch_deg: 初始俯仰角 (度)
      m_fuel: 初始燃料 (kg), None=70%
      t_end: 最大仿真时间 (s)
      dt: 时间步长 (s)
      use_nonideal_actuator: 是否使用非理想执行器
      use_mekf: 是否使用MEKF状态估计 (False=上帝视角)
      seed: 随机种子

    返回:
      result: dict, 含时间序列和诊断信息
    """
    from src.belly_flop.dynamics_6dof import (
        make_initial_state_6dof, rk4_step_6dof
    )

    rng = np.random.default_rng(seed)

    # 初始状态
    state = make_initial_state_6dof(
        h_init=h_init, vz_init=vz_init, vx_init=vx_init,
        theta_pitch_deg=theta_pitch_deg, m_fuel=m_fuel
    )
    state_true = state.copy()

    # 控制器
    controller = PhaseController6DOF(
        use_nonideal_actuator=use_nonideal_actuator,
        use_notch=True
    )

    # MEKF (可选)
    if use_mekf:
        from src.ekf import MEKF
        from src.sensors import IMU, GPS, RadarAltimeter
        imu = IMU(rng)
        gps = GPS(rng)
        radar = RadarAltimeter(rng)
        ekf = MEKF(
            pos0=state_true[0:3],
            vel0=state_true[3:6],
            q0=state_true[6:10],
            dt=dt
        )

    # 记录
    times = []
    states = []
    phases = []
    T_cmds = []
    delta_flaps = []
    tvc_gimbals = []
    pitch_angles = []
    altitudes = []
    velocities = []
    kills = []
    infos = []

    n_steps = int(t_end / dt)
    kill_triggered = False
    kill_reason = ''

    for i in range(n_steps + 1):
        t = i * dt

        # 选择状态源
        if use_mekf:
            state_est, sigma = ekf.get_state()
            # 用估计状态做控制
            state_ctrl = state_true.copy()  # 简化: 控制器用真值, MEKF并行运行
            # 实际工程: state_ctrl = ekf_state
        else:
            state_ctrl = state_true

        # 控制器更新
        T_cmd, delta_flaps_actual, tvc_gimbal_actual, phase, kill, info = \
            controller.update(state_ctrl, dt)

        # 记录
        if i % 10 == 0 or i == n_steps or kill:
            times.append(t)
            states.append(state_true.copy())
            phases.append(phase)
            T_cmds.append(T_cmd)
            delta_flaps.append(delta_flaps_actual.copy())
            tvc_gimbals.append(tvc_gimbal_actual.copy())
            theta_deg = np.rad2deg(get_pitch_angle_from_quat(state_true[6:10]))
            pitch_angles.append(theta_deg)
            altitudes.append(-state_true[2])
            velocities.append(np.linalg.norm(state_true[3:6]))
            kills.append(kill)
            infos.append(info.copy() if isinstance(info, dict) else {})

        if kill:
            kill_triggered = True
            kill_reason = info.get('kill_reason', 'unknown')
            break

        # MEKF更新 (可选)
        if use_mekf:
            # 提取IMU测量
            from src.belly_flop.dynamics_6dof import state_derivative_6dof
            dstate = state_derivative_6dof(
                state_true, T_cmd=T_cmd, delta_flaps=delta_flaps_actual,
                tvc_gimbal=tvc_gimbal_actual
            )
            omega_b_true = state_true[10:13]
            C_bn = quat_to_rotmat(state_true[6:10])
            a_n = dstate[3:6]
            g = gravity(-state_true[2])
            g_n = np.array([0.0, 0.0, g])
            f_n = a_n - g_n
            f_b = C_bn.T @ f_n
            gyro_meas, accel_meas = imu.measure(omega_b_true, f_b, dt)
            ekf.predict(gyro_meas, accel_meas, dt)
            # GPS
            pos_meas, vel_meas, gps_valid = gps.measure(state_true[0:3], state_true[3:6], dt)
            if gps_valid:
                ekf.update_gps(pos_meas, vel_meas)
            # 雷达
            alt_meas, radar_valid = radar.measure(state_true[0:3], dt)
            if radar_valid:
                ekf.update_radar(alt_meas)

        # 动力学积分
        state_true = rk4_step_6dof(
            state_true, T_cmd=T_cmd, delta_flaps=delta_flaps_actual,
            dt=dt, tvc_gimbal=tvc_gimbal_actual
        )

        # 着陆判断
        if -state_true[2] <= 0.0:
            break

    result = {
        't': np.array(times),
        'states': np.array(states),
        'phase': phases,
        'T_cmd': np.array(T_cmds),
        'delta_flaps': np.array(delta_flaps),
        'tvc_gimbal': np.array(tvc_gimbals),
        'pitch_angle': np.array(pitch_angles),
        'altitude': np.array(altitudes),
        'velocity': np.array(velocities),
        'kill': kill_triggered,
        'kill_reason': kill_reason,
        'info': infos,
        'n_steps': i,
        'final_state': state_true.copy(),
    }
    return result
