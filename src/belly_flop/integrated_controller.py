"""
Belly-Flop Step 7E: 全程集成控制器.
=====================================
理论方案 9.0-Final Step 7E.

缺陷23: 三阶段切换状态丢失 → 统一状态结构体, 只改控制器
缺陷24: C++ sigmoid数值差异 → 用tanh()替代sigmoid (见C++翻译)

统一状态结构体 (全阶段一致, 切换时不重置):
  State = [x, h, vx, vz, θ, q, m_fuel]  (7维)
  Control = (T, θ_cmd, δ_extra_fwd, δ_extra_aft)

三阶段状态机:
  BELLY:   θ_cmd=85°(腹部朝下气动减速), T=0, PD阻尼
           切换条件: tgo≤15s 且 V<250m/s 且 h>3km
  FLIP:    bang-bang+PD+前馈 (7D FlipController)
           切换条件: α<10° 且 h>500m
  LANDING: θ_cmd=0°(垂直), T=bang-bang匀减速, PD阻尼
           切换条件: h≤0 (着陆)

关键设计 (缺陷23应对):
  1. 状态结构体全阶段统一, 切换时只改控制器, 不重置状态
  2. FlipController在FLIP阶段接管, 用当前state初始化plan()
  3. 翻转完成后, LANDING控制器从当前state继续, 无状态丢失
  4. 斜坡过渡: BELLY→FLIP和FLIP→LANDING切换时θ_cmd平滑过渡
"""
import numpy as np
from .aero_model import (
    angle_of_attack, trim_flaps, aero_coefficients,
    get_mass, get_Iyy, gravity, atmosphere,
    S_REF, L_REF, T_MAX, T_IDLE, ISP, G0_ISP,
    M_FUEL_INIT, M_DRY, C_DELTA_FWD, C_DELTA_AFT, DELTA_MAX,
)
from .flip_controller import FlipController, THETA_BELLY, THETA_LAND, T_FLIP_MAX
from .controller import (
    TGO_FLIP_TRIGGER, V_FLIP_TRIGGER, H_FLIP_MIN,
    ALPHA_LAND_TRIGGER, H_LAND_MIN, H_FLIP_KILL,
    ENERGY_KILL_RATIO, VZ_LAND_TARGET, VZ_LAND_MAX,
    RAMP_TRANSITION, pd_gains,
)


class IntegratedBellyFlopController:
    """
    Step 7E 全程集成控制器: BELLY → FLIP → LANDING.

    缺陷23: 统一状态结构体, 三阶段切换只改控制器不重置状态.

    接口: update(state, dt) -> (T, theta_cmd, d_extra_fwd, d_extra_aft, phase, kill, info)
    """

    def __init__(self):
        self.phase = 'BELLY'
        self.phase_t = 0.0
        self.theta_cmd_current = THETA_BELLY
        self.theta_cmd_target = THETA_BELLY
        self.ramp_active = False
        self.ramp_start = THETA_BELLY
        self.ramp_end = THETA_BELLY
        self.ramp_t = 0.0

        # FlipController (FLIP阶段使用)
        self.flip_ctrl = None
        self.flip_initialized = False

        # LANDING阶段计时
        self.landing_t = 0.0

    def _compute_tgo(self, state):
        """tgo = (h - H_LAND_MIN) / |vz|."""
        x, h, vx, vz = state[0], state[1], state[2], state[3]
        if abs(vz) < 1.0:
            return 999.0
        h_to_land = h - H_LAND_MIN
        if h_to_land <= 0:
            return 0.0
        return h_to_land / abs(vz)

    def _energy_check(self, state):
        """能量检查 (缺陷9)."""
        x, h, vx, vz, theta, q, m_fuel = state
        V = np.sqrt(vx ** 2 + vz ** 2)
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
        """2秒斜坡过渡 (缺陷7)."""
        if self.ramp_active:
            self.ramp_t += dt
            alpha = min(1.0, self.ramp_t / RAMP_TRANSITION)
            self.theta_cmd_current = self.ramp_start + alpha * (self.ramp_end - self.ramp_start)
            if alpha >= 1.0:
                self.ramp_active = False
                self.theta_cmd_current = self.ramp_end
        else:
            self.theta_cmd_current = self.theta_cmd_target

    def _start_ramp(self, new_target):
        """启动斜坡过渡."""
        if abs(new_target - self.theta_cmd_current) < 1e-6:
            self.theta_cmd_current = new_target
            self.theta_cmd_target = new_target
            self.ramp_active = False
        else:
            self.ramp_start = self.theta_cmd_current
            self.ramp_end = new_target
            self.theta_cmd_target = new_target
            self.ramp_t = 0.0
            self.ramp_active = True

    def _belly_control(self, state):
        """BELLY阶段: θ_cmd=85°, T=0."""
        return 0.0, THETA_BELLY

    def _landing_control(self, state):
        """LANDING阶段: θ_cmd动态倾斜阻尼水平速度, T=bang-bang匀减速."""
        x, h, vx, vz, theta, q, m_fuel = state
        m = get_mass(m_fuel)
        g = gravity(h)

        # 上升时关机
        if vz < -1.0:
            return 0.0, THETA_LAND

        # 下降时匀减速剖面
        h_eff = max(h, 1.0)
        a_brake = (vz ** 2 - VZ_LAND_TARGET ** 2) / (2.0 * h_eff)
        T_needed = m * (g + a_brake)
        T = np.clip(T_needed, 0.0, T_MAX)

        # 水平速度阻尼
        if abs(vz) > 1.0:
            theta_cmd = np.clip(-0.5 * np.arctan2(vx, abs(vz)),
                                -np.deg2rad(10.0), np.deg2rad(10.0))
        else:
            theta_cmd = THETA_LAND

        return T, theta_cmd

    def _compute_pd_damping(self, state, theta_cmd, M):
        """PD主动阻尼 (缺陷3)."""
        theta = state[4]
        q = state[5]
        Kp, Kd = pd_gains(M)

        e_theta = theta_cmd - theta
        e_theta = (e_theta + np.pi) % (2 * np.pi) - np.pi

        delta_extra = Kp * e_theta - Kd * q
        return delta_extra, delta_extra

    def update(self, state, dt):
        """
        控制器更新 (统一接口, 全阶段一致).

        state: [x, h, vx, vz, θ, q, m_fuel] (7维)
        返回: (T, theta_cmd, d_extra_fwd, d_extra_aft, phase, kill, info)
        """
        x, h, vx, vz, theta, q, m_fuel = state
        V = np.sqrt(vx ** 2 + vz ** 2)
        alpha, gamma = angle_of_attack(theta, vx, vz)
        rho, a_sound, p, T_air = atmosphere(h)
        M = V / a_sound if a_sound > 0 else 0.0

        self.phase_t += dt
        info = {'phase': self.phase, 'V': V, 'h': h, 'Mach': M, 'alpha': alpha}

        # ============ 阶段切换 ============
        if self.phase == 'BELLY':
            tgo = self._compute_tgo(state)
            if tgo <= TGO_FLIP_TRIGGER and V < V_FLIP_TRIGGER and h > H_FLIP_MIN:
                self.phase = 'FLIP'
                self.phase_t = 0.0
                # 缺陷23: 不重置状态, FlipController从当前state初始化
                self.flip_ctrl = FlipController(
                    theta0=theta,  # 从当前theta开始翻转 (非硬编码85°)
                    thetaf=THETA_LAND)
                self.flip_ctrl.plan(state, m_fuel)
                self.flip_initialized = True
                info['phase_transition'] = 'BELLY→FLIP'

        elif self.phase == 'FLIP':
            # 翻转完成检查: α < 10° 且 h > 500m
            if self.flip_ctrl is not None and self.flip_ctrl.is_complete(state):
                self.phase = 'LANDING'
                self.phase_t = 0.0
                self.landing_t = 0.0
                self._start_ramp(THETA_LAND)
                info['phase_transition'] = 'FLIP→LANDING'

            # Kill: 翻转后 h < 800m
            if h < H_FLIP_KILL:
                return (0.0, THETA_LAND, 0.0, 0.0, self.phase, True,
                        {**info, 'kill_reason': f'flip_too_low (h={h:.0f}m<{H_FLIP_KILL}m)'})

            # Kill: 翻转超时
            if self.flip_ctrl is not None and self.flip_ctrl.is_timeout():
                return (0.0, THETA_LAND, 0.0, 0.0, self.phase, True,
                        {**info, 'kill_reason': f'flip_timeout (t={self.flip_ctrl.flip_t:.1f}s>{T_FLIP_MAX}s)'})

        elif self.phase == 'LANDING':
            self.landing_t += dt

        # ============ 能量检查 ============
        if self.phase == 'FLIP' or (self.phase == 'LANDING' and h > 200.0):
            kill, reason = self._energy_check(state)
            if kill:
                return (0.0, THETA_LAND, 0.0, 0.0, self.phase, True,
                        {**info, 'kill_reason': reason})

        # ============ 各阶段控制律 ============
        if self.phase == 'BELLY':
            T, theta_cmd_target = self._belly_control(state)
            # 斜坡过渡
            if abs(theta_cmd_target - self.theta_cmd_target) > 1e-6 and not self.ramp_active:
                self._start_ramp(theta_cmd_target)
            self._ramp_update(dt)
            theta_cmd = self.theta_cmd_current
            # PD阻尼
            d_extra_fwd, d_extra_aft = self._compute_pd_damping(state, theta_cmd, M)

        elif self.phase == 'FLIP':
            # 缺陷23: FlipController接管, 用7D的bang-bang+PD+前馈
            T, theta_cmd, d_extra_fwd, d_extra_aft = self.flip_ctrl.control(state, dt)
            # FLIP阶段不需要额外斜坡过渡 (bang-bang本身就是平滑的)
            self.theta_cmd_current = theta_cmd
            self.theta_cmd_target = theta_cmd
            self.ramp_active = False

        else:  # LANDING
            T, theta_cmd_target = self._landing_control(state)
            # 斜坡过渡
            if abs(theta_cmd_target - self.theta_cmd_target) > 1e-6 and not self.ramp_active:
                self._start_ramp(theta_cmd_target)
            self._ramp_update(dt)
            theta_cmd = self.theta_cmd_current
            # PD阻尼
            d_extra_fwd, d_extra_aft = self._compute_pd_damping(state, theta_cmd, M)

        info['theta_cmd'] = theta_cmd
        info['T'] = T
        info['tgo'] = self._compute_tgo(state) if self.phase == 'BELLY' else 0.0

        return T, theta_cmd, d_extra_fwd, d_extra_aft, self.phase, False, info
