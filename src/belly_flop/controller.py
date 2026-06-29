"""
Belly-Flop 多阶段控制器 (Step 7B).
=====================================
三阶段状态机: BELLY → FLIP → LANDING

切换条件 (继承8.0修正):
  BELLY → FLIP:    tgo ≤ 15s 且 V < 250m/s 且 h > 3km
  FLIP → LANDING:  α < 10° 且 h > 500m
  Kill: 翻转后 h < 800m (着陆段不可行)
  Kill: a_needed > 0.7·a_avail (推力不足撞地)

各阶段控制:
  BELLY:   θ_cmd=85°(腹部朝下气动减速), T=0, PD阻尼
  FLIP:    θ_cmd从85°→0°(斜坡翻转), T=T_idle, PD阻尼
  LANDING: θ_cmd=0°(垂直), T=bang-bang匀减速, PD阻尼

缺陷应对:
  缺陷7: 阶段切换姿态阶跃 → 2秒斜坡过渡
  缺陷8: PID固定增益 → Mach分3段增益调度
  缺陷9: 翻转后无能量检查 → a_needed=V²/(2h) > 0.7·a_avail → Kill
  缺陷3: 配平点弱阻尼 → PD主动阻尼 (配平襟翼+额外偏转)
"""
import numpy as np
from .aero_model import (
    angle_of_attack, trim_flaps, aero_coefficients,
    get_mass, get_Iyy, gravity, atmosphere,
    S_REF, L_REF, T_MAX, T_IDLE, ISP, G0_ISP,
    M_FUEL_INIT, M_DRY, C_DELTA_FWD, C_DELTA_AFT, DELTA_MAX,
)


# =====================================================================
# 阶段切换阈值 (继承8.0修正)
# =====================================================================
TGO_FLIP_TRIGGER = 15.0       # s, belly→flip的tgo阈值
V_FLIP_TRIGGER = 250.0        # m/s, belly→flip的速度阈值
H_FLIP_MIN = 3000.0           # m, belly→flip的最低高度
ALPHA_LAND_TRIGGER = np.deg2rad(10.0)  # flip→landing的攻角阈值
H_LAND_MIN = 500.0            # m, flip→landing的最低高度
H_FLIP_KILL = 800.0           # m, 翻转后h<此值→Kill (着陆段不可行)

# 翻转参数
THETA_BELLY = np.deg2rad(85.0)   # belly阶段俯仰角
THETA_LAND = 0.0                  # landing阶段俯仰角 (垂直)
T_FLIP = 4.0                      # s, 翻转时间
RAMP_TRANSITION = 2.0             # s, 阶段切换斜坡过渡时间

# 能量检查
ENERGY_KILL_RATIO = 0.7           # a_needed > 0.7·a_avail → Kill

# 着陆参数
VZ_LAND_TARGET = 3.0              # m/s, 着陆目标垂直速度
VZ_LAND_MAX = 10.0                # m/s, 着陆最大垂直速度 (软着陆)


# =====================================================================
# Mach增益调度 (缺陷8: 按Mach分3段, 非调参, 物理依据)
# =====================================================================
def pd_gains(M):
    """
    PD增益调度, 按Mach分3段.

    物理依据 (增益选择基于力矩平衡分析, 非调参):
      翻转过程需克服气动抵抗力矩 M_aero ≈ Q·S·L·Cm
      PD力矩 M_flap ≈ Q·S·L·(Cδf+Cδa)·Kp·e = Q·S·L·1.4·Kp·e
      要求 M_flap > 2·M_aero (足够控制权限):
        Kp > 2·Cm_max / (1.4·e_typical)
        Cm_max ≈ Cma·sin(85°) ≈ 0.25, e_typical ≈ 0.5rad
        Kp > 2·0.25 / (1.4·0.5) = 0.71

      超声速(M>1.2): 动压大, 气动力矩强, 但Cma小(0.20), Kp=0.8
      跨声速(0.8-1.2): Cma大(0.25), 需更大增益, Kp=1.2
      亚声速(M<0.8): Cma=0.15, 居中, Kp=0.8

    返回: (Kp, Kd) 无量纲, Kp=rad/rad, Kd=rad/(rad/s)=s
    """
    if M > 1.2:
        return 0.80, 0.40    # 超声速: 动压大, Cma=0.20
    elif M > 0.8:
        return 1.20, 0.60    # 跨声速: Cma=0.25(最大), 需大增益
    else:
        return 0.80, 0.40    # 亚声速: Cma=0.15, 动压适中


# =====================================================================
# 多阶段控制器
# =====================================================================
class BellyFlopController:
    """
    星舰Belly-Flop多阶段控制器.

    接口: update(state, dt) -> (T, theta_cmd, d_extra_fwd, d_extra_aft, phase, kill, info)
    """

    def __init__(self):
        self.phase = 'BELLY'
        self.phase_t = 0.0           # 当前阶段已运行时间
        self.theta_cmd_current = THETA_BELLY  # 当前实际输出的θ_cmd (含斜坡过渡)
        self.theta_cmd_target = THETA_BELLY   # 目标θ_cmd
        self.ramp_active = False     # 斜坡过渡激活
        self.ramp_start = THETA_BELLY
        self.ramp_end = THETA_BELLY
        self.ramp_t = 0.0
        self.flip_completed = False  # 翻转完成标志
        self.landing_t = 0.0         # landing阶段计时

    def _compute_tgo(self, state):
        """
        tgo = (h - H_LAND_MIN) / |vz| (到landing入口的剩余时间).
        7B阶段用此定义, 7C-1再用1.2·sqrt(h²+x²)/V (缺陷11).
        物理含义: 假设匀速下降到landing入口h=500m的时间.
        """
        x, h, vx, vz = state[0], state[1], state[2], state[3]
        if abs(vz) < 1.0:
            return 999.0
        h_to_land = h - H_LAND_MIN
        if h_to_land <= 0:
            return 0.0
        return h_to_land / abs(vz)

    def _energy_check(self, state):
        """
        能量检查 (缺陷9).
        a_needed = V²/(2·h): 到地面减速到0需要的加速度
        a_avail = T_max/m - g: 最大制动加速度
        a_needed > 0.7·a_avail → Kill (推力不足)
        """
        x, h, vx, vz, theta, q, m_fuel = state
        V = np.sqrt(vx ** 2 + vz ** 2)
        m = get_mass(m_fuel)
        g = gravity(h)

        if h < 1.0:
            return False, ''  # 太低不检查

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
        """2秒斜坡过渡 (缺陷7: 防阶段切换姿态阶跃)."""
        if self.ramp_active:
            self.ramp_t += dt
            alpha = min(1.0, self.ramp_t / RAMP_TRANSITION)
            # 线性插值
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

    def _flip_control(self, state):
        """FLIP阶段: θ_cmd从85°→0°斜坡, T=T_idle."""
        # 翻转进度
        flip_progress = min(1.0, self.phase_t / T_FLIP)
        theta_cmd = THETA_BELLY * (1.0 - flip_progress)
        return T_IDLE, theta_cmd

    def _landing_control(self, state):
        """
        LANDING阶段: θ_cmd动态倾斜阻尼水平速度, T=bang-bang匀减速.

        推力策略 (工程判断, 非调参):
          1. 上升(vz<0): T=0, 让重力拉回 (防推力过大持续上升)
          2. 下降(vz≥0): 匀减速剖面
             a_brake = (vz² - VZ_LAND_TARGET²) / (2·h)
             T_needed = m·(g + a_brake)
          3. 低空(h<10m): 维持终端速度

        姿态策略:
          θ_cmd = -0.5·arctan2(vx, |vz|), 限幅±10°
          向水平速度反方向倾斜, 推力水平分量阻尼vx (工程判断: 着陆需消水平速度)
        """
        x, h, vx, vz, theta, q, m_fuel = state
        m = get_mass(m_fuel)
        g = gravity(h)

        # 上升时关机 (防持续上升)
        if vz < -1.0:
            return 0.0, THETA_LAND

        # 下降时匀减速剖面 (目标: h=0时vz=VZ_LAND_TARGET)
        # h最小取1.0防除零, 低空仍需减速
        h_eff = max(h, 1.0)
        a_brake = (vz ** 2 - VZ_LAND_TARGET ** 2) / (2.0 * h_eff)

        T_needed = m * (g + a_brake)
        T = np.clip(T_needed, 0.0, T_MAX)

        # 水平速度阻尼: θ_cmd向vx反方向倾斜, 限幅±10°
        if abs(vz) > 1.0:
            theta_cmd = np.clip(-0.5 * np.arctan2(vx, abs(vz)),
                                -np.deg2rad(10.0), np.deg2rad(10.0))
        else:
            theta_cmd = THETA_LAND

        return T, theta_cmd

    def _compute_pd_damping(self, state, theta_cmd, M):
        """
        PD主动阻尼 (缺陷3: 解决配平点弱阻尼).

        额外襟翼偏转 = Kp·(θ_cmd - θ) + Kd·(-q)
        前后翼同向偏转提供俯仰力矩.
        """
        theta = state[4]
        q = state[5]
        Kp, Kd = pd_gains(M)

        e_theta = theta_cmd - theta
        # 归一化到[-π,π]防wrap
        e_theta = (e_theta + np.pi) % (2 * np.pi) - np.pi

        delta_extra = Kp * e_theta - Kd * q
        # 前后翼同向 (都提供俯仰力矩)
        delta_extra_fwd = delta_extra
        delta_extra_aft = delta_extra

        return delta_extra_fwd, delta_extra_aft

    def update(self, state, dt):
        """
        控制器更新.

        state: [x, h, vx, vz, θ, q, m_fuel] (7维)
        dt: 时间步 (s)

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
                self._start_ramp(THETA_BELLY)  # flip内部会递减, 先设到belly
                info['phase_transition'] = 'BELLY→FLIP'

        elif self.phase == 'FLIP':
            # 翻转完成检查: α < 10° 且 h > 500m
            if abs(alpha) < ALPHA_LAND_TRIGGER and h > H_LAND_MIN:
                self.phase = 'LANDING'
                self.phase_t = 0.0
                self.landing_t = 0.0
                self.flip_completed = True
                self._start_ramp(THETA_LAND)
                info['phase_transition'] = 'FLIP→LANDING'

            # Kill: 翻转后 h < 800m
            if h < H_FLIP_KILL:
                return (0.0, THETA_LAND, 0.0, 0.0, self.phase, True,
                        {**info, 'kill_reason': f'flip_too_low (h={h:.0f}m<{H_FLIP_KILL}m)'})

        elif self.phase == 'LANDING':
            self.landing_t += dt

        # ============ 能量检查 (缺陷9, 仅FLIP后高空) ============
        # FLIP阶段全程检查; LANDING阶段仅h>200m检查 (低空committed to landing)
        if self.phase == 'FLIP' or (self.phase == 'LANDING' and h > 200.0):
            kill, reason = self._energy_check(state)
            if kill:
                return (0.0, THETA_LAND, 0.0, 0.0, self.phase, True,
                        {**info, 'kill_reason': reason})

        # ============ 各阶段控制律 ============
        if self.phase == 'BELLY':
            T, theta_cmd_target = self._belly_control(state)
        elif self.phase == 'FLIP':
            T, theta_cmd_target = self._flip_control(state)
        else:  # LANDING
            T, theta_cmd_target = self._landing_control(state)

        # ============ 2秒斜坡过渡 (缺陷7) ============
        # FLIP阶段内部已经是斜坡, 不需要额外过渡
        if self.phase == 'FLIP':
            self.theta_cmd_current = theta_cmd_target
            self.theta_cmd_target = theta_cmd_target
            self.ramp_active = False
        else:
            # BELLY/LANDING阶段: 如果目标变了, 启动斜坡
            if abs(theta_cmd_target - self.theta_cmd_target) > 1e-6 and not self.ramp_active:
                self._start_ramp(theta_cmd_target)
            self._ramp_update(dt)

        theta_cmd = self.theta_cmd_current

        # ============ PD主动阻尼 (缺陷3) ============
        d_extra_fwd, d_extra_aft = self._compute_pd_damping(state, theta_cmd, M)

        info['theta_cmd'] = theta_cmd
        info['T'] = T
        info['tgo'] = self._compute_tgo(state) if self.phase == 'BELLY' else 0.0

        return T, theta_cmd, d_extra_fwd, d_extra_aft, self.phase, False, info
