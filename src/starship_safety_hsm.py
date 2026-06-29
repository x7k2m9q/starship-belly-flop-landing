"""
星舰分级故障状态机 (Phase 8.0)
==============================
理论方案 8.0: 星舰工程级物理与容错架构

三级降级状态机:
  Level 1 (Soft):  雷达多径跳变 → EKF拒绝雷达更新, 切换纯惯导
  Level 2 (Hard):  襟翼卡死检测 → 锁定对侧襟翼, 切推力矢量主导
  Level 3 (Abort): 制导发散(tilt>20°) → 关闭猛禽, 弹道落点预测

设计原则:
  1. 故障检测基于物理量, 非"上帝视角" (不依赖知道真实故障)
  2. 每级降级只降级必要功能, 保留最大控制权限
  3. Level 3必须保证"安全坠毁"而非"横飞乱撞"
  4. 状态机单向升级: NOMINAL→L1→L2→L3, 不可降级回退
     (工程判断: 故障不会自愈, 降级后按降级模式飞到底)

工程判断:
  - 雷达多径是瞬态的, 但EKF一旦被污染需要时间恢复 → 拒绝更新比修复更安全
  - 襟翼卡死是永久的, 对侧襟翼锁定后气动对称, TVC补偿力矩
  - tilt>20°时气动已不可控, 唯一选择是关机+落点预测+海上迫降
"""
import numpy as np
from enum import IntEnum


# ===========================================================================
# 安全等级定义
# ===========================================================================
class SafetyLevel(IntEnum):
    """星舰安全状态等级 (单向升级)."""
    NOMINAL       = 0  # 全部正常
    SOFT_DEGRADED = 1  # Level 1: 雷达多径, 纯惯导
    HARD_DEGRADED = 2  # Level 2: 襟翼卡死, TVC主导
    ABORT         = 3  # Level 3: 制导发散, 关机+落点预测


# ===========================================================================
# 落点预测 (弹道积分)
# ===========================================================================
def predict_landing_point(state, dt=0.5, t_max=120.0):
    """
    Level 3弹道落点预测: 关机后纯弹道积分.

    假设: 发动机关闭, 纯弹道飞行 (重力+气动阻力).
    用途: 判定残骸落点是否在安全海域.

    工程 intuition:
      - 落点预测不需高精度, dt=0.5s 足够 (240步 vs 1200步, 5x加速).
      - 简化大气用指数衰减, 避免 atmosphere() 查表开销.
      - 符号约定: h=高度(正=高), vz=下降率(正=下降), dh/dt=-vz.

    参数:
      state: [x, h, vx, vz, theta, q, m_fuel] (7维)
      dt: 积分步长 (s, 默认0.5, 落点预测不需高精度)
      t_max: 最大积分时间 (s)

    返回:
      dict: {
        'landing_x': 落点水平位置 [m],
        'landing_t': 飞行时间 [s],
        'max_h': 最大高度 [m],
        'in_safe_zone': 是否在安全海域,
        'trajectory': 轨迹 (N, 2) [x, h]
      }
    """
    SAFE_X_MIN = 500.0   # m, 安全海域最小水平距离
    S_REF = 27.0         # m^2 (星舰参考面积, 简化)
    CD_BALLISTIC = 0.8   # 弹道段阻力系数 (腹部朝下)
    G_CONST = 9.80665    # m/s^2
    RHO0 = 1.225         # kg/m^3
    H_SCALE = 8500.0     # m, 大气标高

    x, h, vx, vz, theta, q, m_fuel = state
    m_total = 100000.0 + max(m_fuel, 0.0)  # 干重100t + 剩余燃料

    trajectory = []
    t = 0.0
    max_h = h

    # Euler 积分 (落点预测不需 RK4 精度)
    while t < t_max and h > 0:
        V = np.sqrt(vx * vx + vz * vz)
        if V < 1e-6:
            break

        # 简化大气 (指数衰减, 避免 atmosphere() 查表)
        rho = RHO0 * np.exp(-h / H_SCALE)

        # 阻力 (反对速度方向)
        Q = 0.5 * rho * V * V
        F_drag = Q * CD_BALLISTIC * S_REF
        inv_mV = 1.0 / (V * m_total)

        # dvx/dt = -F_drag * vx / (V*m), dvz/dt = g - F_drag*vz/(V*m)
        ax = -F_drag * vx * inv_mV
        az = G_CONST - F_drag * vz * inv_mV

        # Euler 积分: dh/dt = -vz (vz正=下降, h正=高度)
        vx += ax * dt
        vz += az * dt
        x  += vx * dt
        h  -= vz * dt

        if h > max_h:
            max_h = h

        trajectory.append((x, h))
        t += dt

    landing_x = x
    in_safe = abs(landing_x) > SAFE_X_MIN

    return {
        'landing_x': landing_x,
        'landing_t': t,
        'max_h': max_h,
        'in_safe_zone': in_safe,
        'trajectory': np.array(trajectory) if trajectory else np.zeros((0, 2)),
    }


# ===========================================================================
# 星舰分级故障状态机
# ===========================================================================
class StarshipSafetyHSM:
    """
    星舰分级故障状态机 (Hierarchical State Machine).

    三级降级:
      Level 1 (Soft):  雷达多径跳变 → 拒绝雷达更新, 纯惯导
      Level 2 (Hard):  襟翼卡死 → 锁定对侧襟翼, TVC主导
      Level 3 (Abort): 制导发散 → 关机, 落点预测, 海上迫降

    状态机单向升级: NOMINAL→L1→L2→L3, 不可回退.
    """

    # Level 1: 雷达多径
    RADAR_REJECT_DURATION = 2.0  # 拒绝雷达更新2秒

    # Level 2: 襟翼卡死检测
    FLAP_STUCK_ERR_THRESH = np.deg2rad(5.0)  # 指令与反馈误差阈值5°
    FLAP_STUCK_DURATION = 0.5  # 连续超限0.5秒判定卡死

    # Level 3: 制导发散
    ABORT_TILT_THRESH = np.deg2rad(20.0)  # tilt>20°触发Abort

    def __init__(self):
        self.level = SafetyLevel.NOMINAL

        # Level 1 状态
        self.radar_reject_timer = 0.0
        self.radar_rejected = False

        # Level 2 状态
        self.flap_stuck_timer = 0.0
        self.flap_stuck_side = None  # 'fwd' or 'aft' or None
        self.flap_locked = False  # 对侧襟翼是否已锁定
        self.tvc_dominant = False  # TVC是否主导控制

        # Level 3 状态
        self.abort_triggered = False
        self.abort_reason = ""
        self.landing_prediction = None
        self._last_predict_t = -999.0  # 上次落点预测时间 (节流用)
        self._predict_interval = 1.0   # 落点预测更新间隔 (秒)

        # 统计
        self.l1_trigger_count = 0
        self.l2_trigger_count = 0
        self.l3_trigger_count = 0

        # 襟翼反馈历史 (用于卡死检测)
        self._flap_fwd_cmd_prev = 0.0
        self._flap_aft_cmd_prev = 0.0

    def update(self, state, flap_fwd_cmd, flap_aft_cmd,
               flap_fwd_actual, flap_aft_actual,
               radar_jump_detected, t, dt, phase='BELLY'):
        """
        状态机更新 (每步调用).

        参数:
          state: [x, h, vx, vz, theta, q, m_fuel] (7维)
          flap_fwd_cmd: 前翼指令 [rad]
          flap_aft_cmd: 后翼指令 [rad]
          flap_fwd_actual: 前翼实际偏转 [rad] (来自非理想模型反馈)
          flap_aft_actual: 后翼实际偏转 [rad]
          radar_jump_detected: 雷达多径跳变标志 (来自RadarMultipath)
          t: 当前时间 [s]
          dt: 时间步长 [s]
          phase: 当前飞行阶段 ('BELLY'/'FLIP'/'LANDING')

        返回:
          dict: {
            'level': SafetyLevel,
            'radar_rejected': bool,  # L1: 是否拒绝雷达更新
            'flap_locked_side': str or None,  # L2: 锁定哪侧襟翼
            'tvc_dominant': bool,  # L2: TVC是否主导
            'abort': bool,  # L3: 是否Abort
            'abort_reason': str,
            'landing_prediction': dict or None,
            'kill': bool,  # 是否Kill (L3且落点不安全)
          }
        """
        result = {
            'level': self.level,
            'radar_rejected': False,
            'flap_locked_side': None,
            'tvc_dominant': False,
            'abort': False,
            'abort_reason': '',
            'landing_prediction': None,
            'kill': False,
        }

        # ===============================================================
        # Level 1: 雷达多径检测
        # ===============================================================
        if radar_jump_detected and self.level < SafetyLevel.ABORT:
            self.radar_reject_timer = self.RADAR_REJECT_DURATION
            self.radar_rejected = True
            if self.level < SafetyLevel.SOFT_DEGRADED:
                self.level = SafetyLevel.SOFT_DEGRADED
                self.l1_trigger_count += 1

        # 计时器递减
        if self.radar_reject_timer > 0:
            self.radar_reject_timer -= dt
            self.radar_rejected = True
            result['radar_rejected'] = True
        else:
            self.radar_rejected = False

        # ===============================================================
        # Level 2: 襟翼卡死检测
        # ===============================================================
        if self.level < SafetyLevel.HARD_DEGRADED:
            # 检测前翼卡死
            fwd_err = abs(flap_fwd_cmd - flap_fwd_actual)
            aft_err = abs(flap_aft_cmd - flap_aft_actual)

            fwd_stuck = fwd_err > self.FLAP_STUCK_ERR_THRESH
            aft_stuck = aft_err > self.FLAP_STUCK_ERR_THRESH

            if fwd_stuck or aft_stuck:
                self.flap_stuck_timer += dt
                if fwd_stuck:
                    self.flap_stuck_side = 'fwd'
                else:
                    self.flap_stuck_side = 'aft'
            else:
                self.flap_stuck_timer = 0.0
                self.flap_stuck_side = None

            # 连续超限0.5秒 → 判定卡死
            if self.flap_stuck_timer >= self.FLAP_STUCK_DURATION:
                self.level = SafetyLevel.HARD_DEGRADED
                self.flap_locked = True
                self.tvc_dominant = True
                self.l2_trigger_count += 1

        # Level 2激活后的输出
        if self.level >= SafetyLevel.HARD_DEGRADED:
            result['flap_locked_side'] = self.flap_stuck_side
            result['tvc_dominant'] = True

        # ===============================================================
        # Level 3: 制导发散检测
        # ===============================================================
        theta = state[4]  # 姿态角

        # 工程判断: BELLY阶段theta=85°是正常的, 不算发散.
        # tilt定义: 翻转开始后(|theta|<45°), |theta|即为tilt.
        #
        # Phase 9.0修正 (缺陷30): FLIP阶段是intentional maneuver,
        #   theta从85°→0°必然经过20°-45°区间, 不应触发abort.
        #   仅在LANDING阶段检查tilt (此时theta应接近0°, tilt>20°才是真正发散).
        if phase == 'FLIP':
            tilt = 0.0  # FLIP是intentional maneuver, 不检查tilt
        elif abs(theta) < np.deg2rad(45.0):
            tilt = abs(theta)
        else:
            tilt = 0.0  # BELLY阶段不触发

        if tilt > self.ABORT_TILT_THRESH and self.level < SafetyLevel.ABORT:
            self.level = SafetyLevel.ABORT
            self.abort_triggered = True
            self.abort_reason = f"GUIDANCE_DIVERGENCE (tilt={np.rad2deg(tilt):.1f}deg > 20deg)"
            self.l3_trigger_count += 1

            # 落点预测 (首次触发, 立即计算)
            self.landing_prediction = predict_landing_point(state)
            self._last_predict_t = t

            # 如果落点不在安全海域 → Kill
            if not self.landing_prediction['in_safe_zone']:
                result['kill'] = True

        # Level 3激活后的输出
        if self.level >= SafetyLevel.ABORT:
            result['abort'] = True
            result['abort_reason'] = self.abort_reason
            result['landing_prediction'] = self.landing_prediction

            # 落点预测节流更新: 每 _predict_interval 秒更新一次 (非每步)
            # 工程判断: 10ms内落点不会显著变化, 每秒更新足够.
            # 避免每步调用 predict_landing_point (1200步积分) 导致仿真卡死.
            if (t - self._last_predict_t) >= self._predict_interval:
                self.landing_prediction = predict_landing_point(state)
                self._last_predict_t = t
                result['landing_prediction'] = self.landing_prediction
                if not self.landing_prediction['in_safe_zone']:
                    result['kill'] = True

        result['level'] = self.level
        return result

    def reset(self):
        """重置状态机 (新仿真开始时调用)."""
        self.level = SafetyLevel.NOMINAL
        self.radar_reject_timer = 0.0
        self.radar_rejected = False
        self.flap_stuck_timer = 0.0
        self.flap_stuck_side = None
        self.flap_locked = False
        self.tvc_dominant = False
        self.abort_triggered = False
        self.abort_reason = ""
        self.landing_prediction = None
        self._last_predict_t = -999.0
        self.l1_trigger_count = 0
        self.l2_trigger_count = 0
        self.l3_trigger_count = 0
        self._flap_fwd_cmd_prev = 0.0
        self._flap_aft_cmd_prev = 0.0

    def get_stats(self):
        """返回统计信息."""
        return {
            'final_level': int(self.level),
            'l1_triggers': self.l1_trigger_count,
            'l2_triggers': self.l2_trigger_count,
            'l3_triggers': self.l3_trigger_count,
            'abort_reason': self.abort_reason,
            'landing_safe': (self.landing_prediction['in_safe_zone']
                             if self.landing_prediction else None),
            'landing_x': (self.landing_prediction['landing_x']
                          if self.landing_prediction else None),
        }
