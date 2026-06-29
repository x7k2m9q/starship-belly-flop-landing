"""
星舰 6-DOF 非理想执行器模型 (Phase 7.0 战役二 — 问题20-22)
=============================================================
理论方案7.0:
  - 问题20: 襟翼速率限制 (30°/s)
  - 问题21: Bouc-Wen模型 (Phase 9.0已修复, 标准公式 y=α·x+(1-α)·z)
  - 问题22: 死区补偿 (偏置+Dither, Dither幅值>死区最大值)

4片独立非理想执行器: FL/FR/RL/RR
  每片独立参数: 死区/Bouc-Wen/速率限制
  支持蒙特卡洛随机化

死区补偿策略 (7.0.txt第328行铁律):
  "死区补偿（偏置+Dither）必须加在控制分配之后、实际指令输出前"
  Phase 8.0 "缺陷27"证明: 配平点死区不补偿会导致姿态完全发散

补偿方法:
  1. 偏置: 对小指令(|δ|<dead_zone)施加偏置, 使其越过死区
  2. Dither: 高频抖动信号, 幅值>死区最大值, 频率高于控制带宽
     - Dither频率: 20Hz (高于控制带宽0.5Hz, 低于结构模态2.35Hz)
     - Dither幅值: 1.0° (>死区最大值0.8°)
     - Dither使执行器在小指令区持续微振, 线性化死区特性
"""
import numpy as np
from src.starship_non_ideal import FlapServoNonIdeal


# ============================================================================
# 4片独立非理想襟翼执行器
# ============================================================================
class FlapActuatorSuite6DOF:
    """4片独立非理想襟翼执行器 (FL/FR/RL/RR).

    每片独立:
      - 死区 (0.3°~0.8°, 制造公差)
      - Bouc-Wen滞环 (α=0.3, A=1.0, β=0.5, γ=0.5)
      - 速率限制 (30°/s)

    死区补偿:
      - 偏置+Dither, 加在控制分配之后、实际指令输出前
      - Dither: 20Hz正弦, 幅值1.0° (>死区最大值0.8°)

    参数:
      dead_zones: 4片死区半宽 (rad), 默认全0.5°
      rate_limits: 4片速率限制 (rad/s), 默认全30°/s
      use_compensation: 是否启用死区补偿, 默认True
      dither_freq: Dither频率 (Hz), 默认20
      dither_amp: Dither幅值 (rad), 默认1.0°
    """

    def __init__(self, dead_zones=None, rate_limits=None,
                 use_compensation=True, dither_freq=20.0,
                 dither_amp=np.deg2rad(1.0)):
        # 4片独立死区 (默认0.5°)
        if dead_zones is None:
            dead_zones = [np.deg2rad(0.5)] * 4
        self.dead_zones = list(dead_zones)

        # 4片独立速率限制 (默认30°/s)
        if rate_limits is None:
            rate_limits = [np.deg2rad(30.0)] * 4
        self.rate_limits = list(rate_limits)

        # 死区补偿参数
        self.use_compensation = use_compensation
        self.dither_freq = float(dither_freq)
        self.dither_amp = float(dither_amp)

        # 4片独立Bouc-Wen伺服 (死区=0, 死区在外部补偿处理)
        # Phase 9.0修复: 标准公式 y=α·x+(1-α)·z, gamma=+0.5
        self.servos = [
            FlapServoNonIdeal(dead_zone=0.0, alpha=0.3,
                              A=1.0, beta=0.5, gamma=0.5, n=1)
            for i in range(4)
        ]

        # 4片速率限制跟踪目标 (斜坡跟踪)
        self.delta_target = np.zeros(4)

        # Dither相位 (每片独立, 避免同步)
        self.dither_phases = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])

        # 当前时间
        self.t = 0.0

    def reset(self):
        """重置所有执行器状态."""
        for servo in self.servos:
            servo.reset()
        self.delta_target = np.zeros(4)
        self.t = 0.0

    def update(self, delta_cmds, dt):
        """更新4片襟翼执行器.

        参数:
          delta_cmds: 4片指令偏转 [d_FL, d_FR, d_RL, d_RR] (rad)
          dt: 时间步长 (s)

        返回:
          delta_actuals: 4片实际偏转 (rad), 已限幅±DELTA_MAX

        流程 (缺陷37修正: 速率限制→死区补偿→Bouc-Wen):
          1. 速率限制: 斜坡跟踪目标, |δ_target[k+1]-δ_target[k]| ≤ rate·dt
          2. 死区补偿: 偏置+Dither (确保信号越过死区)
          3. Bouc-Wen滞环: 标准公式 y=α·x+(1-α)·z (无内部死区)
          4. 限幅 ±15°

        缺陷37: 速率限制在死区补偿之前, 否则大指令被限制到<死区被吞掉
          原顺序: 补偿→速率限制→Bouc-Wen(含死区) → 5°指令被限到0.3°<死区0.5°→全0
          修正顺序: 速率限制→补偿→Bouc-Wen(无死区) → 5°指令斜坡跟踪, 补偿确保越过死区
        """
        self.t += dt
        delta_cmds = np.asarray(delta_cmds, dtype=float)
        delta_actuals = np.zeros(4)

        for i in range(4):
            cmd = delta_cmds[i]

            # ---- 1. 速率限制 (斜坡跟踪) ----
            # delta_target以最大速率逼近cmd
            max_delta = self.rate_limits[i] * dt
            delta_error = cmd - self.delta_target[i]
            delta_step = np.clip(delta_error, -max_delta, max_delta)
            self.delta_target[i] += delta_step
            delta_rated = self.delta_target[i]

            # ---- 2. 死区补偿 (偏置+Dither) ----
            if self.use_compensation:
                # Dither: 高频正弦, 幅值>死区最大值
                dither = self.dither_amp * np.sin(
                    2 * np.pi * self.dither_freq * self.t + self.dither_phases[i]
                )
                # 偏置: 对小指令施加偏置, 使其越过死区
                if abs(delta_rated) < self.dead_zones[i] * 2:
                    bias = np.sign(delta_rated) * self.dead_zones[i] if abs(delta_rated) > 1e-10 else 0.0
                    cmd_compensated = delta_rated + bias + dither
                else:
                    cmd_compensated = delta_rated + dither
            else:
                # 无补偿: 直接施加死区
                if abs(delta_rated) < self.dead_zones[i]:
                    cmd_compensated = 0.0
                else:
                    cmd_compensated = delta_rated

            # ---- 3. Bouc-Wen滞环 (无内部死区, dead_zone=0) ----
            delta_actual = self.servos[i].update(cmd_compensated, dt)

            delta_actuals[i] = delta_actual

        return delta_actuals

    def randomize(self, rng):
        """随机化4片执行器参数 (蒙特卡洛用).

        随机化范围:
          - 死区: 0.3°~0.8° (制造公差+磨损)
          - Bouc-Wen参数: 标称±20%
          - 速率限制: 25°/s~35°/s
        """
        for i in range(4):
            dz = np.deg2rad(rng.uniform(0.3, 0.8))
            A = rng.uniform(0.8, 1.2)
            beta = rng.uniform(0.4, 0.6)
            gamma = rng.uniform(0.4, 0.6)
            alpha = rng.uniform(0.24, 0.36)
            rate = np.deg2rad(rng.uniform(25.0, 35.0))

            self.dead_zones[i] = dz
            self.rate_limits[i] = rate
            # Bouc-Wen伺服死区=0, 死区在外部补偿处理
            self.servos[i] = FlapServoNonIdeal(
                dead_zone=0.0, alpha=alpha,
                A=A, beta=beta, gamma=gamma, n=1
            )

        # Dither幅值必须>死区最大值
        max_dz = max(self.dead_zones)
        if self.dither_amp <= max_dz:
            self.dither_amp = max_dz * 1.2  # 确保Dither>死区


# ============================================================================
# TVC非理想模型 (延迟+速率限制)
# ============================================================================
class TVCActuator6DOF:
    """TVC推力矢量控制非理想模型.

    包含:
      - 纯延迟 (80ms, 复用TVCDelay)
      - 速率限制 (20°/s)
      - 限幅 (±10°)

    参数:
      delay: 纯延迟 (s), 默认0.08
      rate_limit: 速率限制 (rad/s), 默认20°/s
      gimbal_limit: 最大偏转 (rad), 默认10°
    """

    def __init__(self, delay=0.08, rate_limit=np.deg2rad(20.0),
                 gimbal_limit=np.deg2rad(10.0), dt=0.01):
        from src.starship_non_ideal import TVCDelay
        # dt_max与仿真步长一致, 确保缓冲区容量足够覆盖delay
        self.tvc_delay = TVCDelay(delay=delay, dt_max=dt)
        self.rate_limit = float(rate_limit)
        self.gimbal_limit = float(gimbal_limit)
        self.gimbal_prev = np.zeros(2)  # [gimbal_y, gimbal_z]
        self.t = 0.0

    def reset(self):
        self.tvc_delay.reset()
        self.gimbal_prev = np.zeros(2)
        self.t = 0.0

    def update(self, gimbal_cmds, dt):
        """更新TVC偏转.

        参数:
          gimbal_cmds: [gimbal_y, gimbal_z] 指令 (rad)
          dt: 时间步长 (s)

        返回:
          gimbal_actuals: [gimbal_y, gimbal_z] 实际偏转 (rad)

        缺陷38: TVCDelay缓冲区未填满时返回当前值而非延迟值.
          修复: 启动阶段(t<delay)返回0, 模拟纯延迟的物理特性.
        """
        self.t += dt
        gimbal_cmds = np.asarray(gimbal_cmds, dtype=float)
        gimbal_actuals = np.zeros(2)

        # 启动阶段: t < delay 时返回0 (纯延迟的物理特性)
        if self.t < self.tvc_delay.delay:
            return gimbal_actuals

        for i in range(2):
            # 1. 纯延迟
            delayed = self.tvc_delay.update(gimbal_cmds[i], self.t)

            # 2. 速率限制
            max_delta = self.rate_limit * dt
            limited = np.clip(
                delayed,
                self.gimbal_prev[i] - max_delta,
                self.gimbal_prev[i] + max_delta
            )

            # 3. 限幅
            gimbal_actuals[i] = np.clip(limited, -self.gimbal_limit, self.gimbal_limit)
            self.gimbal_prev[i] = gimbal_actuals[i]

        return gimbal_actuals
