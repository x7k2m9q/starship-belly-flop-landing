"""
星舰非理想执行器模型 (Phase 8.0)
================================
理论方案 8.0: 星舰工程级物理与容错架构

三大"脏物理"执行器模型:
  1. FlapServoNonIdeal: 襟翼伺服电机死区(±0.5°) + Bouc-Wen摩擦滞环
  2. RaptorIgnitionTransient: 猛禽2代点火瞬态(50ms尖峰→100ms凹陷→稳态)
  3. TVCDelay: TVC喷管纯延迟(80ms)

工程判断:
  - 死区导致大攻角配平点(α=85°)附近高频极限环振荡
  - 点火瞬态推力偏差导致翻转角加速度失控
  - TVC延迟导致二阶跟踪动力学(ωn=2)姿态超调
"""
import numpy as np
from collections import deque


# ============================================================================
# 1. 襟翼伺服电机: 死区 + Bouc-Wen 滞环
# ============================================================================
class FlapServoNonIdeal:
    """襟翼伺服电机非理想模型: 死区 + Bouc-Wen 摩擦滞环.

    物理背景:
      星舰襟翼由电机+减速器驱动, 存在齿轮间隙与静摩擦.
      - 死区(±0.5°): 小指令无法克服静摩擦, 电机不响应.
        在大攻角配平点(α≈85°)附近, 配平襟翼角接近死区边界,
        控制器不断发小修正指令却无输出 → 累积误差 → 突然大修正,
        形成高频极限环振荡(limit cycle).
      - Bouc-Wen滞环: 描述摩擦力的记忆效应. 内部状态 z 跟踪
        滞回回线, 使正向与反向运动产生不同输出, 造成相位滞后.

    Bouc-Wen 模型 (Phase 9.0修正: 标准公式):
      dz/dt = A·(dx/dt) - β·|dx/dt|·|z|^(n-1)·z - γ·(dx/dt)·|z|^n
      其中 x 为输入位移(指令角), z 为滞环内部变量.
      A 控制滞环斜率, β/γ 控制回线形状, n 控制平滑度(n=1 为经典线性滞环).

      实际输出 = α·x + (1-α)·z  (标准Bouc-Wen: α=线性分量比)

    Phase 9.0修正 (缺陷29):
      原模型: output = x - z, gamma=-0.5 → beta+gamma=0 → z无界 → z追踪x → output≈0
      这导致执行器完全阻塞低频信号, 物理上不可能(真实襟翼能保持位置).
      修正: (1) gamma=+0.5 (标准符号, beta+gamma=1.0>0, z有界)
            (2) output = alpha*x + (1-alpha)*z (标准公式, alpha=0.3)
      物理意义: 30%线性响应(位置保持) + 70%滞环响应(摩擦记忆)
      z稳态幅度 = A/(beta+gamma) = 1.0/1.0 = 1.0 rad (有界)

    参数:
      dead_zone: 死区半宽 (rad), 默认 0.5°
      alpha: 线性刚度比 (0=纯滞环, 1=纯线性), 默认 0.3
      A, beta, gamma, n: Bouc-Wen 参数
    """

    # 襟翼最大偏转角 (±15°), 与 belly_flop DELTA_MAX 一致
    DELTA_MAX = np.deg2rad(15.0)

    def __init__(self, dead_zone=np.deg2rad(0.5), alpha=0.3,
                 A=1.0, beta=0.5, gamma=0.5, n=1):
        self.dead_zone = float(dead_zone)   # 死区半宽 (rad)
        self.alpha = float(alpha)           # 线性刚度比 (Phase 9.0: 位置保持)
        self.A = float(A)                   # Bouc-Wen 滞环斜率
        self.beta = float(beta)             # Bouc-Wen 回线参数 (正向)
        self.gamma = float(gamma)           # Bouc-Wen 回线参数 (Phase 9.0: 正值)
        self.n = int(n)                     # Bouc-Wen 平滑度指数
        # 内部状态
        self.z = 0.0          # Bouc-Wen 滞环内部变量
        self.x_prev = 0.0     # 上一拍指令 (用于数值微分)
        self.reset()

    def reset(self):
        # type: () -> None
        """重置内部状态 (z=0, x_prev=0)."""
        self.z = 0.0
        self.x_prev = 0.0

    def update(self, delta_cmd, dt):
        # type: (float, float) -> float
        """更新襟翼伺服输出.

        参数:
          delta_cmd: 指令偏转角 (rad)
          dt: 时间步长 (s)

        返回:
          delta_actual: 实际偏转角 (rad), 已限幅到 ±DELTA_MAX

        流程:
          1. 死区: |delta_cmd| < dead_zone → 强制为 0
          2. Bouc-Wen: 计算滞环变量 z 的导数并 Euler 积分
          3. 输出 = 指令 - z (滞环偏移)
          4. 限幅 ±15°
        """
        # --- 1. 死区: 小指令被静摩擦吞掉 ---
        if abs(delta_cmd) < self.dead_zone:
            delta_cmd = 0.0

        # --- 2. Bouc-Wen 滞环模型 ---
        # 输入速度 (数值微分)
        dx_dt = (delta_cmd - self.x_prev) / dt

        # Bouc-Wen 微分方程:
        #   dz/dt = A·(dx/dt) - β·|dx/dt|·|z|^(n-1)·z - γ·(dx/dt)·|z|^n
        abs_dx = abs(dx_dt)
        abs_z = abs(self.z)
        z_pow_n = abs_z ** self.n              # |z|^n
        z_pow_nm1_z = (abs_z ** (self.n - 1)) * self.z  # |z|^(n-1)·z

        dz_dt = (self.A * dx_dt
                 - self.beta * abs_dx * z_pow_nm1_z
                 - self.gamma * dx_dt * z_pow_n)

        # Euler 积分
        self.z += dz_dt * dt

        # --- 3. 实际输出 = α·指令 + (1-α)·滞环变量 (Phase 9.0标准公式) ---
        delta_actual = self.alpha * delta_cmd + (1.0 - self.alpha) * self.z

        # --- 4. 限幅 ±15° ---
        delta_actual = float(np.clip(delta_actual, -self.DELTA_MAX, self.DELTA_MAX))

        # --- 5. 更新状态 ---
        self.x_prev = delta_cmd

        return delta_actual


# ============================================================================
# 2. 猛禽2代点火瞬态
# ============================================================================
class RaptorIgnitionTransient:
    """猛禽2代(Raptor 2)发动机点火瞬态模型.

    物理背景:
      全流量分级燃烧循环(FFSC)发动机点火时, 燃烧室压力建立存在瞬态:
        - 前 50ms: 燃料先富集, 燃烧不稳定, 推力尖峰至 1.3×T_cmd
          (富燃爆震, 涡轮泵超速)
        - 接下来 100ms: 涡轮泵建立稳定工况, 推力凹陷至 0.7×T_cmd
          (混合比偏低, 燃烧效率不足)
        - 之后: 稳态推力 T_cmd

      翻转(Flip)机动时, 发动机从冷态点火, 点火瞬态的推力偏差
      (±30%)会产生显著角加速度扰动, 若不补偿会导致翻转角速度失控.

    参数:
      spike_factor: 点火尖峰倍率 (默认 1.3)
      droop_factor: 凹陷倍率 (默认 0.7)
      spike_duration: 尖峰持续时间 (s, 默认 0.05)
      droop_duration: 凹陷持续时间 (s, 默认 0.10)
    """

    def __init__(self, spike_factor=1.3, droop_factor=0.7,
                 spike_duration=0.05, droop_duration=0.10):
        # type: (float, float, float, float) -> None
        self.spike_factor = float(spike_factor)       # 尖峰倍率
        self.droop_factor = float(droop_factor)       # 凹陷倍率
        self.spike_duration = float(spike_duration)   # 尖峰时长 (s)
        self.droop_duration = float(droop_duration)   # 凹陷时长 (s)
        # 内部状态
        self.is_ignited = False       # 是否已点火
        self.ignition_time = 0.0      # 点火时刻 (s)
        self.reset()

    def reset(self):
        # type: () -> None
        """重置点火状态."""
        self.is_ignited = False
        self.ignition_time = 0.0

    def update(self, T_cmd, dt, t):
        # type: (float, float, float) -> float
        """更新推力输出 (含点火瞬态).

        参数:
          T_cmd: 指令推力 (N), >0 表示要求点火
          dt: 时间步长 (s, 此处保留接口一致性)
          t: 当前仿真时间 (s)

        返回:
          T_actual: 实际推力 (N)

        瞬态曲线:
          t∈[0, 50ms):      T = T_cmd × spike_factor  (尖峰)
          t∈[50ms, 150ms):  T = T_cmd × droop_factor  (凹陷)
          t≥150ms:          T = T_cmd                  (稳态)
        """
        # 关机: 指令为 0 → 熄火
        if T_cmd == 0.0:
            self.is_ignited = False
            return 0.0

        # 首次收到推力指令 → 点火
        if (not self.is_ignited) and T_cmd > 0.0:
            self.is_ignited = True
            self.ignition_time = t

        # 已点火: 按瞬态曲线输出
        if self.is_ignited:
            dt_since_ignition = t - self.ignition_time
            if dt_since_ignition < self.spike_duration:
                # 尖峰段: 富燃爆震
                return T_cmd * self.spike_factor
            elif dt_since_ignition < (self.spike_duration + self.droop_duration):
                # 凹陷段: 涡轮泵建压中
                return T_cmd * self.droop_factor
            else:
                # 稳态段
                return T_cmd

        # 不应到达此处 (T_cmd>0 已触发点火)
        return T_cmd


# ============================================================================
# 3. TVC 喷管纯延迟
# ============================================================================
class TVCDelay:
    """TVC(推力矢量控制)喷管纯延迟模型.

    物理背景:
      TVC 喷管偏转指令经飞控计算机→伺服驱动器→液压作动筒→喷管,
      整条链路存在纯延迟(transport delay). 与一阶滞后不同, 纯延迟
      不会"平滑"信号, 而是完整保留信号形状仅时间平移.

      80ms 延迟对二阶姿态跟踪动力学(ωn=2 rad/s)影响显著:
        相位裕度损失 ≈ ωn × delay × (180/π) ≈ 2×0.08×57.3 ≈ 9.2°
      若不补偿, 会导致姿态超调甚至失稳.

    实现:
      使用 collections.deque 环形缓冲区存储 (timestamp, gimbal_cmd) 对.
      每次更新时丢弃过期样本, 返回最旧的有效样本 (即延迟后的指令).

    参数:
      delay: 纯延迟时间 (s, 默认 0.08)
      dt_max: 最大时间步长 (s, 默认 0.02), 用于确定缓冲区容量
    """

    def __init__(self, delay=0.08, dt_max=0.02):
        # type: (float, float) -> None
        self.delay = float(delay)
        self.dt_max = float(dt_max)
        # 缓冲区容量: 覆盖 delay 时长 + 1 个冗余样本
        self.buffer_size = int(self.delay / self.dt_max) + 1
        # 环形缓冲区: 存储 (t, gimbal_cmd)
        self.buffer = deque(maxlen=self.buffer_size)
        self.reset()

    def reset(self):
        # type: () -> None
        """清空延迟缓冲区."""
        self.buffer.clear()

    def update(self, gimbal_cmd, t):
        # type: (float, float) -> float
        """更新 TVC 延迟输出.

        参数:
          gimbal_cmd: 当前 TVC 偏转指令 (rad)
          t: 当前仿真时间 (s)

        返回:
          gimbal_actual: 延迟后的 TVC 偏转 (rad)

        算法:
          1. 将 (t, gimbal_cmd) 追加到缓冲区
          2. 丢弃早于 (t - delay) 的过期样本
          3. 返回缓冲区中最旧的样本 (即 delay 秒前的指令)
          4. 启动阶段缓冲区为空 → 返回当前指令 (无延迟)
        """
        # 追加当前样本
        self.buffer.append((t, gimbal_cmd))

        # 丢弃过期样本 (早于 t - delay)
        cutoff = t - self.delay
        while len(self.buffer) > 1 and self.buffer[0][0] < cutoff:
            self.buffer.popleft()

        # 返回最旧的有效样本 (延迟后的指令)
        if len(self.buffer) > 0:
            return self.buffer[0][1]

        # 启动阶段缓冲区为空: 返回当前指令 (无延迟)
        return gimbal_cmd


# ============================================================================
# 4. 非理想执行器套件 (组合)
# ============================================================================
class NonIdealActuatorSuite:
    """非理想执行器套件: 组合襟翼伺服×2 + 猛禽点火 + TVC延迟.

    用于 Phase 8.0 蒙特卡洛仿真, 支持参数随机化以覆盖不确定性包络.

    组成:
      - fwd 襟翼伺服 (FlapServoNonIdeal): 前襟翼
      - aft 襟翼伺服 (FlapServoNonIdeal): 后襟翼
      - 猛禽点火瞬态 (RaptorIgnitionTransient): 主发动机推力
      - TVC 延迟 (TVCDelay): 推力矢量偏转

    参数:
      rng: numpy 随机数生成器 (np.random.Generator), 用于参数随机化
    """

    def __init__(self, rng=None):
        # type: (object) -> None
        # 前后襟翼各一套独立伺服 (独立死区/滞环)
        self.fwd_flap = FlapServoNonIdeal()
        self.aft_flap = FlapServoNonIdeal()
        # 猛禽点火瞬态
        self.raptor = RaptorIgnitionTransient()
        # TVC 延迟
        self.tvc = TVCDelay()
        # 随机数生成器
        self.rng = rng
        if rng is not None:
            self.randomize(rng)

    def randomize(self, rng):
        # type: (object) -> None
        """随机化各执行器参数 (蒙特卡洛用).

        随机化范围:
          - 死区: 0.3° ~ 0.8° (制造公差 + 磨损)
          - Bouc-Wen 参数: 标称值 ±20% (摩擦特性散布)
          - 点火尖峰: 1.2 ~ 1.4 (燃烧不稳定性)
          - TVC 延迟: 60 ~ 100 ms (伺服链路散布)

        参数:
          rng: numpy 随机数生成器
        """
        # --- 襟翼死区: 0.3° ~ 0.8° ---
        dz_fwd = np.deg2rad(rng.uniform(0.3, 0.8))
        dz_aft = np.deg2rad(rng.uniform(0.3, 0.8))

        # --- Bouc-Wen 参数: 标称 ±20% ---
        # Phase 9.0修正: gamma改为正值 (标准Bouc-Wen), alpha=0.3 (位置保持)
        # A=1.0, beta=0.5, gamma=0.5, n=1, alpha=0.3
        A_fwd = rng.uniform(1.0 * 0.8, 1.0 * 1.2)
        A_aft = rng.uniform(1.0 * 0.8, 1.0 * 1.2)
        beta_fwd = rng.uniform(0.5 * 0.8, 0.5 * 1.2)
        beta_aft = rng.uniform(0.5 * 0.8, 0.5 * 1.2)
        gamma_fwd = rng.uniform(0.5 * 0.8, 0.5 * 1.2)   # Phase 9.0: 正值
        gamma_aft = rng.uniform(0.5 * 0.8, 0.5 * 1.2)   # Phase 9.0: 正值
        alpha_fwd = rng.uniform(0.3 * 0.8, 0.3 * 1.2)   # 线性刚度比散布
        alpha_aft = rng.uniform(0.3 * 0.8, 0.3 * 1.2)

        # 重建前后襟翼伺服 (保留随机化参数)
        self.fwd_flap = FlapServoNonIdeal(
            dead_zone=dz_fwd, alpha=alpha_fwd,
            A=A_fwd, beta=beta_fwd, gamma=gamma_fwd, n=1)
        self.aft_flap = FlapServoNonIdeal(
            dead_zone=dz_aft, alpha=alpha_aft,
            A=A_aft, beta=beta_aft, gamma=gamma_aft, n=1)

        # --- 点火尖峰: 1.2 ~ 1.4 ---
        spike = rng.uniform(1.2, 1.4)
        # 凹陷相应调整 (尖峰越高, 凹陷越深, 能量守恒近似)
        droop = rng.uniform(0.6, 0.8)
        self.raptor = RaptorIgnitionTransient(
            spike_factor=spike, droop_factor=droop,
            spike_duration=0.05, droop_duration=0.10)

        # --- TVC 延迟: 60 ~ 100 ms ---
        delay = rng.uniform(0.06, 0.10)
        self.tvc = TVCDelay(delay=delay, dt_max=0.02)

    def update_flaps(self, delta_fwd_cmd, delta_aft_cmd, dt):
        # type: (float, float, float) -> tuple
        """更新前后襟翼伺服.

        参数:
          delta_fwd_cmd: 前襟翼指令 (rad)
          delta_aft_cmd: 后襟翼指令 (rad)
          dt: 时间步长 (s)

        返回:
          (delta_fwd_actual, delta_aft_actual): 实际偏转角 (rad)
        """
        delta_fwd_actual = self.fwd_flap.update(delta_fwd_cmd, dt)
        delta_aft_actual = self.aft_flap.update(delta_aft_cmd, dt)
        return (delta_fwd_actual, delta_aft_actual)

    def update_thrust(self, T_cmd, dt, t):
        # type: (float, float, float) -> float
        """更新猛禽发动机推力 (含点火瞬态).

        参数:
          T_cmd: 指令推力 (N)
          dt: 时间步长 (s)
          t: 当前仿真时间 (s)

        返回:
          T_actual: 实际推力 (N)
        """
        return self.raptor.update(T_cmd, dt, t)

    def update_tvc(self, gimbal_cmd, t):
        # type: (float, float) -> float
        """更新 TVC 偏转 (含纯延迟).

        参数:
          gimbal_cmd: TVC 偏转指令 (rad)
          t: 当前仿真时间 (s)

        返回:
          gimbal_actual: 延迟后的 TVC 偏转 (rad)
        """
        return self.tvc.update(gimbal_cmd, t)

    def reset(self):
        # type: () -> None
        """重置所有子模型状态."""
        self.fwd_flap.reset()
        self.aft_flap.reset()
        self.raptor.reset()
        self.tvc.reset()
