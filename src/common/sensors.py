"""
传感器模型: 陀螺/加计/GPS. 第一天内置噪声, 不允许无噪版本.
高斯白噪声 + 零偏漂移 (随机游走).
"""
import numpy as np


class IMU:
    """陀螺+加计. 输出带噪声和零偏的测量值."""

    def __init__(self, rng,
                 gyro_noise_std=np.radians(0.01),     # rad/s/sqrt(Hz) 简化
                 accel_noise_std=0.02,                # m/s^2/sqrt(Hz)
                 gyro_bias_walk=np.radians(1e-4),     # rad/s 零偏游走/步
                 accel_bias_walk=1e-4,                # m/s^2 零偏游走/步
                 gyro_init_bias=np.radians(0.5),      # 初始零偏量级
                 accel_init_bias=0.05):
        self.rng = rng
        self.gyro_noise_std = gyro_noise_std
        self.accel_noise_std = accel_noise_std
        self.gyro_bias_walk = gyro_bias_walk
        self.accel_bias_walk = accel_bias_walk
        # 初始零偏 (随机)
        self.gyro_bias = rng.normal(0.0, gyro_init_bias, size=3)
        self.accel_bias = rng.normal(0.0, accel_init_bias, size=3)

    def measure(self, omega_body_true, accel_body_true, dt):
        """omega_body_true: b系真实角速度. accel_body_true: b系真实比力(不含重力)."""
        # 零偏随机游走
        self.gyro_bias += self.rng.normal(0.0, self.gyro_bias_walk * np.sqrt(dt), size=3)
        self.accel_bias += self.rng.normal(0.0, self.accel_bias_walk * np.sqrt(dt), size=3)
        gyro_meas = omega_body_true + self.gyro_bias + self.rng.normal(0.0, self.gyro_noise_std, size=3)
        accel_meas = accel_body_true + self.accel_bias + self.rng.normal(0.0, self.accel_noise_std, size=3)
        return gyro_meas, accel_meas


class GPS:
    """GPS: 位置+速度测量, 低频(10Hz), 带噪声."""

    def __init__(self, rng,
                 pos_noise_std=0.5,    # m
                 vel_noise_std=0.1,    # m/s
                 update_rate=10.0):    # Hz
        self.rng = rng
        self.pos_noise_std = pos_noise_std
        self.vel_noise_std = vel_noise_std
        self.update_rate = update_rate
        self._acc = 0.0

    def measure(self, pos_true, vel_true, dt):
        """返回 (pos_meas, vel_meas, valid). valid=True 表示本步有更新."""
        self._acc += dt
        if self._acc >= 1.0 / self.update_rate:
            self._acc = 0.0
            pos_meas = pos_true + self.rng.normal(0.0, self.pos_noise_std, size=3)
            vel_meas = vel_true + self.rng.normal(0.0, self.vel_noise_std, size=3)
            return pos_meas, vel_meas, True
        return None, None, False


class RadarAltimeter:
    """雷达高度计: 高精度高度测量, 末端50m以下使用.
    高频(50Hz), 噪声σ=0.05m, 无累积漂移."""

    def __init__(self, rng,
                 alt_noise_std=0.05,   # m
                 update_rate=50.0,     # Hz
                 max_alt=100.0):       # m, 有效量程
        self.rng = rng
        self.alt_noise_std = alt_noise_std
        self.update_rate = update_rate
        self.max_alt = max_alt
        self._acc = 0.0

    def measure(self, pos_true, dt):
        """返回 (alt_meas, valid). alt_meas = -pos_n[2] (高度, 向上为正).
        valid=True表示本步有更新且在量程内."""
        self._acc += dt
        h_true = -pos_true[2]
        if self._acc >= 1.0 / self.update_rate and h_true < self.max_alt:
            self._acc = 0.0
            # 末端近距离时噪声略增(地表多径)
            noise_scale = 1.0 + max(0.0, (10.0 - h_true) / 10.0) * 0.5
            alt_meas = h_true + self.rng.normal(0.0, self.alt_noise_std * noise_scale)
            return alt_meas, True
        return None, False


class GPSBlackout:
    """GPS黑障: 星舰Mach>1.5且h>10km时等离子体鞘套导致GPS失锁.

    物理机制: 再入段气动加热产生等离子体鞘套, GPS信号被吸收/反射.
    持续时间: 30-60秒 (取决于速度和高度).

    工程处理: EKF跳过update_gps(), 切换纯惯导.
    """
    def __init__(self, mach_threshold=1.5, alt_threshold=10000.0):
        self.mach_threshold = mach_threshold
        self.alt_threshold = alt_threshold
        self.blackout_active = False
        self.blackout_start_time = 0.0
        self.blackout_duration = 0.0

    def check(self, mach, altitude, t):
        """检查GPS是否可用. 返回 (valid, reason)."""
        if mach > self.mach_threshold and altitude > self.alt_threshold:
            if not self.blackout_active:
                self.blackout_active = True
                self.blackout_start_time = t
            self.blackout_duration = t - self.blackout_start_time
            return False, f"GPS_BLACKOUT (Mach={mach:.1f}, h={altitude:.0f}m, duration={self.blackout_duration:.1f}s)"
        else:
            if self.blackout_active:
                # 刚出黑障, GPS需要重新捕获 (额外5s锁定时间)
                if t - (self.blackout_start_time + self.blackout_duration) < 5.0:
                    return False, "GPS_REACQUIRING"
                self.blackout_active = False
            return True, "GPS_OK"

    def reset(self):
        self.blackout_active = False
        self.blackout_start_time = 0.0
        self.blackout_duration = 0.0


class IMUConingCompensation:
    """IMU圆锥误差补偿: 三子样等效旋转矢量算法.

    物理机制: 火箭高频振动时, 陀螺积分产生不可逆的不可交换误差.
    不补偿: 再入10秒后姿态漂移到不可救药.

    算法: 三子样等效旋转矢量
      Phi = d_theta + (1/12) * d_theta1 x d_theta2
    其中 d_theta1, d_theta2 是前两个和当前增量.

    参考: Savage, "Strapdown Inertial Navigation Integration Algorithm"
    """
    def __init__(self):
        self.dtheta_prev1 = np.zeros(3)  # t-2时刻增量
        self.dtheta_prev2 = np.zeros(3)  # t-1时刻增量
        self.has_prev1 = False
        self.has_prev2 = False

    def compensate(self, gyro_meas, dt):
        """对陀螺测量进行圆锥补偿.

        参数:
          gyro_meas: 陀螺角速度测量 [rad/s], b系
          dt: 时间步长 [s]

        返回:
          dtheta_compensated: 补偿后的角度增量 [rad]
        """
        dtheta = gyro_meas * dt  # 当前增量

        if not self.has_prev2:
            # 不足三子样, 无补偿
            if self.has_prev1:
                self.dtheta_prev2 = self.dtheta_prev1.copy()
                self.has_prev2 = True
            self.dtheta_prev1 = dtheta.copy()
            self.has_prev1 = True
            return dtheta
        else:
            # 三子样等效旋转矢量
            # Phi = d_theta + (1/12) * d_theta1 x d_theta2
            # 使用前一个增量和当前增量
            coning_term = (1.0 / 12.0) * np.cross(self.dtheta_prev1, dtheta)
            dtheta_compensated = dtheta + coning_term

            # 滚动缓冲区
            self.dtheta_prev2 = self.dtheta_prev1.copy()
            self.dtheta_prev1 = dtheta.copy()

            return dtheta_compensated

    def reset(self):
        self.dtheta_prev1 = np.zeros(3)
        self.dtheta_prev2 = np.zeros(3)
        self.has_prev1 = False
        self.has_prev2 = False


class RadarMultipath:
    """雷达多径效应: 星舰腹部朝下且襟翼展开时, 雷达波多次反射.

    物理机制: h<100m时, 雷达波在襟翼和地面之间多次反射, 测高跳变.
    模型: h<100m时, 10%概率叠加±15m均匀分布随机跳变.

    工程处理: Level 1安全降级, EKF拒绝雷达更新, 切换纯惯导.
    """
    def __init__(self, rng, alt_threshold=100.0, jump_prob=0.10, jump_amp=15.0):
        self.rng = rng
        self.alt_threshold = alt_threshold
        self.jump_prob = jump_prob
        self.jump_amp = jump_amp
        self.last_jump = 0.0

    def measure(self, alt_true, dt):
        """返回 (alt_meas, is_jump). is_jump=True表示发生多径跳变."""
        if alt_true < self.alt_threshold:
            if self.rng.random() < self.jump_prob:
                # 多径跳变: ±15m均匀分布
                self.last_jump = self.rng.uniform(-self.jump_amp, self.jump_amp)
                return alt_true + self.last_jump, True
            else:
                self.last_jump = 0.0
                return alt_true, False
        else:
            self.last_jump = 0.0
            return alt_true, False

    def reset(self):
        self.last_jump = 0.0
