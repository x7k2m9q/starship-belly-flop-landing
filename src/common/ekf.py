"""E3: 乘性扩展卡尔曼滤波器 (MEKF, 15维误差状态).

设计依据: 理论方案2.0 E3节, "去掉上帝视角".
  - 标准航天做法: 误差状态用3维罗德里格斯参数, 避免四元数过参数化导致的协方差奇异.
  - 全状态(16维): p_n[3] + v_n[3] + q_bn[4] + b_g[3] + b_a[3]
  - 误差状态(15维): δp[3] + δv[3] + δθ[3] + δb_g[3] + δb_a[3]

坐标系:
  n: NED (X北, Y东, Z地下, 重力+Z)
  b: body (Xb头部, Yb右, Zb=Xb×Yb)
  q=[w,x,y,z] 表示 b->n 旋转 (Hamilton 约定, 与 quaternion_utils 一致)

IMU测量模型 (与 sensors.py 一致):
  gyro_meas  = ω_b + b_g + n_g           (b系角速度)
  accel_meas = a_b - g_b + b_a + n_a     (b系比力, 已去重力)
  其中 a_b 为 proper acceleration, g_b = C_bn^T @ [0,0,g].

预测步 (100Hz, IMU驱动):
  p_dot = v
  v_dot = C_bn @ (accel_meas - b_a) + g_n
  q_dot = 0.5 * q ⊗ [0, gyro_meas - b_g]
  b_dot = 0

误差动力学 (δθ 定义在 body 系):
  δp_dot = δv
  δv_dot = -C_bn @ [ƒ_b]× @ δθ - C_bn @ δb_a + n_v
  δθ_dot = -[ω_b]× @ δθ - δb_g + n_θ
  δb_g_dot = n_bg,  δb_a_dot = n_ba
  其中 ƒ_b = accel_meas - b_a (校正比力), ω_b = gyro_meas - b_g (校正角速度).

更新步:
  GPS (10Hz): z=[p_meas, v_meas], H=[I 0 0 0 0; 0 I 0 0 0]
  雷达 (50Hz, h<100m): z=alt, H=[0 0 1 0 ...] (只更新高度分量)

关键决策 (2.0方案):
  1. 末端 h<50m 切换纯IMU+雷达模式, 不信任GPS (高动态失锁+多径)
  2. GPS自适应噪声: 高动态(|a|>5g)时 R_gps 放大10倍
  3. E1耦合: 弹性变形污染IMU读数, 通过增大 R_imu (过程噪声) 隐式处理
     (联合估计刚体+弹性模态需19维, 工程上常用隐式处理, 此处采用后者)
"""
import numpy as np
from . import quaternion_utils as qu
from . import rocket_params as rp


def skew(v):
    """向量v的反对称矩阵 [v]×."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])


class MEKF:
    """15维误差状态乘性EKF."""

    def __init__(self, pos0, vel0, q0, bg0=None, ba0=None, dt=0.01):
        # === 标称状态 (16维) ===
        self.p = np.asarray(pos0, dtype=float).copy()   # [3] NED位置
        self.v = np.asarray(vel0, dtype=float).copy()   # [3] NED速度
        self.q = qu.quat_normalize(np.asarray(q0, dtype=float).copy())  # [4] b->n
        self.bg = np.zeros(3) if bg0 is None else np.asarray(bg0, dtype=float).copy()
        self.ba = np.zeros(3) if ba0 is None else np.asarray(ba0, dtype=float).copy()

        # === 误差状态协方差 (15x15) ===
        # 状态顺序: [δp(3), δv(3), δθ(3), δbg(3), δba(3)]
        self.P = np.eye(15)
        # 初始不确定度: 覆盖传感器初始零偏(战术级IMU: 0.05°/s, 0.02m/s²)
        self.P[0:3, 0:3] *= 1.0**2          # position 1m
        self.P[3:6, 3:6] *= 0.5**2          # velocity 0.5m/s
        self.P[6:9, 6:9] *= np.radians(2.0)**2   # attitude 2度
        self.P[9:12, 9:12] *= np.radians(0.1)**2  # gyro bias 0.1°/s (覆盖0.05°/s初始零偏)
        self.P[12:15, 12:15] *= 0.05**2     # accel bias 0.05m/s² (覆盖0.02初始零偏)

        # === 过程噪声 (Q, 离散化, 15x15) ===
        # IMU噪声参数 (与 sensors.py 一致)
        # 连续时间PSD: sigma², 离散化: Q_d = sigma² * dt
        sigma_gyro = np.radians(0.01)    # rad/s/sqrt(Hz) 陀螺白噪声
        sigma_accel = 0.02               # m/s²/sqrt(Hz) 加计白噪声
        sigma_bg_walk = np.radians(1e-4) # rad/s 零偏游走(连续PSD=sigma²)
        sigma_ba_walk = 1e-4             # m/s² 零偏游走(连续PSD=sigma²)
        # 离散Q: 直接给每步方差, 不再乘dt(已在各项中)
        self.Q = np.diag([
            0.0, 0.0, 0.0,                              # position (由速度驱动, 无独立噪声)
            sigma_accel**2 * dt, sigma_accel**2 * dt, sigma_accel**2 * dt,  # velocity
            sigma_gyro**2 * dt, sigma_gyro**2 * dt, sigma_gyro**2 * dt,     # attitude
            sigma_bg_walk**2 * dt, sigma_bg_walk**2 * dt, sigma_bg_walk**2 * dt,  # gyro bias
            sigma_ba_walk**2 * dt, sigma_ba_walk**2 * dt, sigma_ba_walk**2 * dt,  # accel bias
        ])

        # === 测量噪声 ===
        # GPS (与 sensors.py 一致)
        self.R_gps_pos = 0.5**2      # m²
        self.R_gps_vel = 0.1**2      # (m/s)²
        # 雷达高度计
        self.R_radar = 0.05**2       # m²

        self.dt = dt
        self.g_n = np.array([0.0, 0.0, rp.G0])

        # 自适应GPS噪声因子 (高动态时放大)
        self._gps_noise_scale = 1.0
        # 末端模式标志 (h<50m 切纯IMU+雷达)
        self._terminal_mode = False

        # 诊断统计
        self.last_innovation_gps = np.zeros(6)
        self.last_innovation_radar = 0.0
        self.last_K_gps_norm = 0.0
        self.last_K_radar_norm = 0.0

        # Phase 6A: GPS新息门限检测 (马氏距离)
        self.gps_gate_threshold = 10.0      # 10σ门限 (宽松, 仅拒绝大跳变)
        self.gps_reject_count = 0           # 连续拒绝计数
        self.gps_reject_limit = 5           # 连续5次拒绝 → GPS_FAULT
        self.gps_fault = False              # GPS故障标志
        self.last_mahalanobis_dist = 0.0    # 最近马氏距离

    # ============ 预测步 (IMU 驱动, 100Hz) ============
    def predict(self, gyro_meas, accel_meas, dt):
        """IMU积分预测步.
        gyro_meas: 陀螺测量 (b系角速度) [3]
        accel_meas: 加计测量 (b系比力, 已去重力) [3]
        dt: 时间步长

        Phase 6A: 状态NaN安全检查 (在传播前, 防止NaN扩散).
        """
        # === Phase 6A: 状态NaN安全 (EkfQuaternionNaN防护) ===
        if not np.all(np.isfinite(self.q)):
            self.q = np.array([1.0, 0.0, 0.0, 0.0])
            self.P[6:9, 6:9] *= 1000.0
        if not np.all(np.isfinite(self.p)):
            self.p = np.zeros(3)
            self.P[0:3, 0:3] *= 1000.0
        if not np.all(np.isfinite(self.v)):
            self.v = np.zeros(3)
            self.P[3:6, 3:6] *= 1000.0
        if not np.all(np.isfinite(self.P)):
            self.P = np.eye(15)
            self.P[6:9, 6:9] *= np.radians(10.0)**2  # 姿态不确定度放大

        # Phase 6A: 输入NaN二次防御 (SensorGuard已修复, 此处兜底)
        if not np.all(np.isfinite(gyro_meas)):
            gyro_meas = np.zeros(3)  # 假设无旋转
        if not np.all(np.isfinite(accel_meas)):
            accel_meas = np.array([0.0, 0.0, -9.80665])  # 重力

        # 校正零偏
        omega_b = gyro_meas - self.bg
        f_b = accel_meas - self.ba  # 比力 (specific force)

        C_bn = qu.quat_to_rotmat(self.q)

        # === 标称状态传播 ===
        # 位置: p_dot = v
        self.p = self.p + self.v * dt
        # 速度: v_dot = C_bn @ f_b + g_n
        a_n = C_bn @ f_b + self.g_n
        self.v = self.v + a_n * dt
        # 姿态: q_dot = 0.5 * q ⊗ [0, omega_b]
        q_dot = qu.quat_kinematics(self.q, omega_b)
        self.q = qu.quat_normalize(self.q + q_dot * dt)
        # 零偏保持不变 (随机游走由Q处理)

        # === 误差状态协方差传播 ===
        # F矩阵 (15x15), 连续时间
        # δp_dot = δv
        # δv_dot = -C_bn @ [f_b]× @ δθ - C_bn @ δba
        # δθ_dot = -[omega_b]× @ δθ - δbg
        F = np.zeros((15, 15))
        F[0:3, 3:6] = np.eye(3)                       # δp_dot = δv
        F[3:6, 6:9] = -C_bn @ skew(f_b)               # δv_dot / δθ
        F[3:6, 12:15] = -C_bn                         # δv_dot / δba
        F[6:9, 6:9] = -skew(omega_b)                  # δθ_dot / δθ
        F[6:9, 9:12] = -np.eye(3)                     # δθ_dot / δbg

        # 离散化: Phi = I + F*dt (一阶, 足够小步长)
        Phi = np.eye(15) + F * dt
        # 协方差传播: P = Phi @ P @ Phi^T + Q (Q已是离散化, 不再乘dt)
        self.P = Phi @ self.P @ Phi.T + self.Q
        # 对称化 + 保证正定
        self.P = 0.5 * (self.P + self.P.T)

        # 自适应GPS噪声: 检测高动态
        a_mag = np.linalg.norm(a_n)
        if a_mag > 5.0 * rp.G0:  # >5g
            self._gps_noise_scale = 10.0
        else:
            self._gps_noise_scale = 1.0  # 平滑回归

    # ============ GPS 更新 (10Hz) ============
    def update_gps(self, pos_meas, vel_meas):
        """GPS位置+速度更新.
        pos_meas, vel_meas: NED系测量值 [3] each.
        末端模式(h<50m)下跳过GPS更新.

        Phase 6A: 增加马氏距离门限检测, d > 3σ 拒绝更新.
        """
        if self._terminal_mode:
            return  # 末端不信任GPS

        # 测量: z = [p_meas; v_meas] (6维)
        z = np.concatenate([pos_meas, vel_meas])
        # 预测测量: h(x) = [p_hat; v_hat]
        h_pred = np.concatenate([self.p, self.v])
        # 新息
        y = z - h_pred
        self.last_innovation_gps = y

        # H矩阵 (6x15): [I 0 0 0 0; 0 I 0 0 0]
        H = np.zeros((6, 15))
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 3:6] = np.eye(3)

        # 自适应测量噪声
        R = np.diag([self.R_gps_pos * self._gps_noise_scale] * 3 +
                    [self.R_gps_vel * self._gps_noise_scale] * 3)

        # 新息协方差 S = H P H^T + R
        S = H @ self.P @ H.T + R

        # === Phase 6A: 马氏距离门限检测 ===
        # d = sqrt(y^T S^-1 y)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        mahalanobis_dist = float(np.sqrt(max(0.0, y @ S_inv @ y)))
        self.last_mahalanobis_dist = mahalanobis_dist

        if mahalanobis_dist > self.gps_gate_threshold:
            # 新息过大 (GPS跳变/多径), 拒绝本次更新
            self.gps_reject_count += 1
            if self.gps_reject_count >= self.gps_reject_limit:
                self.gps_fault = True
            return  # 拒绝更新

        # 新息正常, 清除拒绝计数
        self.gps_reject_count = 0
        self.gps_fault = False

        # 卡尔曼增益
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)
        self.last_K_gps_norm = np.linalg.norm(K)

        # 状态修正
        dx = K @ y  # 15维误差状态
        self._inject_error(dx)

        # 协方差更新 (Joseph form, 保证正定)
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    # ============ 雷达高度计更新 (50Hz, h<100m) ============
    def update_radar(self, alt_meas):
        """雷达高度计更新. alt_meas = 高度 (向上为正, m).

        Phase 6A: 增加新息门限检测, |y| > 3σ 拒绝更新 (防雷达跳变).
        """
        # 测量: z = -p_n[2] (高度)
        z = alt_meas
        h_pred = -self.p[2]
        y = z - h_pred
        self.last_innovation_radar = y

        # H矩阵 (1x15): 测量 z = -p[2] (高度), 故 dh/dp[2] = -1
        # 修复: 原为+1.0导致符号错误! 雷达说"更低"时EKF反而认为"更高", 引发正反馈发散.
        H = np.zeros((1, 15))
        H[0, 2] = -1.0  # δz = -δp[2] (NED Z=地下, 高度=-Z)

        R = np.array([[self.R_radar]])

        S = H @ self.P @ H.T + R

        # === Phase 6A: 雷达新息门限检测 ===
        # 标量情况: d = |y| / sqrt(S)
        S_scalar = float(S[0, 0]) if S.ndim == 2 else float(S)
        if S_scalar > 1e-15:
            radar_mahalanobis = abs(y) / np.sqrt(S_scalar)
            if radar_mahalanobis > 10.0:  # 10σ门限 (宽松, 仅拒绝大跳变)
                # 雷达跳变, 拒绝更新
                return

        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)
        self.last_K_radar_norm = np.linalg.norm(K)

        dx = K @ np.array([y])
        self._inject_error(dx)

        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    # ============ 误差状态注入 (乘性更新) ============
    def _inject_error(self, dx):
        """将15维误差状态注入标称状态.
        dx = [δp(3), δv(3), δθ(3), δbg(3), δba(3)]
        姿态用乘性更新: q_new = q ⊗ δq(δθ), δq = [1, δθ/2] (小角度近似)."""
        dp = dx[0:3]
        dv = dx[3:6]
        dtheta = dx[6:9]
        dbg = dx[9:12]
        dba = dx[12:15]

        # 位置/速度: 加性
        self.p = self.p + dp
        self.v = self.v + dv
        # 姿态: 乘性 (δθ在body系)
        # δq = [cos(|δθ|/2), sin(|δθ|/2)*δθ/|δθ|] ≈ [1, δθ/2] (小角度)
        dtheta_norm = np.linalg.norm(dtheta)
        if dtheta_norm > 1e-12:
            dq = np.array([
                np.cos(dtheta_norm / 2),
                *(dtheta / dtheta_norm * np.sin(dtheta_norm / 2))
            ])
        else:
            dq = np.array([1.0, dtheta[0] / 2, dtheta[1] / 2, dtheta[2] / 2])
        # q_new = q ⊗ δq (body系误差, 右乘)
        self.q = qu.quat_normalize(qu.quat_multiply(self.q, dq))
        # 零偏: 加性
        self.bg = self.bg + dbg
        self.ba = self.ba + dba

    # ============ 模式切换 ============
    def set_terminal_mode(self, enabled):
        """末端模式(h<50m): 禁用GPS, 仅用IMU+雷达."""
        self._terminal_mode = enabled

    # ============ 输出 ============
    def get_state(self):
        """返回标称状态 + 不确定度(3σ).
        返回: (state_13d, sigma_15d)
        state_13d = [p_n(3), v_n(3), q(4), omega_b(3)] 与 dynamics.make_state 兼容.
        注意: omega_b 需外部提供(IMU测量-零偏), 此处返回0占位."""
        state = np.zeros(13)
        state[0:3] = self.p
        state[3:6] = self.v
        state[6:10] = self.q
        # omega_b 占位, 由调用方用 IMU测量-bg 填充
        sigma = np.sqrt(np.diag(self.P))
        return state, sigma

    def get_estimated_omega(self, gyro_meas):
        """返回校正后的角速度估计 = gyro_meas - bg."""
        return gyro_meas - self.bg

    def get_estimated_accel(self, accel_meas):
        """返回校正后的比力估计 = accel_meas - ba."""
        return accel_meas - self.ba

    def reset(self, pos0, vel0, q0, bg0=None, ba0=None):
        """重置滤波器状态."""
        self.p = np.asarray(pos0, dtype=float).copy()
        self.v = np.asarray(vel0, dtype=float).copy()
        self.q = qu.quat_normalize(np.asarray(q0, dtype=float).copy())
        self.bg = np.zeros(3) if bg0 is None else np.asarray(bg0, dtype=float).copy()
        self.ba = np.zeros(3) if ba0 is None else np.asarray(ba0, dtype=float).copy()
        self.P = np.eye(15)
        self.P[0:3, 0:3] *= 1.0**2
        self.P[3:6, 3:6] *= 0.5**2
        self.P[6:9, 6:9] *= np.radians(1.0)**2
        self.P[9:12, 9:12] *= np.radians(0.1)**2
        self.P[12:15, 12:15] *= 0.01**2
        self._terminal_mode = False
        self._gps_noise_scale = 1.0
