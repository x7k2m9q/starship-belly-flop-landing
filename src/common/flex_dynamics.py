"""E1: 弹性体+推进剂晃动动力学.
物理模型:
  - 弹性体: 悬臂梁前2阶弯曲模态(模态叠加法)
  - 晃动: 等效弹簧-质量模型(NASA-SP-106)
  - 陷波器: 二阶IIR, 在弯曲频率处挖深坑

设计决策: 不改变13维state向量, 弹性/晃动作为内部状态管理.
  弹性变形作为"扰动"输出, 加到IMU测量和气动力上.
  这样dynamics.py/sensors.py/attitude_control.py改动最小.

坐标系: b系, Xb指向头部. 悬臂梁根部在尾部(发动机端), 尖端在头部.
  x∈[0,L], x=0根部(尾部), x=L尖端(头部).
  弹性变形方向: Yb/Zb横向.
"""
import numpy as np
from . import rocket_params as rp


# === 弹性体参数(悬臂梁公式计算) ===
# ω_n = (β_n/L)² * √(EI/μ)
# β1=1.875, β2=4.694 (悬臂梁特征值)
L_BODY = rp.LENGTH  # 30m
# 线密度 μ = 总质量/长度 (满载近似)
MU_LINE = (rp.DRY_MASS + rp.INIT_FUEL) / L_BODY  # ~1733 kg/m
# 抗弯刚度 EI (铝合金薄壁圆筒)
# E=70GPa, I=π/8*(D_out⁴-D_in⁴), 壁厚3mm
_E_AL = 70e9
_T_WALL = 0.003
_D_OUT = rp.DIAMETER
_D_IN = rp.DIAMETER - 2 * _T_WALL
_I_SECTION = np.pi / 8 * (_D_OUT**4 - _D_IN**4)
EI_STIFF = _E_AL * _I_SECTION  # ~2.47e10 N·m²

# 弯曲频率
_OMEGA_FLEX = np.sqrt(EI_STIFF / MU_LINE)  # √(EI/μ) ~3775
OMEGA_FLEX_1 = (1.875 / L_BODY)**2 * _OMEGA_FLEX  # 一阶 ~14.75 rad/s = 2.35Hz
OMEGA_FLEX_2 = (4.694 / L_BODY)**2 * _OMEGA_FLEX  # 二阶 ~92.5 rad/s = 14.7Hz
ZETA_FLEX = 0.02  # 阻尼比(薄壁结构低阻尼)

# 模态形状(悬臂梁近似): φ_n(x) = 0.5*(1-cos(n*π*x/L))
# x=0根部(尾部), x=L尖端(头部)
def flex_mode_shape(x, n):
    """悬臂梁第n阶模态形状. x∈[0,L]."""
    return 0.5 * (1.0 - np.cos(n * np.pi * x / L_BODY))

def flex_mode_slope(x, n):
    """模态形状对x的导数(斜率)."""
    return 0.5 * n * np.pi / L_BODY * np.sin(n * np.pi * x / L_BODY)


# === 晃动参数(NASA-SP-106简化) ===
# 2个罐: LOX(上/头部) + RP-1(下/尾部)
# 罐位置(Xb坐标, 相对几何中心)
X_TANK_LOX = 8.0    # LOX罐在头部方向
X_TANK_RP1 = -6.0   # RP-1罐在尾部方向
# 晃动频率 ω_slosh = √(g/L_eff), L_eff≈罐半径
R_TANK = rp.DIAMETER / 2.0
OMEGA_SLOSH = np.sqrt(rp.G0 / R_TANK)  # ~2.42 rad/s = 0.39Hz
ZETA_SLOSH = 0.01  # 无防晃板, 极低阻尼
# 晃动质量比(随液位变化, 简化用固定值0.5)
SLOSH_MASS_RATIO = 0.5


class FlexDynamics:
    """弹性体+晃动动力学.
    内部状态: eta1, eta1_dot, eta2, eta2_dot(弹性) + xi1, xi1_dot, xi2, xi2_dot(晃动)
    输出: IMU扰动(角速度+加速度), 气动力矩扰动, 晃动力矩.
    """
    def __init__(self):
        # 弹性体模态坐标 [eta1, eta1_dot, eta2, eta2_dot]
        self.eta = np.zeros(4)
        # 晃动位移 [xi1, xi1_dot, xi2, xi2_dot] (2个罐, 各Yb/Zb方向简化为1维)
        self.xi = np.zeros(4)

        # IMU安装位置(Xb坐标, 相对几何中心)
        self.x_imu = 5.0  # IMU在头部方向(靠近LOX罐)
        # 气动压心位置
        self.x_cp = -3.0  # 压心在尾部方向

        # 罐参数
        self.tanks = [
            {'x': X_TANK_LOX, 'm_ratio': SLOSH_MASS_RATIO},
            {'x': X_TANK_RP1, 'm_ratio': SLOSH_MASS_RATIO},
        ]

    def update(self, dt, a_lateral_b, omega_b, fuel_mass, thrust_b=None):
        """更新弹性体和晃动状态.
        a_lateral_b: 横向加速度(b系) [ay, az] m/s²
        omega_b: 刚体角速度(b系) [wx, wy, wz] rad/s
        fuel_mass: 当前燃料质量 kg
        thrust_b: 推力(b系) [Fx,Fy,Fz] N (用于发动机激励)
        返回: (imu_omega_disturb, imu_accel_disturb, aero_moment_disturb, slosh_moment)
        """
        # === 弹性体模态方程 ===
        # η_ddot + 2ζωη_dot + ω²η = Q(t)
        # Q = 广义气动力 + 发动机摆角激励 + 刚体角加速度耦合
        for i in range(2):
            n = i + 1
            omega_n = OMEGA_FLEX_1 if n == 1 else OMEGA_FLEX_2
            eta_ddot = (-2 * ZETA_FLEX * omega_n * self.eta[2*i+1]
                        - omega_n**2 * self.eta[2*i])
            # 发动机激励(推力横向分量通过模态形状导数耦合)
            if thrust_b is not None:
                # 发动机在x=0(根部), 横向推力F_yz激励弯曲
                F_lat = np.sqrt(thrust_b[1]**2 + thrust_b[2]**2)
                phi_prime_0 = flex_mode_slope(0.01, n)  # 根部斜率(近似)
                eta_ddot += phi_prime_0 * F_lat / (MU_LINE * L_BODY)
            # 数值积分(半隐式Euler)
            self.eta[2*i+1] += eta_ddot * dt
            self.eta[2*i] += self.eta[2*i+1] * dt

        # === 晃动方程 ===
        # ξ_ddot + 2ζωξ_dot + ω²ξ = -a_lateral
        for i in range(2):
            a_lat = a_lateral_b[i] if i < len(a_lateral_b) else 0.0
            xi_ddot = (-2 * ZETA_SLOSH * OMEGA_SLOSH * self.xi[2*i+1]
                       - OMEGA_SLOSH**2 * self.xi[2*i] - a_lat)
            self.xi[2*i+1] += xi_ddot * dt
            self.xi[2*i] += self.xi[2*i+1] * dt

        # === 计算扰动输出 ===
        # IMU角速度扰动 = Σ φ'(x_imu) * eta_dot
        imu_omega_dist = np.zeros(3)
        for i in range(2):
            n = i + 1
            phi_prime_imu = flex_mode_slope(self.x_imu + L_BODY/2, n)  # x_imu相对几何中心, 转换到[0,L]
            imu_omega_dist[1 + i] = phi_prime_imu * self.eta[2*i+1]  # Yb/Zb方向

        # IMU加速度扰动 = Σ φ''(x_imu) * eta * ω² (简化)
        imu_accel_dist = np.zeros(3)
        for i in range(2):
            n = i + 1
            omega_n = OMEGA_FLEX_1 if n == 1 else OMEGA_FLEX_2
            # 二阶导数近似: φ''(x) ≈ -(nπ/L)² * φ(x)
            phi_imu = flex_mode_shape(self.x_imu + L_BODY/2, n)
            imu_accel_dist[1 + i] = -(n * np.pi / L_BODY)**2 * phi_imu * self.eta[2*i] * omega_n**2

        # 气动力矩扰动 = 弹性变形改变局部攻角 → 附加气动力矩
        aero_moment_dist = np.zeros(3)
        for i in range(2):
            n = i + 1
            phi_prime_cp = flex_mode_slope(self.x_cp + L_BODY/2, n)
            # 局部攻角变化 ≈ φ'(x_cp) * eta / V (简化, 不除V避免除零)
            aero_moment_dist[1 + i] = phi_prime_cp * self.eta[2*i] * 0.01  # 缩放因子

        # 晃动力矩 = Σ x_tank × F_slosh
        slosh_moment = np.zeros(3)
        m_total = rp.DRY_MASS + fuel_mass
        for i, tank in enumerate(self.tanks):
            m_slosh = tank['m_ratio'] * fuel_mass * 0.1  # 晃动质量(简化)
            F_slosh = -OMEGA_SLOSH**2 * m_slosh * self.xi[2*i]  # 弹簧力
            slosh_moment[1 + i] = tank['x'] * F_slosh  # 力矩 = x × F

        return imu_omega_dist, imu_accel_dist, aero_moment_dist, slosh_moment

    def reset(self):
        """重置弹性/晃动状态."""
        self.eta = np.zeros(4)
        self.xi = np.zeros(4)


class NotchFilter:
    """二阶陷波器(IIR). 在指定频率处挖深坑, 防止控制器激励弯曲共振.
    H(z) = (z² - 2cos(ωn)z + 1) / (z² - 2r*cos(ωn)z + r²)
    ωn = 弯曲频率(归一化), r = 深度因子(0.9~0.99, 越大坑越窄)
    """
    def __init__(self, freq_hz, depth=0.1, sample_rate=100.0):
        wn = 2 * np.pi * freq_hz / sample_rate
        r = 1.0 - depth  # depth越大, r越小, 坑越深
        # 分子系数: z² - 2cos(ωn)z + 1
        self.b = np.array([1.0, -2 * np.cos(wn), 1.0])
        # 分母系数: z² - 2r*cos(ωn)z + r²
        self.a = np.array([1.0, -2 * r * np.cos(wn), r * r])
        self.x_buf = np.zeros(2)  # 输入缓冲
        self.y_buf = np.zeros(2)  # 输出缓冲

    def filter(self, x):
        """滤波单个样本. 返回滤波后的值."""
        # 直接II型实现
        y = (self.b[0] * x + self.b[1] * self.x_buf[0] + self.b[2] * self.x_buf[1]
             - self.a[1] * self.y_buf[0] - self.a[2] * self.y_buf[1]) / self.a[0]
        # 更新缓冲
        self.x_buf[1] = self.x_buf[0]
        self.x_buf[0] = x
        self.y_buf[1] = self.y_buf[0]
        self.y_buf[0] = y
        return y

    def reset(self):
        self.x_buf = np.zeros(2)
        self.y_buf = np.zeros(2)


class NotchFilterBank:
    """陷波器组: 在ω1和ω2处各挖一个坑. 串联在姿态误差通道."""
    def __init__(self, sample_rate=100.0):
        freq1_hz = OMEGA_FLEX_1 / (2 * np.pi)  # ~2.35Hz
        freq2_hz = OMEGA_FLEX_2 / (2 * np.pi)  # ~14.7Hz
        self.notch1 = NotchFilter(freq1_hz, depth=0.1, sample_rate=sample_rate)
        self.notch2 = NotchFilter(freq2_hz, depth=0.1, sample_rate=sample_rate)

    def filter(self, x):
        """串联滤波: 先notch1再notch2."""
        return self.notch2.filter(self.notch1.filter(x))

    def reset(self):
        self.notch1.reset()
        self.notch2.reset()
