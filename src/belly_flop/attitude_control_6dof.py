"""
星舰 6-DOF 姿态控制器 (Phase 7.0 战役二)
==========================================
理论方案7.0 问题17-18:
  - 问题17: PD增益在6DOF下需重新整定 (Kp=2·I·wn², Kd=2·ζ·wn·I)
  - 问题18: 四元数控制律双覆盖错误 (sign(qw)处理)

控制律: 四元数误差PD (复用前项目架构)
  e_q = q_actual^{-1} ⊗ q_des  (body系误差, 修复10)
  e_vec = e_q[1:4]  (矢量部分, ≈θ_err/2)
  sign(qw)处理双覆盖: w<0整体取反
  M_cmd = Kp·e_vec + Kd·e_omega + I·omega_des_dot

增益物理整定 (自适应惯量):
  Kp = 2·I·wn²  (补偿e_vec=θ/2因子, 使闭环为标准二阶)
  Kd = 2·ζ·wn·I
  wn = 2π·0.5 ≈ 3.14 rad/s (~0.5Hz)
  ζ = 0.9

陷波滤波器 (复用flex_dynamics.NotchFilterBank):
  中心频率: 2.35Hz, 14.7Hz (结构模态)
  串联在pitch/yaw误差通道, 防止激励弯曲共振
  滚转通道(Xb)无弯曲耦合, 不过滤

输出:
  M_cmd → allocate_flaps → 4片襟翼偏转
  TVC gimbal → 推力矢量控制 (俯仰/偏航)
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.quaternion_utils import quat_multiply, quat_inverse, quat_normalize
from src.flex_dynamics import NotchFilterBank
from src.belly_flop.control_allocation_6dof import allocate_flaps_normalized
from src.belly_flop.aero_model_6dof import (
    get_inertia_tensor, DELTA_MAX,
)


class AttitudeController6DOF:
    """6-DOF姿态控制器 (四元数PD + 陷波滤波)."""

    def __init__(self, wn=2 * np.pi * 0.5, zeta=0.9, sample_rate=100.0,
                 use_notch=True):
        """
        参数:
          wn: 自然频率 (rad/s), 默认~0.5Hz
          zeta: 阻尼比, 默认0.9
          sample_rate: 采样率 (Hz), 默认100
          use_notch: 是否使用陷波滤波器
        """
        self.wn = wn
        self.zeta = zeta
        self.sample_rate = sample_rate
        self.use_notch = use_notch

        # 陷波器组 (pitch/yaw各一组, 含角速度通道)
        if use_notch:
            self.notch_pitch = NotchFilterBank(sample_rate=sample_rate)
            self.notch_yaw = NotchFilterBank(sample_rate=sample_rate)
            self.notch_pitch_omega = NotchFilterBank(sample_rate=sample_rate)
            self.notch_yaw_omega = NotchFilterBank(sample_rate=sample_rate)

        # 状态记录
        self.last_M_cmd = np.zeros(3)
        self.last_delta_flaps = np.zeros(4)
        self.last_tvc_gimbal = np.zeros(2)

    def compute_torque(self, q_des, omega_des, q_actual, omega_actual,
                       I_body, m_fuel, Q_dyn, omega_des_dot=None):
        """
        计算期望力矩和襟翼/TVC指令.

        参数:
          q_des: 期望四元数 [w,x,y,z]
          omega_des: 期望角速度 (body系) [p,q,r]
          q_actual: 实际四元数
          omega_actual: 实际角速度 (body系)
          I_body: 3x3转动惯量张量
          m_fuel: 当前燃料 (kg, 用于惯量自适应)
          Q_dyn: 动压 (Pa)
          omega_des_dot: 期望角加速度 (前馈)

        返回:
          M_cmd: 期望力矩 (body系) [Mx,My,Mz]
          delta_flaps: 4片襟翼偏转 [d_FL,d_FR,d_RL,d_RR] (rad)
          tvc_gimbal: TVC偏转 [gimbal_y, gimbal_z] (rad)
        """
        if omega_des_dot is None:
            omega_des_dot = np.zeros(3)

        # ---- 四元数误差 (body系, 修复10) ----
        # e_q = q_actual^{-1} ⊗ q_des
        # e_vec[0]=滚转, e_vec[1]=俯仰, e_vec[2]=偏航
        e_q = quat_multiply(quat_inverse(q_actual), q_des)
        if e_q[0] < 0.0:
            e_q = -e_q
        e_vec = e_q[1:4].copy()

        # 角速度误差
        e_omega = omega_des - omega_actual

        # ---- 陷波滤波 (pitch/yaw通道) ----
        if self.use_notch:
            e_vec[1] = self.notch_pitch.filter(e_vec[1])
            e_vec[2] = self.notch_yaw.filter(e_vec[2])
            e_omega[1] = self.notch_pitch_omega.filter(e_omega[1])
            e_omega[2] = self.notch_yaw_omega.filter(e_omega[2])

        # ---- 增益 (自适应惯量) ----
        # Kp = 2·I·wn² (补偿e_vec=θ/2)
        # Kd = 2·ζ·wn·I
        i_diag = np.diag(I_body)
        Kp = 2.0 * i_diag * (self.wn ** 2)
        Kd = 2.0 * self.zeta * self.wn * i_diag

        # ---- 力矩指令 ----
        M_cmd = Kp * e_vec + Kd * e_omega + I_body @ omega_des_dot

        # ---- 控制分配 ----
        # 襟翼分配 (归一化, 防饱和保持方向)
        delta_flaps = allocate_flaps_normalized(M_cmd, Q_dyn)

        # TVC分配 (俯仰/偏航, 滚转不可控)
        # TVC力矩 = T·sin(gimbal)·L_tvc
        # 简化: TVC辅助襟翼, 在低动压时主导
        tvc_gimbal = np.zeros(2)
        # TVC增益 (在低动压时增大)
        tvc_gain = max(0.0, 1.0 - Q_dyn / 5000.0)  # Q<5000时TVC活跃
        if tvc_gain > 0.01:
            # TVC俯仰: gimbal_y
            tvc_gimbal[0] = np.clip(
                M_cmd[1] * tvc_gain / (1e6 + 1e-10),
                -np.deg2rad(10), np.deg2rad(10)
            )
            # TVC偏航: gimbal_z
            tvc_gimbal[1] = np.clip(
                M_cmd[2] * tvc_gain / (1e6 + 1e-10),
                -np.deg2rad(10), np.deg2rad(10)
            )

        # 记录状态
        self.last_M_cmd = M_cmd.copy()
        self.last_delta_flaps = delta_flaps.copy()
        self.last_tvc_gimbal = tvc_gimbal.copy()

        return M_cmd, delta_flaps, tvc_gimbal

    def reset(self):
        """重置控制器状态."""
        self.last_M_cmd = np.zeros(3)
        self.last_delta_flaps = np.zeros(4)
        self.last_tvc_gimbal = np.zeros(2)
        if self.use_notch:
            self.notch_pitch.reset()
            self.notch_yaw.reset()
            self.notch_pitch_omega.reset()
            self.notch_yaw_omega.reset()


def make_desired_quaternion(theta_pitch_deg, phi_roll_deg=0.0, psi_yaw_deg=0.0):
    """
    创建期望四元数 (便捷接口).

    theta_pitch_deg: 俯仰角 (0°=垂直, 90°=水平)
    phi_roll_deg: 滚转角
    psi_yaw_deg: 偏航角
    """
    from src.belly_flop.dynamics_6dof import euler_angle_to_quat
    return euler_angle_to_quat(theta_pitch_deg, phi_roll_deg, psi_yaw_deg)
