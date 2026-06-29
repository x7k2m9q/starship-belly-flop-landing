"""
星舰 6-DOF 动力学 (Phase 7.0 — 3DOF→6DOF 升级)
==================================================
理论方案 7.0: 13维刚体状态 + 燃料 = 14维

状态向量 (14维):
  X = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, p, q, r, m_fuel]
       |--位置(NED)--| |--速度(NED)--| |----四元数----| |--角速度(b)--| |燃料|

坐标系 (与前项目一致, 锁死):
  NED系 (n系): X=北, Y=东, Z=地(向下为正). 重力 g_n=[0,0,9.80665].
  箭体系 (b系): Xb=头部, Yb=右, Zb=Xb×Yb.
  四元数 q=[w,x,y,z]: b系->n系旋转. v_n = C_b^n(q) @ v_b.
  推力沿 +Xb (头部方向, 减速时朝下).

星舰Belly-Flop姿态:
  BELLY (θ≈85°): 火箭接近水平, 腹部朝下. q从Q_VERT绕Y轴旋转85°.
  LANDING (θ≈0°): 火箭垂直, 头朝上. q=Q_VERT.

关键物理:
  1. 气动力在气流系计算, 转到body系再转到NED系
  2. 推力在body系沿+Xb, 转到NED系
  3. 重力在NED系 [0,0,g]
  4. 四元数运动学: qdot = 0.5 * q ⊗ [0, omega_b]
  5. 欧拉方程: I * domega/dt = M - omega × (I * omega)
  6. 质心漂移: 燃料消耗导致cg后移, 转动惯量变化

积分: RK4, dt=0.01s, 每步四元数归一化
"""
import numpy as np
from src.quaternion_utils import (
    quat_multiply, quat_normalize, quat_to_rotmat, quat_kinematics,
    Q_VERT, SQRT2,
)
from src.belly_flop.aero_model_6dof import (
    aero_forces_and_moments_6dof, atmosphere_6dof,
    get_inertia_tensor, get_mass, gravity,
    S_REF, L_REF, T_MAX, T_IDLE, ISP, G0_ISP,
    M_DRY, M_FUEL_INIT, DELTA_MAX,
    CYB, CLB, CNB,  # 侧向气动导数
)


def euler_angle_to_quat(theta_pitch_deg, phi_roll_deg=0.0, psi_yaw_deg=0.0):
    """
    欧拉角(俯仰/滚转/偏航) -> 四元数.

    星舰Belly-Flop俯仰角定义 (与3DOF一致):
      θ=0°:  火箭垂直, 头朝上 → Q_VERT (绕Y轴90°)
      θ=90°: 火箭水平, 腹部朝下 → identity (绕Y轴0°)

    即: 绕Y轴的旋转角度 = 90° - θ_pitch
    Q_VERT = [sqrt2/2, 0, sqrt2/2, 0] 对应 θ=0° (垂直)

    旋转顺序: ZYX (yaw->pitch->roll), q = q_yaw ⊗ q_pitch ⊗ q_roll
    """
    # 绕Y轴旋转 (90°-θ): θ=0→90°旋转(垂直), θ=90→0°旋转(水平)
    theta_rot = np.deg2rad(90.0 - theta_pitch_deg)
    phi = np.deg2rad(phi_roll_deg)
    psi = np.deg2rad(psi_yaw_deg)

    # 绕Y轴(俯仰): [cos(theta_rot/2), 0, sin(theta_rot/2), 0]
    q_pitch = np.array([np.cos(theta_rot / 2), 0, np.sin(theta_rot / 2), 0])
    # 绕X轴(滚转): [cos(phi/2), sin(phi/2), 0, 0]
    q_roll = np.array([np.cos(phi / 2), np.sin(phi / 2), 0, 0])
    # 绕Z轴(偏航): [cos(psi/2), 0, 0, sin(psi/2)]
    q_yaw = np.array([np.cos(psi / 2), 0, 0, np.sin(psi / 2)])

    q = quat_multiply(q_yaw, quat_multiply(q_pitch, q_roll))
    return quat_normalize(q)


def get_pitch_angle_from_quat(q):
    """
    从四元数提取俯仰角(与3DOF的theta对应).

    体X轴在NED系中的方向:
      x_body_n = C_b^n @ [1,0,0]
    俯仰角 = x_body_n 与水平面(XY)的夹角
      theta = atan2(-x_body_n[z], sqrt(x_body_n[x]² + x_body_n[y]²))

    theta=0°: 垂直(头朝上), x_body_n=[0,0,-1] (向上)
    theta=90°: 水平(腹部朝下), x_body_n=[1,0,0] (向北)
    """
    C = quat_to_rotmat(q)
    x_body_n = C @ np.array([1.0, 0.0, 0.0])
    # theta = 从垂直方向到x_body_n的夹角
    # x_body_n=[0,0,-1](上) -> theta=0
    # x_body_n=[1,0,0](北) -> theta=90°
    horizontal = np.sqrt(x_body_n[0] ** 2 + x_body_n[1] ** 2)
    theta = np.arctan2(horizontal, -x_body_n[2])
    return theta


def get_tilt_angle_from_quat(q):
    """
    tilt = 体X轴与垂直方向(上)的夹角.
    tilt=0°: 垂直. tilt=90°: 水平.
    与get_pitch_angle_from_quat在无滚转时一致.
    """
    C = quat_to_rotmat(q)
    x_body_n = C @ np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, -1.0])  # NED向上
    cos_tilt = np.clip(np.dot(x_body_n, up), -1.0, 1.0)
    return np.arccos(cos_tilt)


def state_derivative_6dof(state, T_cmd, delta_flaps, tvc_gimbal=[0.0, 0.0],
                          M_external=None):
    """
    6-DOF 状态导数.

    参数:
      state: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, p, q, r, m_fuel] (14维)
        pz: NED高度(负值, -10000表示10000m高)
        vz: NED垂直速度(正值=下降)
      T_cmd: 推力指令 (N), 沿+Xb
      delta_flaps: [d_FL, d_FR, d_RL, d_RR] 4片襟翼偏转 (rad)
        FL=前左, FR=前右, RL=后左, RR=后右
      tvc_gimbal: [gimbal_y, gimbal_z] TVC偏转 (rad), 绕Yb和Zb
      M_external: 外部力矩 (body系, N·m), 用于测试/扰动注入

    返回: dstate/dt (14维)
    """
    # 解包状态
    pos_n = state[0:3]        # NED位置
    vel_n = state[3:6]        # NED速度
    q = state[6:10]           # 四元数 [w,x,y,z]
    omega_b = state[10:13]    # body系角速度 [p, q, r]
    m_fuel = state[13]

    # 质量/惯量
    m = get_mass(m_fuel)
    I_tensor = get_inertia_tensor(m_fuel)  # 3x3 对角阵
    g = gravity(-pos_n[2])  # 高度 = -pz

    # 旋转矩阵
    C_bn = quat_to_rotmat(q)  # b->n

    # body系速度
    vel_b = C_bn.T @ vel_n

    # ---- 气动力/力矩 (6DOF) ----
    # 高度 = -pz (NED: pz负=高度正)
    h = -pos_n[2]
    F_aero_b, M_aero_b = aero_forces_and_moments_6dof(
        vel_b, q, h, delta_flaps)

    # ---- 推力 (body系, 含TVC偏转) ----
    # TVC偏转: gimbal_y>0 = 俯仰抬头(正力矩), gimbal_z>0 = 偏航
    # 推力方向: 绕Yb旋转gy, 绕Zb旋转gz
    # 正gy → 推力向+Zb偏 → 尾部被推向+Zb(下) → 抬头(正俯仰)
    gy, gz = tvc_gimbal
    T_dir_b = np.array([
        np.cos(gy) * np.cos(gz),
        np.sin(gz),
        np.sin(gy) * np.cos(gz),
    ])
    T_actual = T_cmd  # 实际推力(点火瞬态在外部处理)
    F_thrust_b = T_actual * T_dir_b

    # 推力力矩 (TVC偏转产生): M = r_TVC × F_thrust
    # 假设推力作用点在质心后方 x_TVC = -L_REF*0.4 (尾部)
    x_tvc = -L_REF * 0.4
    r_tvc_b = np.array([x_tvc, 0.0, 0.0])
    M_thrust_b = np.cross(r_tvc_b, F_thrust_b)

    # ---- 总力/力矩 (body系) ----
    F_total_b = F_aero_b + F_thrust_b
    M_total_b = M_aero_b + M_thrust_b
    if M_external is not None:
        M_total_b = M_total_b + M_external

    # ---- 转到NED系 ----
    F_total_n = C_bn @ F_total_b
    # 重力 (NED系)
    F_gravity_n = np.array([0.0, 0.0, m * g])

    # ---- 运动方程 ----
    # 位置导数 = 速度
    dp_dt = vel_n

    # 速度导数 = (F_aero + F_thrust)/m + gravity (NED系)
    dv_dt = (F_total_n + F_gravity_n) / m

    # 四元数运动学: qdot = 0.5 * q ⊗ [0, omega_b]
    dq_dt = quat_kinematics(q, omega_b)

    # 欧拉方程: I * domega/dt = M - omega × (I * omega)
    I_omega = I_tensor @ omega_b
    gyro_couple = np.cross(omega_b, I_omega)  # 陀螺耦合项
    domega_dt = np.linalg.solve(I_tensor, M_total_b - gyro_couple)

    # 燃料消耗
    dm_dt = -T_actual / (ISP * G0_ISP)

    # 组装导数
    dstate = np.zeros(14)
    dstate[0:3] = dp_dt
    dstate[3:6] = dv_dt
    dstate[6:10] = dq_dt
    dstate[10:13] = domega_dt
    dstate[13] = dm_dt

    return dstate


def rk4_step_6dof(state, T_cmd, delta_flaps, dt, tvc_gimbal=[0.0, 0.0],
                  M_external=None):
    """
    RK4 单步积分 (6-DOF).

    每步结束后:
      1. 四元数归一化 (防止数值漂移)
      2. 燃料非负
    """
    k1 = state_derivative_6dof(state, T_cmd, delta_flaps, tvc_gimbal, M_external)
    k2 = state_derivative_6dof(state + 0.5 * dt * k1, T_cmd, delta_flaps, tvc_gimbal, M_external)
    k3 = state_derivative_6dof(state + 0.5 * dt * k2, T_cmd, delta_flaps, tvc_gimbal, M_external)
    k4 = state_derivative_6dof(state + dt * k3, T_cmd, delta_flaps, tvc_gimbal, M_external)
    new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # 四元数归一化
    new_state[6:10] = quat_normalize(new_state[6:10])

    # 燃料非负
    new_state[13] = max(new_state[13], 0.0)

    return new_state


def make_initial_state_6dof(h_init=10000.0, vz_init=300.0, vx_init=50.0,
                            theta_pitch_deg=85.0, m_fuel=None):
    """
    创建6-DOF初始状态.

    默认: 10km高度, 300m/s下降, 50m/s水平, 85°俯仰(BELLY).
    """
    if m_fuel is None:
        m_fuel = M_FUEL_INIT * 0.7  # 70%燃料

    # NED位置: pz = -h (高度取负)
    pos_n = np.array([0.0, 0.0, -h_init])
    # NED速度: vz正=下降
    vel_n = np.array([vx_init, 0.0, vz_init])
    # 四元数: 85°俯仰
    q = euler_angle_to_quat(theta_pitch_deg)
    # 角速度: 0
    omega_b = np.array([0.0, 0.0, 0.0])

    state = np.zeros(14)
    state[0:3] = pos_n
    state[3:6] = vel_n
    state[6:10] = q
    state[10:13] = omega_b
    state[13] = m_fuel
    return state
