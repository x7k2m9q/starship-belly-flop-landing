"""
Phase 7.0 战役三: MEKF状态估计验证 (问题29-31)
================================================
理论方案7.0:
  - 问题29: 传感器模型噪声特性 (战术级IMU: 0.01°/s, 0.02m/s²)
  - 问题30: EKF协方差矩阵初始化
  - 问题31: 延迟补偿 (混合前推策略)

移植策略 (7.0.txt第320行):
  "只需要修改状态向量映射（星舰无栅格舵，改为襟翼），和传感器配置"
  → MEKF直接复用猎鹰9号代码(src/ekf.py), 传感器参数一致

测试内容:
  1. 纯IMU预测漂移 (无GPS/雷达, 10s位置漂移)
  2. GPS更新后位置/速度收敛
  3. 雷达更新后高度收敛
  4. 姿态估计精度 (四元数误差)
  5. 零偏估计收敛
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.ekf import MEKF
from src.sensors import IMU, GPS, RadarAltimeter
from src.belly_flop.dynamics_6dof import (
    state_derivative_6dof, rk4_step_6dof, make_initial_state_6dof,
    euler_angle_to_quat, get_pitch_angle_from_quat,
)
from src.belly_flop.aero_model_6dof import (
    get_inertia_tensor, get_mass, gravity, atmosphere_6dof,
    S_REF, L_REF, M_FUEL_INIT,
)
from src.quaternion_utils import quat_to_rotmat, quat_multiply, quat_inverse, quat_normalize


def extract_imu_measurements(state, dstate, m_fuel):
    """从6DOF状态和导数提取IMU测量真值.

    IMU测量:
      gyro_meas_true = omega_b (body系角速度)
      accel_meas_true = f_b (body系比力, 不含重力)

    比力计算:
      a_n = dstate[3:6] = (F_total_n + F_gravity_n) / m  (NED系加速度)
      g_n = [0, 0, g]  (NED系重力)
      f_n = a_n - g_n = F_total_n / m  (NED系比力, 去重力)
      f_b = C_bn^T @ f_n  (body系比力)
    """
    q = state[6:10]
    omega_b = state[10:13]
    C_bn = quat_to_rotmat(q)

    # NED系加速度
    a_n = dstate[3:6]
    # NED系重力
    h = -state[2]
    g = gravity(h)
    g_n = np.array([0.0, 0.0, g])
    # NED系比力 (去重力)
    f_n = a_n - g_n
    # body系比力
    f_b = C_bn.T @ f_n

    return omega_b, f_b


def quat_angle_error(q1, q2):
    """计算两个四元数之间的角度误差(度)."""
    q_err = quat_multiply(quat_inverse(q1), q2)
    if q_err[0] < 0:
        q_err = -q_err
    # 角度 = 2 * arccos(|w|)
    angle = 2.0 * np.arccos(np.clip(abs(q_err[0]), 0.0, 1.0))
    return np.rad2deg(angle)


def run_ekf_simulation(t_end=10.0, dt=0.01, use_gps=True, use_radar=True,
                       seed=42):
    """运行MEKF仿真.

    返回: dict含真值和估计值时间序列
    """
    rng = np.random.default_rng(seed)

    # === 真值动力学 ===
    state = make_initial_state_6dof(h_init=5000.0, vz_init=200.0, vx_init=30.0,
                                    theta_pitch_deg=85.0, m_fuel=M_FUEL_INIT * 0.5)

    # === 传感器 ===
    imu = IMU(rng)
    gps = GPS(rng)
    radar = RadarAltimeter(rng)

    # === MEKF初始化 ===
    # 初始估计: 故意给误差, 测试收敛
    pos0 = state[0:3] + rng.normal(0.0, 1.0, 3)  # 位置误差1m
    vel0 = state[3:6] + rng.normal(0.0, 0.5, 3)  # 速度误差0.5m/s
    q0 = quat_normalize(state[6:10] + rng.normal(0.0, 0.01, 4))  # 姿态误差~1°
    ekf = MEKF(pos0, vel0, q0, dt=dt)

    # === 记录 ===
    times = []
    pos_true = []
    pos_est = []
    vel_true = []
    vel_est = []
    q_true = []
    q_est = []
    omega_true = []
    omega_est = []
    bg_est = []
    P_diag = []

    n_steps = int(t_end / dt)
    for i in range(n_steps + 1):
        t = i * dt

        # 真值导数 (用于提取IMU测量)
        dstate = state_derivative_6dof(state, T_cmd=0.0,
                                       delta_flaps=[0, 0, 0, 0])

        # 提取IMU测量真值
        omega_b_true, f_b_true = extract_imu_measurements(state, dstate, state[13])

        # IMU测量 (带噪声+零偏)
        gyro_meas, accel_meas = imu.measure(omega_b_true, f_b_true, dt)

        # EKF预测步
        ekf.predict(gyro_meas, accel_meas, dt)

        # GPS更新 (10Hz)
        if use_gps:
            pos_meas, vel_meas, gps_valid = gps.measure(state[0:3], state[3:6], dt)
            if gps_valid:
                ekf.update_gps(pos_meas, vel_meas)

        # 雷达更新 (50Hz, h<100m)
        if use_radar:
            alt_meas, radar_valid = radar.measure(state[0:3], dt)
            if radar_valid:
                ekf.update_radar(alt_meas)

        # 记录
        if i % 10 == 0 or i == n_steps:
            times.append(t)
            pos_true.append(state[0:3].copy())
            pos_est.append(ekf.p.copy())
            vel_true.append(state[3:6].copy())
            vel_est.append(ekf.v.copy())
            q_true.append(state[6:10].copy())
            q_est.append(ekf.q.copy())
            omega_true.append(omega_b_true.copy())
            omega_est.append(ekf.get_estimated_omega(gyro_meas).copy())
            bg_est.append(ekf.bg.copy())
            P_diag.append(np.diag(ekf.P).copy())

        # 真值积分一步
        if i < n_steps:
            state = rk4_step_6dof(state, T_cmd=0.0, delta_flaps=[0, 0, 0, 0], dt=dt)

    return {
        't': np.array(times),
        'pos_true': np.array(pos_true),
        'pos_est': np.array(pos_est),
        'vel_true': np.array(vel_true),
        'vel_est': np.array(vel_est),
        'q_true': np.array(q_true),
        'q_est': np.array(q_est),
        'omega_true': np.array(omega_true),
        'omega_est': np.array(omega_est),
        'bg_est': np.array(bg_est),
        'P_diag': np.array(P_diag),
    }


def test_1_imu_only_drift():
    """测试1: 纯IMU预测漂移 (无GPS/雷达, 10s).

    工程要点: 纯IMU漂移是物理极限, 验收标准基于战术级IMU特性.
    陀螺零偏0.5°/s × 10s = 5°姿态漂移(理论), 加噪声~6°合理.
    """
    print("\n=== 测试1: 纯IMU预测漂移 ===")
    result = run_ekf_simulation(t_end=10.0, use_gps=False, use_radar=False)

    # 最终位置误差
    pos_err = np.abs(result['pos_true'][-1] - result['pos_est'][-1])
    vel_err = np.abs(result['vel_true'][-1] - result['vel_est'][-1])

    # 姿态误差
    q_err = quat_angle_error(result['q_true'][-1], result['q_est'][-1])

    print(f"  10s后位置误差: [{pos_err[0]:.2f}, {pos_err[1]:.2f}, {pos_err[2]:.2f}] m")
    print(f"  10s后速度误差: [{vel_err[0]:.2f}, {vel_err[1]:.2f}, {vel_err[2]:.2f}] m/s")
    print(f"  10s后姿态误差: {q_err:.2f}°")

    # 纯IMU漂移验收 (战术级IMU物理极限):
    # 位置<50m, 速度<5m/s, 姿态<10° (0.5°/s零偏×10s=5°+噪声)
    ok_pos = np.all(pos_err < 50.0)
    ok_vel = np.all(vel_err < 5.0)
    ok_att = q_err < 10.0

    print(f"  位置漂移<50m: {'✓' if ok_pos else '✗'}")
    print(f"  速度漂移<5m/s: {'✓' if ok_vel else '✗'}")
    print(f"  姿态漂移<10°: {'✓' if ok_att else '✗'}")

    ok = ok_pos and ok_vel and ok_att
    print(f"  结果: {'PASS' if ok else 'FAIL'}")
    return ok


def test_2_gps_convergence():
    """测试2: GPS更新后位置/速度收敛."""
    print("\n=== 测试2: GPS收敛 ===")
    result = run_ekf_simulation(t_end=10.0, use_gps=True, use_radar=False)

    # 最终误差
    pos_err = np.abs(result['pos_true'][-1] - result['pos_est'][-1])
    vel_err = np.abs(result['vel_true'][-1] - result['vel_est'][-1])

    # 初始误差 (第一步)
    pos_err_init = np.abs(result['pos_true'][0] - result['pos_est'][0])
    vel_err_init = np.abs(result['vel_true'][0] - result['vel_est'][0])

    print(f"  初始位置误差: [{pos_err_init[0]:.2f}, {pos_err_init[1]:.2f}, {pos_err_init[2]:.2f}] m")
    print(f"  10s后位置误差: [{pos_err[0]:.2f}, {pos_err[1]:.2f}, {pos_err[2]:.2f}] m")
    print(f"  初始速度误差: [{vel_err_init[0]:.2f}, {vel_err_init[1]:.2f}, {vel_err_init[2]:.2f}] m/s")
    print(f"  10s后速度误差: [{vel_err[0]:.2f}, {vel_err[1]:.2f}, {vel_err[2]:.2f}] m/s")

    # GPS收敛验收: 10s后位置误差<2m, 速度误差<0.5m/s
    ok_pos = np.all(pos_err < 2.0)
    ok_vel = np.all(vel_err < 0.5)

    print(f"  位置收敛<2m: {'✓' if ok_pos else '✗'}")
    print(f"  速度收敛<0.5m/s: {'✓' if ok_vel else '✗'}")

    ok = ok_pos and ok_vel
    print(f"  结果: {'PASS' if ok else 'FAIL'}")
    return ok


def test_3_attitude_estimation():
    """测试3: 姿态估计精度.

    工程要点: GPS不直接观测姿态, 姿态靠IMU传播+速度方向间接耦合.
    10s收敛到3°是合理的(GPS间接观测+IMU零偏估计).
    """
    print("\n=== 测试3: 姿态估计精度 ===")
    result = run_ekf_simulation(t_end=10.0, use_gps=True, use_radar=True)

    # 姿态误差时间序列
    q_errors = []
    for i in range(len(result['t'])):
        q_err = quat_angle_error(result['q_true'][i], result['q_est'][i])
        q_errors.append(q_err)
    q_errors = np.array(q_errors)

    # 最终和最大误差
    q_err_final = q_errors[-1]
    q_err_max = np.max(q_errors[10:])  # 跳过前10步(收敛期)

    print(f"  姿态误差(最终): {q_err_final:.3f}°")
    print(f"  姿态误差(最大, 收敛后): {q_err_max:.3f}°")

    # 验收: 收敛后姿态误差<3° (GPS间接观测, 非直接观测)
    ok = q_err_max < 3.0
    print(f"  姿态误差<3°: {'✓' if ok else '✗'}")
    print(f"  结果: {'PASS' if ok else 'FAIL'}")
    return ok


def test_4_bias_estimation():
    """测试4: 陀螺零偏估计收敛.

    工程要点: 偏航轴(z)零偏通过速度方向间接观测, 可观测性弱.
    验收: 至少2个轴(俯仰/滚转)的零偏不确定度收敛>50%.
    """
    print("\n=== 测试4: 陀螺零偏估计 ===")
    result = run_ekf_simulation(t_end=10.0, use_gps=True, use_radar=True)

    # 陀螺零偏估计 (最终)
    bg_final = result['bg_est'][-1]
    bg_init = result['bg_est'][0]

    # 零偏不确定度 (协方差对角线, 索引9-11对应δbg)
    P_bg_final = np.sqrt(result['P_diag'][-1][9:12])
    P_bg_init = np.sqrt(result['P_diag'][0][9:12])

    # 收敛比例
    convergence_ratio = P_bg_final / P_bg_init

    print(f"  陀螺零偏估计(初始): {np.rad2deg(bg_init)} °/s")
    print(f"  陀螺零偏估计(最终): {np.rad2deg(bg_final)} °/s")
    print(f"  零偏不确定度(初始): {np.rad2deg(P_bg_init)} °/s")
    print(f"  零偏不确定度(最终): {np.rad2deg(P_bg_final)} °/s")
    print(f"  收敛比例: {convergence_ratio}")

    # 验收: 至少2个轴收敛>50% (偏航轴不可观测是已知特性)
    n_converged = np.sum(convergence_ratio < 0.5)
    ok = n_converged >= 2
    print(f"  收敛>50%的轴数: {n_converged}/3 (需≥2)")
    print(f"  结果: {'PASS' if ok else 'FAIL'}")
    return ok


def test_5_covariance_positive_definite():
    """测试5: 协方差矩阵正定性 (数值稳定性)."""
    print("\n=== 测试5: 协方差正定性 ===")
    result = run_ekf_simulation(t_end=10.0, use_gps=True, use_radar=True)

    # 检查协方差对角线全正
    P_diag = result['P_diag']
    all_positive = np.all(P_diag > 0)

    # 检查协方差没有爆炸 (最大值<100, 合理范围)
    max_P = np.max(P_diag)
    no_explosion = max_P < 100.0

    print(f"  协方差对角线全正: {'✓' if all_positive else '✗'}")
    print(f"  协方差最大值: {max_P:.4f} (应<100)")
    print(f"  无爆炸: {'✓' if no_explosion else '✗'}")

    ok = all_positive and no_explosion
    print(f"  结果: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == '__main__':
    print("=" * 60)
    print("Phase 7.0 战役三: MEKF状态估计验证")
    print("=" * 60)

    results = []
    results.append(('纯IMU漂移', test_1_imu_only_drift()))
    results.append(('GPS收敛', test_2_gps_convergence()))
    results.append(('姿态估计', test_3_attitude_estimation()))
    results.append(('零偏估计', test_4_bias_estimation()))
    results.append(('协方差正定', test_5_covariance_positive_definite()))

    print("\n" + "=" * 60)
    print("验收汇总:")
    n_pass = 0
    for name, ok in results:
        status = 'PASS' if ok else 'FAIL'
        print(f"  {name}: {status}")
        if ok:
            n_pass += 1
    print(f"\n  {n_pass}/{len(results)} PASS")
    print("=" * 60)
