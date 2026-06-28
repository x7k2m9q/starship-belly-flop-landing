"""
Phase 7.0 战役一: 6DOF物理底座开环验证
========================================
验收标准 (理论方案7.0):
  1. 自由落体位移误差 < 1e-6
  2. 四元数范数1000步后误差 < 1e-6
  3. 质心后移单调 (满载→空载)
  4. 气动力方向正确 (阻力向后, 升力垂直速度, 侧力垂直对称面)
  5. 恒定力矩角速度线性增加
  6. 转动惯量随燃料消耗变化
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.belly_flop.dynamics_6dof import (
    state_derivative_6dof, rk4_step_6dof, make_initial_state_6dof,
    euler_angle_to_quat, get_pitch_angle_from_quat, get_tilt_angle_from_quat,
)
from src.belly_flop.aero_model_6dof import (
    aero_forces_and_moments_6dof, airflow_angles_6dof,
    get_inertia_tensor, get_mass, gravity,
    aero_coefficients, trim_flaps_6dof,
    S_REF, L_REF, M_DRY, M_FUEL_INIT, T_MAX, T_IDLE,
    CYB, CLB, CNB, DELTA_MAX,
)
from src.quaternion_utils import quat_to_rotmat, quat_normalize, Q_VERT


def test_1_free_fall():
    """测试1: 自由落体 (无气动力, 无推力).

    工程级验证: 用scipy高精度积分器(DOP853)作为参考解.
    原因: gravity(h)是高度函数, 变重力下无简单解析解.
    100km高度下落4.75m, g变化约1.47e-6, 解析解0.5*g*t²误差~1e-6.
    用DOP853(自适应8阶)作为真值, 验证dt=0.01的RK4精度.
    """
    from scipy.integrate import solve_ivp

    print("\n=== 测试1: 自由落体 ===")
    # 高空100km, 无大气, 纯重力
    state0 = make_initial_state_6dof(h_init=100000.0, vz_init=0.0, vx_init=0.0,
                                     theta_pitch_deg=0.0, m_fuel=0.0)
    dt = 0.01
    t_end = 1.0

    # 待测: dt=0.01的RK4
    state = state0.copy()
    for i in range(int(t_end / dt)):
        state = rk4_step_6dof(state, T_cmd=0.0, delta_flaps=[0, 0, 0, 0], dt=dt)

    # 参考解: scipy DOP853 (自适应步长, 8阶精度)
    def dynamics(t, y):
        return state_derivative_6dof(y, T_cmd=0.0, delta_flaps=[0, 0, 0, 0])

    sol = solve_ivp(dynamics, [0, t_end], state0, method='DOP853',
                    rtol=1e-12, atol=1e-12, dense_output=True)
    state_ref = sol.y[:, -1]

    # 解析解(恒定重力, 仅作参考显示)
    g0 = gravity(100000.0)
    disp_analytic = 0.5 * g0 * t_end ** 2
    vz_analytic = g0 * t_end

    # 实际位移
    h_actual = -state[2]
    disp_actual = 100000.0 - h_actual
    vz_actual = state[5]

    # 参考解位移
    h_ref = -state_ref[2]
    disp_ref = 100000.0 - h_ref
    vz_ref = state_ref[5]

    # 误差 = 实际 - 参考解 (验证RK4精度, 不含变重力模型误差)
    disp_err = abs(disp_actual - disp_ref)
    vz_err = abs(vz_actual - vz_ref)

    # 同时显示与解析解的差异(含变重力效应)
    disp_err_analytic = abs(disp_actual - disp_analytic)

    print(f"  解析解(恒g): disp={disp_analytic:.6f}, vz={vz_analytic:.6f}")
    print(f"  参考解(DOP853): disp={disp_ref:.6f}, vz={vz_ref:.6f}")
    print(f"  实际(RK4 dt=0.01): disp={disp_actual:.6f}, vz={vz_actual:.6f}")
    print(f"  vs参考解误差: disp={disp_err:.2e}, vz={vz_err:.2e}")
    print(f"  vs解析解误差: disp={disp_err_analytic:.2e} (含变重力效应)")
    print(f"  结果: {'PASS' if disp_err < 1e-6 and vz_err < 1e-6 else 'FAIL'}")
    return disp_err < 1e-6 and vz_err < 1e-6


def test_2_quaternion_norm():
    """测试2: 四元数范数保持."""
    print("\n=== 测试2: 四元数范数 (1000步) ===")
    state = make_initial_state_6dof(h_init=10000.0, vz_init=300.0, vx_init=50.0,
                                    theta_pitch_deg=85.0)
    dt = 0.01

    max_norm_err = 0.0
    for i in range(1000):
        state = rk4_step_6dof(state, T_cmd=0.0,
                              delta_flaps=[0, 0, 0, 0], dt=dt)
        q = state[6:10]
        norm = np.linalg.norm(q)
        err = abs(norm - 1.0)
        if err > max_norm_err:
            max_norm_err = err

    print(f"  1000步后四元数范数最大误差: {max_norm_err:.2e}")
    print(f"  结果: {'PASS' if max_norm_err < 1e-6 else 'FAIL'}")
    return max_norm_err < 1e-6


def test_3_constant_torque():
    """测试3: 恒定力矩下角速度线性增加."""
    print("\n=== 测试3: 恒定力矩 (角速度线性增加) ===")
    # 无气动力(高空), 施加恒定推力+TVC偏转产生恒定俯仰力矩
    state = make_initial_state_6dof(h_init=100000.0, vz_init=0.0, vx_init=0.0,
                                    theta_pitch_deg=0.0, m_fuel=0.0)
    dt = 0.01
    t_end = 1.0

    # TVC偏转5°产生俯仰力矩
    gimbal_y = np.deg2rad(5.0)
    T = T_IDLE  # 恒定推力

    for i in range(int(t_end / dt)):
        state = rk4_step_6dof(state, T_cmd=T,
                              delta_flaps=[0, 0, 0, 0], dt=dt,
                              tvc_gimbal=[gimbal_y, 0.0])

    # 理论: M = T * sin(gimbal) * x_tvc
    # Iyy * dq/dt = M (忽略陀螺耦合, 初始omega=0)
    I_tensor = get_inertia_tensor(0.0)
    Iyy = I_tensor[1, 1]
    x_tvc = -L_REF * 0.4
    M_thrust = T * np.sin(gimbal_y) * abs(x_tvc)  # 力矩大小
    q_expected = M_thrust / Iyy * t_end  # 角速度 = M/I * t

    q_actual = state[11]  # pitch rate (body系q分量)

    q_err = abs(q_actual - q_expected) / max(abs(q_expected), 1e-10)

    print(f"  Iyy = {Iyy:.2f} kg·m²")
    print(f"  推力力矩 M = {M_thrust:.2f} N·m")
    print(f"  角速度: 理论={q_expected:.6f}, 实际={q_actual:.6f}, 相对误差={q_err:.2e}")
    print(f"  结果: {'PASS' if q_err < 0.01 else 'FAIL'}")
    return q_err < 0.01


def test_4_aero_force_directions():
    """测试4: 气动力方向正确性."""
    print("\n=== 测试4: 气动力方向 ===")
    h = 5000.0  # 中等高度
    rho, a_sound, _, _ = atmosphere_6dof(h)

    results = []

    # 4a: α=0° (顺气流), 阻力应沿-Xb, 升力≈0
    vel_b = np.array([300.0, 0.0, 0.0])  # 纯轴向流
    F, M = aero_forces_and_moments_6dof(vel_b, Q_VERT, h, [0, 0, 0, 0])
    print(f"  α=0°: Fx={F[0]:.1f}N (应<0,阻力), Fz={F[2]:.1f}N (应≈0,无升力)")
    ok_a = F[0] < 0 and abs(F[2]) < abs(F[0]) * 0.1
    results.append(ok_a)

    # 4b: α=85° (BELLY), 阻力大, 升力存在
    # body系: u=V·cos(85°), w=V·sin(85°)
    V = 300.0
    alpha = np.deg2rad(85.0)
    vel_b = np.array([V * np.cos(alpha), 0.0, V * np.sin(alpha)])
    F, M = aero_forces_and_moments_6dof(vel_b, Q_VERT, h, [0, 0, 0, 0])
    print(f"  α=85°: Fx={F[0]:.1f}N, Fz={F[2]:.1f}N, |F|={np.linalg.norm(F):.1f}N")
    ok_b = np.linalg.norm(F) > 1000  # 大攻角应有大阻力
    results.append(ok_b)

    # 4c: β=5° (侧滑), 应有侧向力 Fy
    vel_b = np.array([300.0, 300.0 * np.sin(np.deg2rad(5.0)) / np.cos(np.deg2rad(0)), 0.0])
    # 简化: v = V·sin(β), u = V·cos(β)
    beta = np.deg2rad(5.0)
    vel_b = np.array([300.0 * np.cos(beta), 300.0 * np.sin(beta), 0.0])
    F, M = aero_forces_and_moments_6dof(vel_b, Q_VERT, h, [0, 0, 0, 0])
    print(f"  β=5°: Fy={F[1]:.1f}N (应<0,反向侧力), Mx={M[0]:.1f}N·m, Mz={M[2]:.1f}N·m")
    ok_c = F[1] < 0  # CYβ<0, 侧滑产生反向侧力
    results.append(ok_c)

    # 4d: 襟翼俯仰力矩方向
    # 前翼正偏(抬头+), 后翼0
    d_flaps = [np.deg2rad(10), np.deg2rad(10), 0.0, 0.0]
    vel_b = np.array([300.0, 0.0, 0.0])
    F, M = aero_forces_and_moments_6dof(vel_b, Q_VERT, h, d_flaps)
    print(f"  前翼+10°: My={M[1]:.1f}N·m (应>0,抬头)")
    ok_d = M[1] > 0
    results.append(ok_d)

    # 4e: 差动滚转力矩
    # FL+RR正, FR+RL负 → 正滚转
    d_flaps = [np.deg2rad(10), np.deg2rad(-10), np.deg2rad(-10), np.deg2rad(10)]
    F, M = aero_forces_and_moments_6dof(vel_b, Q_VERT, h, d_flaps)
    print(f"  差动(FL+RR+,FR+RL-): Mx={M[0]:.1f}N·m (应>0,正滚转)")
    ok_e = M[0] > 0
    results.append(ok_e)

    all_pass = all(results)
    print(f"  结果: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_5_mass_properties():
    """测试5: 质量特性 (转动惯量变化)."""
    print("\n=== 测试5: 质量特性 ===")
    # 满载
    I_full = get_inertia_tensor(M_FUEL_INIT)
    # 空载
    I_empty = get_inertia_tensor(0.0)
    # 半载
    I_half = get_inertia_tensor(M_FUEL_INIT * 0.5)

    print(f"  Iyy 满载: {I_full[1,1]:.2f}")
    print(f"  Iyy 半载: {I_half[1,1]:.2f}")
    print(f"  Iyy 空载: {I_empty[1,1]:.2f}")

    # Iyy应随燃料减少而减小
    ok = I_full[1, 1] > I_half[1, 1] > I_empty[1, 1]
    iyy_change = (I_full[1, 1] - I_empty[1, 1]) / I_full[1, 1] * 100
    print(f"  Iyy变化: {iyy_change:.1f}% (满载→空载)")
    print(f"  结果: {'PASS' if ok else 'FAIL'}")
    return ok


def test_6_pitch_angle_consistency():
    """测试6: 俯仰角与四元数一致性."""
    print("\n=== 测试6: 俯仰角-四元数一致性 ===")
    results = []

    for theta_deg in [0, 30, 60, 85, 90]:
        q = euler_angle_to_quat(theta_deg)
        theta_recovered = np.rad2deg(get_pitch_angle_from_quat(q))
        tilt = np.rad2deg(get_tilt_angle_from_quat(q))
        err = abs(theta_recovered - theta_deg)
        print(f"  θ={theta_deg}°: 恢复={theta_recovered:.2f}°, tilt={tilt:.2f}°, 误差={err:.4f}°")
        ok = err < 0.01
        results.append(ok)

    all_pass = all(results)
    print(f"  结果: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_7_belly_flop_trajectory():
    """测试7: BELLY阶段开环轨迹 (验证气动减速有效)."""
    print("\n=== 测试7: BELLY开环轨迹 ===")
    state = make_initial_state_6dof(h_init=10000.0, vz_init=300.0, vx_init=50.0,
                                    theta_pitch_deg=85.0)
    dt = 0.01
    t_end = 10.0

    h_init = -state[2]
    vz_init = state[5]
    V_init = np.sqrt(state[3]**2 + state[5]**2)

    for i in range(int(t_end / dt)):
        # 配平襟翼
        from src.belly_flop.aero_model_6dof import trim_flaps_6dof, aero_coefficients
        vel_b = quat_to_rotmat(state[6:10]).T @ state[3:6]
        alpha, beta, V = airflow_angles_6dof(vel_b)
        h = -state[2]
        rho, a_sound, _, _ = atmosphere_6dof(h)
        M = V / a_sound if a_sound > 0 else 0
        d_flaps = trim_flaps_6dof(alpha, M)

        state = rk4_step_6dof(state, T_cmd=0.0, delta_flaps=d_flaps, dt=dt)

    h_final = -state[2]
    vz_final = state[5]
    V_final = np.sqrt(state[3]**2 + state[5]**2)
    theta_final = np.rad2deg(get_pitch_angle_from_quat(state[6:10]))

    print(f"  初始: h={h_init:.0f}m, V={V_init:.1f}m/s, vz={vz_init:.1f}m/s, θ=85°")
    print(f"  10s后: h={h_final:.0f}m, V={V_final:.1f}m/s, vz={vz_final:.1f}m/s, θ={theta_final:.1f}°")
    print(f"  减速: ΔV={V_init - V_final:.1f}m/s (气动减速应有效)")
    print(f"  下降: Δh={h_init - h_final:.0f}m")

    # 验收: 气动减速有效(V减小), θ保持85°附近(配平)
    ok = (V_final < V_init) and (abs(theta_final - 85.0) < 10.0)
    print(f"  结果: {'PASS' if ok else 'FAIL'}")
    return ok


# 便捷导入
from src.belly_flop.aero_model_6dof import atmosphere_6dof


if __name__ == '__main__':
    print("=" * 60)
    print("Phase 7.0 战役一: 6DOF物理底座开环验证")
    print("=" * 60)

    results = []
    results.append(('自由落体', test_1_free_fall()))
    results.append(('四元数范数', test_2_quaternion_norm()))
    results.append(('恒定力矩', test_3_constant_torque()))
    results.append(('气动力方向', test_4_aero_force_directions()))
    results.append(('质量特性', test_5_mass_properties()))
    results.append(('俯仰角一致性', test_6_pitch_angle_consistency()))
    results.append(('BELLY开环轨迹', test_7_belly_flop_trajectory()))

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
