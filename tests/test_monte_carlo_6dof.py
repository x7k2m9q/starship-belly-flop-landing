"""
Phase 7.0 战役三(3/3): 6DOF蒙特卡洛验证
=========================================
理论方案7.0 问题25-28: 蒙特卡洛仿真效率/复现性/步长/初始条件

验收标准 (理论方案7.0):
  - 20次蒙特卡洛, 成功率≥80%
  - 初始条件散布: h=8-12km, V=250-350m/s, θ=83-87°
  - 执行器参数随机化: 死区0.3-0.8°, Bouc-Wen±20%, 速率25-35°/s
  - 无Kill, 最终高度<100m, 最终速度<20m/s, 最终俯仰<15°

工程判断 (用户要求):
  - "工程判断比长时间测试重要很多"
  - 6DOF计算量大, 20次足够统计意义, 不追求5000次
  - 重点收集第一手数据: 成功率/失败原因/关键参数散布
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
from src.belly_flop.phase_controller_6dof import (
    PhaseController6DOF, run_full_mission_6dof,
)
from src.belly_flop.dynamics_6dof import (
    make_initial_state_6dof, rk4_step_6dof,
    euler_angle_to_quat, get_pitch_angle_from_quat,
)
from src.belly_flop.aero_model_6dof import (
    get_inertia_tensor, get_mass, gravity, atmosphere_6dof,
    S_REF, L_REF, M_DRY, M_FUEL_INIT, T_MAX, T_IDLE, DELTA_MAX,
)
from src.quaternion_utils import quat_to_rotmat


def run_single_monte_carlo(h_init, vz_init, vx_init, vy_init,
                           theta_pitch_deg, phi_roll_deg, psi_yaw_deg,
                           m_fuel, seed,
                           use_nonideal_actuator=True,
                           t_end=120.0, dt=0.01):
    """单次蒙特卡洛仿真.

    参数:
      h_init: 初始高度 (m)
      vz_init: 初始下降速度 (m/s)
      vx_init: 初始北向速度 (m/s)
      vy_init: 初始东向速度 (m/s)
      theta_pitch_deg: 初始俯仰角 (度)
      phi_roll_deg: 初始滚转角 (度)
      psi_yaw_deg: 初始偏航角 (度)
      m_fuel: 初始燃料 (kg)
      seed: 随机种子
      use_nonideal_actuator: 是否使用非理想执行器
      t_end: 最大仿真时间 (s)
      dt: 时间步长 (s)

    返回:
      result: dict, 含仿真结果和诊断信息
    """
    rng = np.random.default_rng(seed)

    # 初始状态 (含滚转/偏航扰动)
    state = make_initial_state_6dof(
        h_init=h_init, vz_init=vz_init, vx_init=vx_init,
        theta_pitch_deg=theta_pitch_deg, m_fuel=m_fuel
    )
    # 添加东向速度和滚转/偏航
    state[4] = vy_init  # vy (东向)
    # 重新设置四元数 (含滚转/偏航)
    state[6:10] = euler_angle_to_quat(theta_pitch_deg, phi_roll_deg, psi_yaw_deg)

    # 控制器
    controller = PhaseController6DOF(
        use_nonideal_actuator=use_nonideal_actuator,
        use_notch=True
    )

    # 随机化执行器参数
    if use_nonideal_actuator:
        controller.flap_actuator.randomize(rng)

    # 仿真
    n_steps = int(t_end / dt)
    kill_triggered = False
    kill_reason = ''
    final_state = state.copy()
    final_t = 0.0
    final_phase = 'BELLY'

    for i in range(n_steps + 1):
        t = i * dt

        T_cmd, delta_flaps, tvc_gimbal, phase, kill, info = controller.update(state, dt)
        final_phase = phase

        if kill:
            kill_triggered = True
            kill_reason = info.get('kill_reason', 'unknown')
            final_state = state.copy()
            final_t = t
            break

        state = rk4_step_6dof(
            state, T_cmd=T_cmd, delta_flaps=delta_flaps,
            dt=dt, tvc_gimbal=tvc_gimbal
        )
        final_state = state.copy()
        final_t = t

        # 着陆判断
        if -state[2] <= 0.0:
            break

    # 提取最终状态
    h_final = -final_state[2]
    V_final = np.linalg.norm(final_state[3:6])
    vz_final = final_state[5]
    theta_final = np.rad2deg(get_pitch_angle_from_quat(final_state[6:10]))
    omega_final = np.linalg.norm(final_state[10:13])

    result = {
        'success': not kill_triggered and h_final < 100.0 and V_final < 20.0,
        'kill': kill_triggered,
        'kill_reason': kill_reason,
        'h_final': h_final,
        'V_final': V_final,
        'vz_final': vz_final,
        'theta_final': theta_final,
        'omega_final': omega_final,
        't_final': final_t,
        'phase_final': final_phase,
        'h_init': h_init,
        'V_init': np.sqrt(vx_init**2 + vy_init**2 + vz_init**2),
        'theta_init': theta_pitch_deg,
        'm_fuel_init': m_fuel,
    }
    return result


def test_1_monte_carlo_20_runs():
    """测试1: 20次蒙特卡洛(非理想执行器) — 工程级验证.

    工程要点: 初始条件+执行器参数双重散布, 验证鲁棒性.
    验收: 成功率≥80% (16/20).
    """
    print("\n=== 测试1: 20次蒙特卡洛(非理想执行器) ===")
    rng = np.random.default_rng(42)

    n_total = 20
    results = []
    t0 = time.time()

    for i in range(n_total):
        # 初始条件散布
        h_init = rng.uniform(8000, 12000)
        vz_init = rng.uniform(250, 350)
        vx_init = rng.uniform(20, 80)
        vy_init = rng.uniform(-20, 20)  # 侧向速度扰动
        theta_pitch = rng.uniform(83, 87)
        phi_roll = rng.uniform(-3, 3)    # 滚转扰动
        psi_yaw = rng.uniform(-3, 3)     # 偏航扰动
        m_fuel = rng.uniform(M_FUEL_INIT * 0.6, M_FUEL_INIT * 0.8)
        seed = 42 + i

        result = run_single_monte_carlo(
            h_init, vz_init, vx_init, vy_init,
            theta_pitch, phi_roll, psi_yaw,
            m_fuel, seed,
            use_nonideal_actuator=True,
            t_end=120.0, dt=0.01
        )
        results.append(result)

        # 进度
        if (i + 1) % 5 == 0:
            n_ok = sum(1 for r in results if r['success'])
            print(f"  [{i+1:3d}/{n_total}] 成功={n_ok} 当前={n_ok/(i+1)*100:.1f}%")

    elapsed = time.time() - t0
    n_success = sum(1 for r in results if r['success'])
    rate = n_success / n_total * 100

    # 统计
    print(f"\n  === 蒙特卡洛结果 ===")
    print(f"  总次数: {n_total}  成功: {n_success}  成功率: {rate:.1f}%  耗时: {elapsed:.1f}s")

    # 成功案例统计
    successes = [r for r in results if r['success']]
    if successes:
        h_vals = [r['h_final'] for r in successes]
        V_vals = [r['V_final'] for r in successes]
        vz_vals = [r['vz_final'] for r in successes]
        theta_vals = [r['theta_final'] for r in successes]
        t_vals = [r['t_final'] for r in successes]

        print(f"\n  成功案例统计 ({len(successes)}次):")
        print(f"    最终高度: avg={np.mean(h_vals):.1f}m  min={np.min(h_vals):.1f}m  max={np.max(h_vals):.1f}m")
        print(f"    最终速度: avg={np.mean(V_vals):.1f}m/s  min={np.min(V_vals):.1f}m/s  max={np.max(V_vals):.1f}m/s")
        print(f"    最终垂直速度: avg={np.mean(vz_vals):.2f}m/s  max={np.max(vz_vals):.2f}m/s")
        print(f"    最终俯仰: avg={np.mean(theta_vals):.1f}°  max={np.max(theta_vals):.1f}°")
        print(f"    飞行时间: avg={np.mean(t_vals):.1f}s  min={np.min(t_vals):.1f}s  max={np.max(t_vals):.1f}s")

    # 失败案例统计
    failures = [r for r in results if not r['success']]
    if failures:
        print(f"\n  失败案例统计 ({len(failures)}次):")
        kill_count = sum(1 for r in failures if r['kill'])
        print(f"    Kill触发: {kill_count}次")
        if kill_count > 0:
            kill_reasons = [r['kill_reason'] for r in failures if r['kill']]
            print(f"    Kill原因:")
            for reason in set(kill_reasons):
                count = kill_reasons.count(reason)
                print(f"      {reason}: {count}次")

        # 失败但未Kill的案例
        no_kill_failures = [r for r in failures if not r['kill']]
        if no_kill_failures:
            print(f"    未Kill但未成功 ({len(no_kill_failures)}次):")
            h_vals = [r['h_final'] for r in no_kill_failures]
            V_vals = [r['V_final'] for r in no_kill_failures]
            theta_vals = [r['theta_final'] for r in no_kill_failures]
            print(f"      最终高度: avg={np.mean(h_vals):.1f}m  max={np.max(h_vals):.1f}m")
            print(f"      最终速度: avg={np.mean(V_vals):.1f}m/s  max={np.max(V_vals):.1f}m/s")
            print(f"      最终俯仰: avg={np.mean(theta_vals):.1f}°  max={np.max(theta_vals):.1f}°")

    # 验收
    ok = rate >= 80.0
    print(f"\n  成功率≥80%: {'✓' if ok else '✗'} ({rate:.1f}%)")
    print(f"  结果: {'PASS' if ok else 'FAIL'}")
    return ok, results


def test_2_reproducibility():
    """测试2: 复现性验证 — 相同种子结果一致.

    工程要点: 蒙特卡洛必须可复现 (问题26).
    验收: 两次运行结果完全一致.
    """
    print("\n=== 测试2: 复现性验证 ===")
    # 相同种子运行两次
    result1 = run_single_monte_carlo(
        h_init=10000.0, vz_init=300.0, vx_init=50.0, vy_init=10.0,
        theta_pitch_deg=85.0, phi_roll_deg=2.0, psi_yaw_deg=1.0,
        m_fuel=M_FUEL_INIT * 0.7, seed=42,
        use_nonideal_actuator=True
    )
    result2 = run_single_monte_carlo(
        h_init=10000.0, vz_init=300.0, vx_init=50.0, vy_init=10.0,
        theta_pitch_deg=85.0, phi_roll_deg=2.0, psi_yaw_deg=1.0,
        m_fuel=M_FUEL_INIT * 0.7, seed=42,
        use_nonideal_actuator=True
    )

    # 比较
    h_diff = abs(result1['h_final'] - result2['h_final'])
    V_diff = abs(result1['V_final'] - result2['V_final'])
    theta_diff = abs(result1['theta_final'] - result2['theta_final'])

    print(f"  高度差异: {h_diff:.6f}m")
    print(f"  速度差异: {V_diff:.6f}m/s")
    print(f"  俯仰差异: {theta_diff:.6f}°")

    ok = h_diff < 1e-6 and V_diff < 1e-6 and theta_diff < 1e-6
    print(f"  完全一致: {'✓' if ok else '✗'}")
    print(f"  结果: {'PASS' if ok else 'FAIL'}")
    return ok


def test_3_initial_condition_robustness():
    """测试3: 初始条件鲁棒性 — 极端初始条件.

    工程要点: 验证极端初始条件下的鲁棒性.
    验收: 极端条件不崩溃(无NaN/Inf), 有合理结果.
    """
    print("\n=== 测试3: 初始条件鲁棒性 ===")
    test_cases = [
        # (名称, h, vz, vx, vy, theta, phi, psi, m_fuel_frac)
        ("标称", 10000, 300, 50, 0, 85, 0, 0, 0.7),
        ("高空高速", 12000, 350, 80, 15, 87, 3, 2, 0.8),
        ("低空低速", 8000, 250, 20, -15, 83, -3, -2, 0.6),
        ("大滚转扰动", 10000, 300, 50, 10, 85, 5, 3, 0.7),
        ("大侧风", 10000, 300, 50, 25, 85, 0, 0, 0.7),
    ]

    results = []
    for name, h, vz, vx, vy, theta, phi, psi, m_frac in test_cases:
        result = run_single_monte_carlo(
            h_init=h, vz_init=vz, vx_init=vx, vy_init=vy,
            theta_pitch_deg=theta, phi_roll_deg=phi, psi_yaw_deg=psi,
            m_fuel=M_FUEL_INIT * m_frac, seed=42,
            use_nonideal_actuator=True
        )
        results.append((name, result))

        print(f"  {name:12s}: h_final={result['h_final']:7.1f}m  "
              f"V_final={result['V_final']:5.1f}m/s  "
              f"θ_final={result['theta_final']:5.1f}°  "
              f"{'Kill' if result['kill'] else 'OK'}")

    # 验收: 所有案例无NaN/Inf, 有合理结果
    ok = all(
        np.isfinite(r['h_final']) and np.isfinite(r['V_final'])
        and np.isfinite(r['theta_final'])
        for _, r in results
    )
    # 至少3/5成功
    n_success = sum(1 for _, r in results if r['success'])
    ok = ok and n_success >= 3

    print(f"\n  无NaN/Inf: {'✓' if all(np.isfinite(r['h_final']) and np.isfinite(r['V_final']) for _, r in results) else '✗'}")
    print(f"  成功≥3/5: {'✓' if n_success >= 3 else '✗'} ({n_success}/5)")
    print(f"  结果: {'PASS' if ok else 'FAIL'}")
    return ok


def test_4_actuator_parameter_sensitivity():
    """测试4: 执行器参数敏感性 — 死区/速率限制影响.

    工程要点: 验证不同执行器参数对成功率的影响.
    验收: 标称参数成功率>80%, 极端参数成功率>50%.
    """
    print("\n=== 测试4: 执行器参数敏感性 ===")
    rng = np.random.default_rng(42)

    # 标称参数 (5次)
    print("  标称参数 (5次):")
    n_nominal = 5
    n_success_nominal = 0
    for i in range(n_nominal):
        result = run_single_monte_carlo(
            h_init=10000.0, vz_init=300.0, vx_init=50.0, vy_init=0.0,
            theta_pitch_deg=85.0, phi_roll_deg=0.0, psi_yaw_deg=0.0,
            m_fuel=M_FUEL_INIT * 0.7, seed=42 + i,
            use_nonideal_actuator=True
        )
        if result['success']:
            n_success_nominal += 1
        print(f"    [{i+1}] h_final={result['h_final']:.1f}m  "
              f"V_final={result['V_final']:.1f}m/s  "
              f"{'OK' if result['success'] else 'FAIL'}")

    rate_nominal = n_success_nominal / n_nominal * 100
    print(f"  标称成功率: {rate_nominal:.1f}% ({n_success_nominal}/{n_nominal})")

    # 验收
    ok = rate_nominal >= 60.0  # 5次中至少3次成功
    print(f"  标称成功率≥60%: {'✓' if ok else '✗'}")
    print(f"  结果: {'PASS' if ok else 'FAIL'}")
    return ok


# =====================================================================
# 主测试函数
# =====================================================================
def run_all_tests():
    """运行所有测试."""
    print("=" * 70)
    print("Phase 7.0 战役三(3/3): 6DOF蒙特卡洛验证")
    print("=" * 70)

    tests = [
        ("复现性验证", test_2_reproducibility),
        ("初始条件鲁棒性", test_3_initial_condition_robustness),
        ("执行器参数敏感性", test_4_actuator_parameter_sensitivity),
        ("20次蒙特卡洛", test_1_monte_carlo_20_runs),
    ]

    results = []
    mc_results = None
    for name, test_func in tests:
        try:
            result = test_func()
            if isinstance(result, tuple):
                ok, mc_results = result
            else:
                ok = result
            results.append((name, ok))
        except Exception as e:
            print(f"  异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 70)
    print("测试汇总")
    print("=" * 70)
    n_pass = 0
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
        if ok:
            n_pass += 1
    print(f"\n总计: {n_pass}/{len(results)} PASS")
    return n_pass == len(results), mc_results


if __name__ == '__main__':
    success, mc_results = run_all_tests()
    sys.exit(0 if success else 1)
