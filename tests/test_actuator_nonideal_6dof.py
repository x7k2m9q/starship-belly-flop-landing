"""
Phase 7.0 战役二: 4片独立非理想执行器验证 (问题20-22)
=====================================================
验收标准 (理论方案7.0):
  问题20: 速率限制 30°/s — 阶跃响应不超速率
  问题21: Bouc-Wen标准公式 — 阶跃稳态误差<0.5°
  问题22: 死区补偿 — PD指令0.3°时实际偏转>0.5° (越过死区)

测试内容:
  1. Bouc-Wen阶跃响应: 稳态误差<0.5°
  2. 死区补偿: 小指令0.3° → 实际偏转>0.5°
  3. 速率限制: 30°/s不超速
  4. 4片独立性: 不同参数不互相影响
  5. 正弦扫频: 无异常阻塞或相位突变
  6. Dither幅值>死区最大值
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.belly_flop.actuator_nonideal_6dof import FlapActuatorSuite6DOF, TVCActuator6DOF


def test_1_bouc_wen_step_response():
    """测试1: Bouc-Wen阶跃响应 — 稳态误差<0.5° (问题21)."""
    print("\n=== 测试1: Bouc-Wen阶跃响应 ===")
    # 不启用死区补偿, 纯Bouc-Wen
    actuator = FlapActuatorSuite6DOF(use_compensation=False)
    dt = 0.01

    # 阶跃指令: 5°
    cmd = np.deg2rad(5.0)
    delta_cmds = [cmd, cmd, cmd, cmd]

    # 仿真2s
    n_steps = int(2.0 / dt)
    delta_history = []
    for i in range(n_steps):
        delta_actual = actuator.update(delta_cmds, dt)
        delta_history.append(delta_actual.copy())

    delta_history = np.array(delta_history)
    # 稳态值(最后10个采样的平均)
    delta_steady = np.mean(delta_history[-10:], axis=0)
    steady_err = np.abs(delta_steady - cmd) * 180.0 / np.pi  # 转为度

    print(f"  指令: {np.rad2deg(cmd):.1f}°")
    print(f"  稳态实际: {np.rad2deg(delta_steady)}")
    print(f"  稳态误差: {steady_err}°")
    print(f"  验收<0.5°: {'✓' if np.all(steady_err < 0.5) else '✗'}")

    ok = np.all(steady_err < 0.5)
    print(f"  结果: {'PASS' if ok else 'FAIL'}")
    return ok


def test_2_deadzone_compensation():
    """测试2: 死区补偿 — PD指令0.3°时实际偏转>0.5° (问题22)."""
    print("\n=== 测试2: 死区补偿 ===")
    dt = 0.01

    # 无补偿: 0.3°指令应被死区吞掉
    actuator_no_comp = FlapActuatorSuite6DOF(
        dead_zones=[np.deg2rad(0.5)] * 4,
        use_compensation=False
    )
    cmd = np.deg2rad(0.3)  # 小于死区0.5°
    delta_cmds = [cmd, cmd, cmd, cmd]

    # 仿真1s
    for i in range(int(1.0 / dt)):
        delta_no_comp = actuator_no_comp.update(delta_cmds, dt)

    # 有补偿: 0.3°指令应越过死区
    actuator_with_comp = FlapActuatorSuite6DOF(
        dead_zones=[np.deg2rad(0.5)] * 4,
        use_compensation=True,
        dither_freq=20.0,
        dither_amp=np.deg2rad(1.0)
    )
    delta_with_comp_history = []
    for i in range(int(1.0 / dt)):
        delta_with_comp = actuator_with_comp.update(delta_cmds, dt)
        delta_with_comp_history.append(delta_with_comp.copy())

    delta_with_comp_history = np.array(delta_with_comp_history)
    # 有补偿时, 平均偏转应>0.5° (Dither使执行器越过死区)
    delta_avg_comp = np.mean(np.abs(delta_with_comp_history[-50:]), axis=0)

    print(f"  死区: 0.5°")
    print(f"  指令: 0.3° (小于死区)")
    print(f"  无补偿最终偏转: {np.rad2deg(np.abs(delta_no_comp))}°")
    print(f"  有补偿平均偏转: {np.rad2deg(delta_avg_comp)}°")
    print(f"  验收有补偿>0.5°: {'✓' if np.all(delta_avg_comp > np.deg2rad(0.5)) else '✗'}")

    ok = np.all(delta_avg_comp > np.deg2rad(0.5))
    print(f"  结果: {'PASS' if ok else 'FAIL'}")
    return ok


def test_3_rate_limit():
    """测试3: 速率限制 — 30°/s不超速 (问题20).

    工程要点: 关闭Dither补偿, 纯测速率限制.
    原因: Dither是20Hz高频信号, 幅值1°, 每步变化~95°/s,
    不受速率限制约束(速率限制只限低频指令跟踪).
    """
    print("\n=== 测试3: 速率限制 ===")
    # 无死区无补偿, 纯测速率限制
    actuator = FlapActuatorSuite6DOF(
        dead_zones=[0.0] * 4,
        rate_limits=[np.deg2rad(30.0)] * 4,
        use_compensation=False
    )
    dt = 0.01

    # 阶跃指令: 0° → 15° (大阶跃, 触发速率限制)
    cmd_initial = [0.0, 0.0, 0.0, 0.0]
    cmd_step = [np.deg2rad(15.0)] * 4

    # 先稳定在0
    for i in range(int(0.5 / dt)):
        actuator.update(cmd_initial, dt)

    # 阶跃
    delta_history = []
    for i in range(int(1.0 / dt)):
        delta_actual = actuator.update(cmd_step, dt)
        delta_history.append(delta_actual.copy())

    delta_history = np.array(delta_history)

    # 计算实际速率 (°/s)
    rates = np.diff(delta_history, axis=0) / dt * 180.0 / np.pi
    max_rate = np.max(np.abs(rates))

    # 理论: 30°/s, dt=0.01 → 每步最多0.3°
    print(f"  速率限制: 30°/s")
    print(f"  阶跃: 0° → 15°")
    print(f"  最大实际速率: {max_rate:.2f}°/s")
    print(f"  验收≤30.5°/s: {'✓' if max_rate < 30.5 else '✗'}")

    ok = max_rate < 30.5
    print(f"  结果: {'PASS' if ok else 'FAIL'}")
    return ok


def test_4_independence():
    """测试4: 4片独立性 — 不同参数不互相影响."""
    print("\n=== 测试4: 4片独立性 ===")
    # 4片不同死区
    dead_zones = [np.deg2rad(0.3), np.deg2rad(0.5), np.deg2rad(0.7), np.deg2rad(0.8)]
    actuator = FlapActuatorSuite6DOF(
        dead_zones=dead_zones,
        use_compensation=False
    )
    dt = 0.01

    # 不同指令
    cmds = [np.deg2rad(2.0), np.deg2rad(4.0), np.deg2rad(6.0), np.deg2rad(8.0)]

    # 仿真1s
    delta_history = []
    for i in range(int(1.0 / dt)):
        delta_actual = actuator.update(cmds, dt)
        delta_history.append(delta_actual.copy())

    delta_history = np.array(delta_history)
    delta_steady = np.mean(delta_history[-10:], axis=0)

    # 检查每片是否独立响应
    print(f"  死区: {[f'{np.rad2deg(dz):.1f}°' for dz in dead_zones]}")
    print(f"  指令: {[f'{np.rad2deg(c):.1f}°' for c in cmds]}")
    print(f"  稳态: {[f'{np.rad2deg(d):.2f}°' for d in delta_steady]}")

    # 每片应接近其指令(大指令不受死区影响)
    errors = np.abs(delta_steady - np.array(cmds)) * 180.0 / np.pi
    ok = np.all(errors < 1.0)
    print(f"  稳态误差: {[f'{e:.2f}°' for e in errors]}")
    print(f"  验收<1°: {'✓' if ok else '✗'}")
    print(f"  结果: {'PASS' if ok else 'FAIL'}")
    return ok


def test_5_sine_sweep():
    """测试5: 正弦扫频 — 无异常阻塞或相位突变 (7.0.txt第346行)."""
    print("\n=== 测试5: 正弦扫频 ===")
    actuator = FlapActuatorSuite6DOF(use_compensation=True)
    dt = 0.01

    # 扫频: 0.1Hz → 5Hz, 10s
    t_total = 10.0
    n_steps = int(t_total / dt)
    t_arr = np.arange(n_steps) * dt

    # 对数扫频
    freqs = np.logspace(np.log10(0.1), np.log10(5.0), n_steps)
    phases = 2 * np.pi * np.cumsum(freqs) * dt
    cmd_signal = np.deg2rad(3.0) * np.sin(phases)

    delta_history = []
    cmd_history = []
    for i in range(n_steps):
        cmd = cmd_signal[i]
        delta_actual = actuator.update([cmd, cmd, cmd, cmd], dt)
        delta_history.append(delta_actual[0])
        cmd_history.append(cmd)

    delta_history = np.array(delta_history)
    cmd_history = np.array(cmd_history)

    # 检查: 无异常阻塞(输出不应恒为0)
    # 检查: 无相位突变(输出与输入的相关性应保持正值)
    nonzero_ratio = np.sum(np.abs(delta_history) > np.deg2rad(0.1)) / len(delta_history)
    correlation = np.corrcoef(cmd_history, delta_history)[0, 1]

    print(f"  扫频: 0.1Hz → 5Hz, 10s")
    print(f"  非零输出比例: {nonzero_ratio*100:.1f}%")
    print(f"  输入-输出相关系数: {correlation:.3f}")
    print(f"  验收非零>80%: {'✓' if nonzero_ratio > 0.8 else '✗'}")
    print(f"  验收相关>0.5: {'✓' if correlation > 0.5 else '✗'}")

    ok = nonzero_ratio > 0.8 and correlation > 0.5
    print(f"  结果: {'PASS' if ok else 'FAIL'}")
    return ok


def test_6_dither_amplitude():
    """测试6: Dither幅值>死区最大值 (问题22铁律)."""
    print("\n=== 测试6: Dither幅值>死区最大值 ===")
    # 测试随机化后Dither幅值仍>死区最大值
    rng = np.random.default_rng(42)
    actuator = FlapActuatorSuite6DOF(use_compensation=True)
    actuator.randomize(rng)

    max_deadzone = max(actuator.dead_zones)
    dither_amp = actuator.dither_amp

    print(f"  随机化后最大死区: {np.rad2deg(max_deadzone):.2f}°")
    print(f"  Dither幅值: {np.rad2deg(dither_amp):.2f}°")
    print(f"  验收Dither>死区: {'✓' if dither_amp > max_deadzone else '✗'}")

    ok = dither_amp > max_deadzone
    print(f"  结果: {'PASS' if ok else 'FAIL'}")
    return ok


def test_7_tvc_delay():
    """测试7: TVC延迟 — 80ms纯延迟."""
    print("\n=== 测试7: TVC延迟 ===")
    tvc = TVCActuator6DOF(delay=0.08)
    dt = 0.01

    # 阶跃指令
    cmd = np.deg2rad(5.0)
    gimbal_cmds = [cmd, 0.0]

    # 仿真0.2s
    t_arr = []
    actual_arr = []
    for i in range(int(0.2 / dt)):
        t = i * dt
        gimbal_actual = tvc.update(gimbal_cmds, dt)
        t_arr.append(t)
        actual_arr.append(gimbal_actual[0])

    t_arr = np.array(t_arr)
    actual_arr = np.array(actual_arr)

    # 找到首次输出>0.5°的时刻
    threshold = np.deg2rad(0.5)
    first_output_idx = None
    for i in range(len(actual_arr)):
        if abs(actual_arr[i]) > threshold:
            first_output_idx = i
            break

    if first_output_idx is not None:
        delay_measured = t_arr[first_output_idx]
        print(f"  指令: {np.rad2deg(cmd):.1f}°")
        print(f"  首次输出>0.5°时刻: {delay_measured:.3f}s")
        print(f"  预期延迟: 0.08s")
        # 允许±20ms误差
        ok = abs(delay_measured - 0.08) < 0.02
    else:
        print(f"  无输出!")
        ok = False

    print(f"  验收延迟≈0.08±0.02s: {'✓' if ok else '✗'}")
    print(f"  结果: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == '__main__':
    print("=" * 60)
    print("Phase 7.0 战役二: 4片独立非理想执行器验证")
    print("=" * 60)

    results = []
    results.append(('Bouc-Wen阶跃', test_1_bouc_wen_step_response()))
    results.append(('死区补偿', test_2_deadzone_compensation()))
    results.append(('速率限制', test_3_rate_limit()))
    results.append(('4片独立性', test_4_independence()))
    results.append(('正弦扫频', test_5_sine_sweep()))
    results.append(('Dither幅值', test_6_dither_amplitude()))
    results.append(('TVC延迟', test_7_tvc_delay()))

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
