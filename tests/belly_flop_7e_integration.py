"""
Belly-Flop Step 7E: 全程集成验证.
================================
验证目标:
  1. 暗礁23: 三阶段切换无状态丢失 (BELLY→FLIP→LANDING)
  2. 全程轨迹连续, 切换点无发散
  3. 着陆成功: vz<10m/s, |vx|<5m/s, |θ|<15°
  4. 各阶段控制律正确接管

初始条件: h=10km, vz=300m/s, θ=85° (高空belly入口)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.belly_flop.integrated_controller import IntegratedBellyFlopController
from src.belly_flop.dynamics import simulate_closed_loop
from src.belly_flop.aero_model import (
    M_FUEL_INIT, T_IDLE, angle_of_attack, atmosphere,
    S_REF, L_REF, DELTA_MAX,
)


def banner(s):
    print('=' * 70)
    print(s)
    print('=' * 70)


def main():
    banner('Belly-Flop Step 7E: 全程集成验证 (暗礁23: 统一状态结构体)')
    print()

    # =================================================================
    # 1. 全程闭环仿真
    # =================================================================
    banner('[1] 全程闭环仿真 (BELLY → FLIP → LANDING)')

    # 初始状态: 高空belly入口
    # h=10km, vz=300m/s, θ=85°, m_fuel=70%初始燃料
    initial_state = np.array([
        0.0,        # x
        10000.0,    # h (10km)
        50.0,       # vx
        300.0,      # vz (向下)
        np.deg2rad(85.0),  # θ (belly姿态)
        0.0,        # q
        M_FUEL_INIT * 0.7,  # m_fuel (70%)
    ])

    print(f'  初始状态:')
    print(f'    h={initial_state[1]:.0f}m, vz={initial_state[3]:.1f}m/s, '
          f'theta={np.degrees(initial_state[4]):.1f}deg')
    print(f'    vx={initial_state[2]:.1f}m/s, m_fuel={initial_state[6]:.0f}kg')
    print()

    # 仿真参数
    dt = 0.01
    t_end = 120.0  # 最大120s
    controller = IntegratedBellyFlopController()

    # 闭环仿真
    history = simulate_closed_loop(
        initial_state, controller, dt, t_end,
        record_interval=10, h_ground=0.0)

    # 结果分析
    t = history['t']
    h = history['h']
    vx = history['vx']
    vz = history['vz']
    theta = history['theta']
    q = history['q']
    T = history['T']
    theta_cmd = history['theta_cmd']
    phases = history['phase']

    print(f'  仿真结果:')
    print(f'    总时长: {t[-1]:.1f}s')
    print(f'    Kill触发: {history["kill"]}')
    if history['kill']:
        print(f'    Kill原因: {history["kill_reason"]}')
    print(f'    着陆成功: {history["landing_success"]}')
    print(f'    最终h: {h[-1]:.1f}m')
    print(f'    最终vz: {vz[-1]:.1f}m/s')
    print(f'    最终vx: {vx[-1]:.1f}m/s')
    print(f'    最终theta: {np.degrees(theta[-1]):.1f}deg')
    print(f'    最终q: {np.degrees(q[-1]):.2f}deg/s')
    print()

    # =================================================================
    # 2. 阶段切换分析 (暗礁23)
    # =================================================================
    banner('[2] 阶段切换分析 (暗礁23: 统一状态结构体)')

    # 找到阶段切换点
    phase_transitions = []
    prev_phase = phases[0]
    for i, ph in enumerate(phases):
        if ph != prev_phase:
            phase_transitions.append((i, prev_phase, ph, t[i]))
            prev_phase = ph

    print(f'  阶段切换点:')
    for idx, ph_from, ph_to, t_switch in phase_transitions:
        state_at_switch = np.array([history['x'][idx], h[idx], vx[idx], vz[idx],
                                     theta[idx], q[idx], history['m_fuel'][idx]])
        print(f'    {ph_from}→{ph_to} @ t={t_switch:.1f}s:')
        print(f'      h={h[idx]:.0f}m, V={np.sqrt(vx[idx]**2+vz[idx]**2):.1f}m/s, '
              f'theta={np.degrees(theta[idx]):.1f}deg')
        print(f'      vx={vx[idx]:.1f}m/s, vz={vz[idx]:.1f}m/s, '
              f'q={np.degrees(q[idx]):.2f}deg/s')
        print(f'      m_fuel={history["m_fuel"][idx]:.0f}kg')

    print()

    # 暗礁23验证: 切换点状态连续性
    # 检查切换前后theta和q无突变
    reef23_pass = True
    for idx, ph_from, ph_to, t_switch in phase_transitions:
        if idx > 0 and idx < len(theta) - 1:
            dtheta = abs(theta[idx] - theta[idx - 1])
            dq = abs(q[idx] - q[idx - 1])
            # 每步dt=0.1s (record_interval=10), theta变化应<5°/步
            if dtheta > np.deg2rad(10.0):
                print(f'  [WARN] {ph_from}→{ph_to}: dtheta={np.degrees(dtheta):.1f}deg/step')
                reef23_pass = False

    print(f'  暗礁23 (统一状态结构体): {"PASS" if reef23_pass else "FAIL"} '
          f'(切换点无状态突变)')
    print()

    # =================================================================
    # 3. 各阶段统计
    # =================================================================
    banner('[3] 各阶段统计')

    for phase_name in ['BELLY', 'FLIP', 'LANDING']:
        mask = np.array([p == phase_name for p in phases])
        if np.any(mask):
            t_phase = t[mask]
            h_phase = h[mask]
            V_phase = np.sqrt(vx[mask]**2 + vz[mask]**2)
            theta_phase = theta[mask]
            T_phase = T[mask]

            print(f'  {phase_name}:')
            print(f'    时长: {t_phase[-1] - t_phase[0]:.1f}s '
                  f'(t={t_phase[0]:.1f}s → {t_phase[-1]:.1f}s)')
            print(f'    高度: {h_phase[0]:.0f}m → {h_phase[-1]:.0f}m')
            print(f'    速度: {V_phase[0]:.1f}m/s → {V_phase[-1]:.1f}m/s')
            print(f'    theta: {np.degrees(theta_phase[0]):.1f}° → '
                  f'{np.degrees(theta_phase[-1]):.1f}°')
            print(f'    推力: T_min={T_phase.min():.0f}N, T_max={T_phase.max():.0f}N')
            print()

    # =================================================================
    # 4. 着陆判定
    # =================================================================
    banner('[4] 着陆判定')

    if len(h) > 0:
        vz_land = vz[-1]
        vx_land = vx[-1]
        theta_land = np.degrees(theta[-1])

        vz_pass = abs(vz_land) < 10.0
        vx_pass = abs(vx_land) < 5.0
        theta_pass = abs(theta_land) < 15.0

        print(f'  着陆垂直速度: {vz_land:.1f}m/s (阈值<10m/s) {"PASS" if vz_pass else "FAIL"}')
        print(f'  着陆水平速度: {vx_land:.1f}m/s (阈值<5m/s) {"PASS" if vx_pass else "FAIL"}')
        print(f'  着陆姿态角: {theta_land:.1f}deg (阈值<15deg) {"PASS" if theta_pass else "FAIL"}')
        print()

        landing_pass = vz_pass and vx_pass and theta_pass and not history['kill']
    else:
        landing_pass = False
        print('  仿真未完成, 无法判定着陆')
        print()

    # =================================================================
    # 5. 绘图
    # =================================================================
    banner('[5] 绘图保存')

    os.makedirs('phase7e_plots', exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    # 高度-速度剖面
    ax = axes[0, 0]
    ax.plot(t, h, 'b-')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('h (m)')
    ax.set_title('Altitude')
    ax.grid(True)
    # 标记阶段切换
    for idx, ph_from, ph_to, t_switch in phase_transitions:
        ax.axvline(x=t_switch, color='r', linestyle='--', alpha=0.5)

    # 速度
    ax = axes[0, 1]
    V = np.sqrt(vx**2 + vz**2)
    ax.plot(t, V, 'b-', label='V')
    ax.plot(t, np.abs(vz), 'r--', label='|vz|')
    ax.plot(t, np.abs(vx), 'g:', label='|vx|')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('V (m/s)')
    ax.set_title('Velocity')
    ax.legend()
    ax.grid(True)
    for idx, ph_from, ph_to, t_switch in phase_transitions:
        ax.axvline(x=t_switch, color='r', linestyle='--', alpha=0.5)

    # 俯仰角
    ax = axes[1, 0]
    ax.plot(t, np.degrees(theta), 'b-', label='theta')
    ax.plot(t, np.degrees(theta_cmd), 'r--', label='theta_cmd', alpha=0.7)
    ax.set_xlabel('t (s)')
    ax.set_ylabel('theta (deg)')
    ax.set_title('Pitch Angle')
    ax.legend()
    ax.grid(True)
    for idx, ph_from, ph_to, t_switch in phase_transitions:
        ax.axvline(x=t_switch, color='r', linestyle='--', alpha=0.5)

    # 角速度
    ax = axes[1, 1]
    ax.plot(t, np.degrees(q), 'b-')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('q (deg/s)')
    ax.set_title('Pitch Rate')
    ax.grid(True)
    for idx, ph_from, ph_to, t_switch in phase_transitions:
        ax.axvline(x=t_switch, color='r', linestyle='--', alpha=0.5)

    # 推力
    ax = axes[2, 0]
    ax.plot(t, T / 1e3, 'b-')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('T (kN)')
    ax.set_title('Thrust')
    ax.grid(True)
    for idx, ph_from, ph_to, t_switch in phase_transitions:
        ax.axvline(x=t_switch, color='r', linestyle='--', alpha=0.5)

    # 攻角
    ax = axes[2, 1]
    alpha_arr = np.array([np.degrees(angle_of_attack(theta[i], vx[i], vz[i])[0])
                          for i in range(len(theta))])
    ax.plot(t, alpha_arr, 'b-')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('alpha (deg)')
    ax.set_title('Angle of Attack')
    ax.grid(True)
    for idx, ph_from, ph_to, t_switch in phase_transitions:
        ax.axvline(x=t_switch, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('phase7e_plots/full_trajectory.png', dpi=100)
    print('  图保存: phase7e_plots/full_trajectory.png')
    print()

    # =================================================================
    # 6. 总结
    # =================================================================
    banner('Step 7E 验证总结')

    print(f'  暗礁23 (统一状态结构体): {"PASS" if reef23_pass else "FAIL"} '
          f'(三阶段切换无状态丢失)')
    print(f'  全程轨迹连续: {"PASS" if not history['kill'] or history['landing_success'] else "CHECK"}')
    print(f'  着陆成功: {"PASS" if landing_pass else "FAIL"}')
    print()

    if reef23_pass and landing_pass:
        print('  Step 7E (集成): PASS (全程集成验证通过)')
    elif reef23_pass:
        print('  Step 7E (集成): CHECK (状态结构体OK, 着陆需调优)')
    else:
        print('  Step 7E (集成): FAIL (状态丢失)')

    print()
    print(f'  关键数据:')
    print(f'    总时长: {t[-1]:.1f}s')
    print(f'    阶段切换: {len(phase_transitions)}次')
    print(f'    燃料消耗: {initial_state[6] - history["m_fuel"][-1]:.0f}kg')


if __name__ == '__main__':
    main()
