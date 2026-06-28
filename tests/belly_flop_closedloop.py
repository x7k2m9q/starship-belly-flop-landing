"""
Belly-Flop Step 7B 闭环验证.
==============================
完整再入剖面: BELLY(10km) → FLIP → LANDING → 着陆

验证项:
  1. 三阶段切换正确 (belly→flip→landing)
  2. 不触发Kill (翻转后h>800m, 能量检查通过)
  3. PD主动阻尼解决暗礁3 (θ跟踪有界, 不发散)
  4. 着陆成功 (vz<10m/s, |vx|<5m/s, |theta|<15°)

初始条件:
  h=10km, vx=50m/s, vz=300m/s, θ=85°, q=0, m_fuel=50000kg
  (亚声速再入, V≈304m/s, Mach~0.9@10km)
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.belly_flop.dynamics import simulate_closed_loop
from src.belly_flop.controller import BellyFlopController
from src.belly_flop.aero_model import (
    S_REF, L_REF, M_DRY, M_FUEL_INIT, M_TOTAL_INIT,
    get_Iyy, get_mass, T_MAX, T_IDLE,
)

DT = 0.01
T_END = 120.0  # 最大仿真时长
RECORD_INTERVAL = 10

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def run_closed_loop_test():
    """完整再入闭环测试."""
    print("=" * 70)
    print("Belly-Flop Step 7B 闭环验证")
    print("完整再入: BELLY → FLIP → LANDING → 着陆")
    print("=" * 70)

    # 初始状态 (高速再入, 满足belly→flip触发条件)
    initial_state = [
        0.0,                    # x
        10000.0,                # h
        0.0,                    # vx
        500.0,                  # vz (高速再入)
        np.deg2rad(85.0),       # θ (腹部朝下)
        0.0,                    # q
        M_FUEL_INIT,            # m_fuel
    ]

    print(f"\n初始状态:")
    print(f"  h=10km, vx=0m/s, vz=500m/s, θ=85°, q=0")
    print(f"  V=500.0m/s, m_fuel={M_FUEL_INIT}kg")
    print(f"  dt={DT}s, t_end={T_END}s")

    # 控制器
    controller = BellyFlopController()

    # 闭环仿真
    result = simulate_closed_loop(initial_state, controller, DT, T_END,
                                  record_interval=RECORD_INTERVAL, h_ground=0.0)

    # 分析结果
    t = result['t']
    h = result['h']
    vx = result['vx']
    vz = result['vz']
    theta = result['theta']
    q = result['q']
    T = result['T']
    theta_cmd = result['theta_cmd']
    phases = result['phase']
    V = np.sqrt(vx ** 2 + vz ** 2)

    print(f"\n仿真结果:")
    print(f"  仿真时长: {t[-1]:.1f}s")
    print(f"  Kill: {result['kill']}")
    if result['kill']:
        print(f"  Kill原因: {result['kill_reason']}")
    print(f"  着陆成功: {result['landing_success']}")

    # 阶段切换分析
    phase_changes = []
    prev_phase = phases[0]
    for i, ph in enumerate(phases):
        if ph != prev_phase:
            phase_changes.append((t[i], prev_phase, ph, h[i], V[i], np.degrees(theta[i])))
            prev_phase = ph

    print(f"\n阶段切换:")
    for tc, p1, p2, h_tc, v_tc, th_tc in phase_changes:
        print(f"  t={tc:.1f}s: {p1}→{p2}, h={h_tc:.0f}m, V={v_tc:.1f}m/s, θ={th_tc:.1f}°")

    # 最终状态
    print(f"\n最终状态:")
    print(f"  h={h[-1]:.1f}m, vx={vx[-1]:.1f}m/s, vz={vz[-1]:.1f}m/s")
    print(f"  θ={np.degrees(theta[-1]):.1f}°, V={V[-1]:.1f}m/s")
    print(f"  m_fuel={result['m_fuel'][-1]:.0f}kg (消耗={M_FUEL_INIT-result['m_fuel'][-1]:.0f}kg)")

    # 验收
    print(f"\n验收:")
    if result['kill']:
        print(f"  FAILED: {result['kill_reason']}")
        status = 'FAILED'
    elif result['landing_success']:
        print(f"  PASS: 软着陆成功")
        status = 'PASS'
    elif h[-1] > 0:
        print(f"  MARGINAL: 未触地 (h={h[-1]:.0f}m)")
        status = 'MARGINAL'
    else:
        print(f"  FAILED: 硬着陆 (vz={vz[-1]:.1f}, vx={vx[-1]:.1f}, θ={np.degrees(theta[-1]):.1f}°)")
        status = 'FAILED'

    # 暗礁检查
    print(f"\n暗礁检查:")
    # 暗礁7: 阶段切换姿态阶跃
    if len(phase_changes) > 0:
        max_theta_jump = 0
        for i in range(1, len(theta_cmd)):
            jump = abs(theta_cmd[i] - theta_cmd[i-1])
            if jump > max_theta_jump:
                max_theta_jump = jump
        print(f"  暗礁7 (阶段切换阶跃): max|Δθ_cmd/step| = {np.degrees(max_theta_jump):.3f}° "
              f"({'PASS' if np.degrees(max_theta_jump) < 5.0 else 'CHECK'})")

    # 暗礁3: PD阻尼效果 (θ跟踪误差)
    theta_err = theta_cmd - theta
    theta_err_deg = np.degrees(theta_err)
    # 归一化
    theta_err_deg = np.array([(e + 180) % 360 - 180 for e in theta_err_deg])
    max_err = np.max(np.abs(theta_err_deg))
    print(f"  暗礁3 (PD阻尼): max|θ_cmd-θ| = {max_err:.2f}° "
          f"({'PASS' if max_err < 15.0 else 'CHECK'})")

    # 暗礁9: 能量检查
    if len(phase_changes) >= 1:
        # 找flip→landing切换点
        for tc, p1, p2, h_tc, v_tc, th_tc in phase_changes:
            if p2 == 'LANDING':
                m_tc = get_mass(M_FUEL_INIT)  # 近似
                a_needed = v_tc ** 2 / (2 * h_tc)
                a_avail = T_MAX / m_tc - 9.81
                ratio = a_needed / a_avail
                print(f"  暗礁9 (能量检查@landing): a_needed={a_needed:.1f}, "
                      f"a_avail={a_avail:.1f}, ratio={ratio:.2f} "
                      f"({'PASS' if ratio < 0.7 else 'KILL'})")

    return result, status


def plot_closed_loop(result, save_dir='phase7b_plots'):
    """画6张图: (a)高度 (b)速度 (c)θ/θ_cmd (d)推力 (e)攻角 (f)阶段."""
    os.makedirs(save_dir, exist_ok=True)

    t = result['t']
    h = result['h']
    vx = result['vx']
    vz = result['vz']
    theta = result['theta']
    theta_cmd = result['theta_cmd']
    T = result['T']
    phases = result['phase']
    V = np.sqrt(vx ** 2 + vz ** 2)

    alpha_deg = np.array([np.degrees(a['alpha']) for a in result['aero']])
    Mach = np.array([a['Mach'] for a in result['aero']])
    Q_kPa = np.array([a['Q'] / 1000.0 for a in result['aero']])

    # 阶段着色
    phase_colors = {'BELLY': 'green', 'FLIP': 'orange', 'LANDING': 'blue'}
    phase_arr = np.array(phases)

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('Belly-Flop Step 7B 闭环验证', fontsize=14, fontweight='bold')

    # (a) 高度-时间
    ax = axes[0, 0]
    ax.plot(t, h / 1000.0, 'b-', linewidth=1.5)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('高度 (km)')
    ax.set_title('(a) 高度-时间')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=3.0, color='orange', linestyle='--', alpha=0.5, label='h=3km (flip触发)')
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='h=800m (Kill)')
    ax.legend(fontsize=8)

    # (b) 速度-时间
    ax = axes[0, 1]
    ax.plot(t, V, 'r-', linewidth=1.5, label='V')
    ax.plot(t, vz, 'r--', linewidth=1, alpha=0.5, label='vz')
    ax.plot(t, vx, 'g--', linewidth=1, alpha=0.5, label='vx')
    ax.axhline(y=250, color='orange', linestyle='--', alpha=0.5, label='V=250 (flip触发)')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('速度 (m/s)')
    ax.set_title('(b) 速度-时间')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # (c) θ/θ_cmd-时间
    ax = axes[1, 0]
    ax.plot(t, np.degrees(theta), 'b-', linewidth=1.5, label='θ (实际)')
    ax.plot(t, np.degrees(theta_cmd), 'r--', linewidth=1.5, label='θ_cmd')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('俯仰角 (°)')
    ax.set_title('(c) θ/θ_cmd-时间 (PD跟踪)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # (d) 推力-时间
    ax = axes[1, 1]
    ax.plot(t, T / 1e3, 'k-', linewidth=1.5)
    ax.axhline(y=T_MAX / 1e3, color='r', linestyle='--', alpha=0.5, label='T_max')
    ax.axhline(y=T_IDLE / 1e3, color='g', linestyle='--', alpha=0.5, label='T_idle')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('推力 (kN)')
    ax.set_title('(d) 推力-时间')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # (e) 攻角-时间
    ax = axes[2, 0]
    ax.plot(t, alpha_deg, 'g-', linewidth=1.5)
    ax.axhline(y=85, color='orange', linestyle='--', alpha=0.5, label='α=85° (配平)')
    ax.axhline(y=10, color='r', linestyle='--', alpha=0.5, label='α=10° (landing触发)')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('攻角 (°)')
    ax.set_title('(e) 攻角-时间')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # (f) 阶段-时间 (着色背景)
    ax = axes[2, 1]
    ax.plot(t, Mach, 'm-', linewidth=1.5, label='Mach')
    # 阶段背景色
    for i in range(len(t) - 1):
        ph = phase_arr[i]
        color = phase_colors.get(ph, 'gray')
        ax.axvspan(t[i], t[i + 1], alpha=0.2, color=color)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('Mach数')
    ax.set_title('(f) Mach-时间 (背景色=阶段)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    fname = os.path.join(save_dir, 'closed_loop_full.png')
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {fname}")


def main():
    result, status = run_closed_loop_test()
    plot_closed_loop(result)

    print("\n" + "=" * 70)
    print(f"Step 7B 闭环验证完成: {status}")
    print("=" * 70)


if __name__ == '__main__':
    main()
