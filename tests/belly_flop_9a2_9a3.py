"""
Phase 9.0 Step 9a: 偏置扫描 + Dither + 补偿 (缺陷28修复后)
==========================================================
缺陷28: Bouc-Wen(A=1.0)完全阻塞恒定信号 → 偏置必须绕过执行器
修复: 偏置在执行器输出后叠加 (机械预紧力模型)

偏置扫描:
  - F-1.25/A+1.25 (反向小偏置, 理论方案)
  - F-2.5/A+2.5   (反向中偏置)
  - F-3.5/A+3.5   (反向大偏置, 临界死区逃逸)
  - F+1.25/A+1.25 (同向偏置, 对照)

验收: Safe Landing > 30% (9a-1)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.belly_flop.phase9_controller import Phase9BellyFlopController
from src.belly_flop.dynamics import rk4_step
from src.belly_flop.aero_model import M_FUEL_INIT, atmosphere
from src.starship_non_ideal import NonIdealActuatorSuite
from src.starship_safety_hsm import StarshipSafetyHSM
from src.sensors import GPSBlackout, RadarMultipath

DT = 0.01
T_MAX = 120.0
N = int(T_MAX / DT)

H_INIT = 10000.0
VZ_INIT = 300.0
VX_INIT = 50.0
THETA_INIT = np.deg2rad(85.0)


def run_single(seed, bias_fwd_deg=0.0, bias_aft_deg=0.0,
               dither_enable=False, gain_comp_enable=False):
    """单次仿真. 偏置绕过执行器(缺陷28修复)."""
    rng = np.random.default_rng(seed)
    m_fuel_init = M_FUEL_INIT * 0.7 * (1.0 + rng.uniform(-0.05, 0.05))
    vz_init = VZ_INIT * (1.0 + rng.uniform(-0.10, 0.10))
    vx_init = VX_INIT * (1.0 + rng.uniform(-0.20, 0.20))

    state = np.array([0.0, H_INIT, vx_init, vz_init, THETA_INIT, 0.0, m_fuel_init])

    controller = Phase9BellyFlopController(
        bias_enable=True,
        dither_enable=dither_enable,
        gain_comp_enable=gain_comp_enable,
        delta_fwd_bias_deg=bias_fwd_deg,
        delta_aft_bias_deg=bias_aft_deg,
        dither_freq=8.0, dither_amp_deg=1.0, gain_comp=1.1,
    )

    actuators = NonIdealActuatorSuite(rng=rng)
    actuators.randomize(rng)
    hsm = StarshipSafetyHSM()
    gps_blk = GPSBlackout()
    radar_mp = RadarMultipath(rng=rng)

    belly_theta_err = 0.0
    l3_trigger_phase = None
    outcome = 'UNKNOWN'

    for k in range(N):
        t = k * DT
        T_cmd, theta_cmd, d_fwd, d_aft, phase, kill, info = controller.update(state, DT)
        if kill:
            outcome = 'KILL'
            break

        T_act = actuators.update_thrust(T_cmd, DT, t)
        # 缺陷28修复: PD送入执行器, 偏置在执行器后叠加(绕过Bouc-Wen)
        d_fwd_act, d_aft_act = actuators.update_flaps(d_fwd, d_aft, DT)
        d_fwd_act += info.get('bias_fwd_out', 0.0)
        d_aft_act += info.get('bias_aft_out', 0.0)

        h = state[1]
        radar_jump = False
        if h < 100.0:
            _, radar_jump = radar_mp.measure(h, DT)
        V = np.sqrt(state[2]**2 + state[3]**2)
        rho, a_snd, _, _ = atmosphere(h)
        M = V / a_snd if a_snd > 0 else 0.0
        gps_valid, _ = gps_blk.check(M, h, t)

        hsm_r = hsm.update(state, d_fwd, d_aft, d_fwd_act, d_aft_act,
                           radar_jump, t, DT, phase=phase)
        if hsm_r['kill']:
            outcome = 'UNSAFE_CRASH'
            break
        if hsm_r['abort'] and l3_trigger_phase is None:
            l3_trigger_phase = phase
        if hsm_r['abort']:
            T_act = 0.0
            if hsm_r['landing_prediction'] is not None:
                if hsm_r['landing_prediction']['in_safe_zone']:
                    outcome = 'SAFE_ABORT'
                else:
                    outcome = 'UNSAFE_CRASH'
                break

        state = rk4_step(state, T_act, theta_cmd, DT, d_fwd_act, d_aft_act)
        if phase == 'BELLY':
            err = abs(np.rad2deg(state[4]) - 85.0)
            if err > belly_theta_err:
                belly_theta_err = err
        if state[1] <= 0:
            state[1] = 0.0
            vz_final = state[3]
            theta_final = abs(np.rad2deg(state[4]))
            if abs(vz_final) < 6.0 and theta_final < 15.0:
                outcome = 'SAFE_LANDING'
            else:
                outcome = 'UNSAFE_CRASH'
            break

    return {
        'outcome': outcome, 'belly_theta_err': belly_theta_err,
        'l3_trigger_phase': l3_trigger_phase,
        'final_h': state[1], 'final_vz': state[3],
        'final_theta': np.rad2deg(state[4]),
        'safety_level': int(hsm.level),
    }


def run_mc(n_mc, **kwargs):
    """MC测试."""
    outcomes = {'SAFE_LANDING': 0, 'SAFE_ABORT': 0, 'UNSAFE_CRASH': 0, 'KILL': 0, 'UNKNOWN': 0}
    l3_phases = {'BELLY': 0, 'FLIP': 0, 'LANDING': 0, None: 0}
    belly_errs = []
    t0 = time.time()
    for i in range(n_mc):
        r = run_single(seed=i + 100, **kwargs)
        outcomes[r['outcome']] += 1
        l3_phases[r['l3_trigger_phase']] += 1
        belly_errs.append(r['belly_theta_err'])
    elapsed = time.time() - t0
    landing_rate = outcomes['SAFE_LANDING'] / n_mc
    return landing_rate, outcomes, l3_phases, belly_errs, elapsed


def print_result(label, n_mc, landing_rate, outcomes, l3_phases, belly_errs, elapsed):
    print(f"\n  [{label}] {n_mc}次MC ({elapsed:.1f}s)")
    print(f"    Safe Landing: {outcomes['SAFE_LANDING']:3d}/{n_mc} = {landing_rate*100:.0f}%")
    print(f"    Safe Abort:   {outcomes['SAFE_ABORT']:3d}/{n_mc} = {outcomes['SAFE_ABORT']/n_mc*100:.0f}%")
    print(f"    Unsafe Crash: {outcomes['UNSAFE_CRASH']:3d}/{n_mc} = {outcomes['UNSAFE_CRASH']/n_mc*100:.0f}%")
    print(f"    L3触发: BELLY={l3_phases['BELLY']}, FLIP={l3_phases['FLIP']}, LANDING={l3_phases['LANDING']}")
    print(f"    BELLY err: mean={np.mean(belly_errs):.1f}°, max={np.max(belly_errs):.1f}°")


if __name__ == '__main__':
    N_MC = 20
    print("=" * 70)
    print("Phase 9.0 Step 9a: 偏置扫描 (缺陷28修复: 偏置绕过执行器)")
    print("=" * 70)

    configs = [
        ('无偏置',         0.0, 0.0, False, False),
        ('F-1.25/A+1.25', -1.25, 1.25, False, False),
        ('F-2.5/A+2.5',   -2.5, 2.5, False, False),
        ('F-3.5/A+3.5',   -3.5, 3.5, False, False),
        ('F+1.25/A+1.25',  1.25, 1.25, False, False),
    ]

    results = []
    for label, bf, ba, di, gc in configs:
        r = run_mc(N_MC, bias_fwd_deg=bf, bias_aft_deg=ba,
                   dither_enable=di, gain_comp_enable=gc)
        print_result(label, N_MC, *r)
        results.append((label, r[0]))

    # 找最优偏置
    best = max(results, key=lambda x: x[1])
    print(f"\n  >>> 最优偏置: {best[0]} (Landing={best[1]*100:.0f}%)")

    # 对最优偏置测试Dither和补偿
    best_label, best_bf, best_ba = best[0], None, None
    for label, bf, ba, _, _ in configs:
        if label == best[0]:
            best_bf, best_ba = bf, ba
            break

    print(f"\n  --- 对最优偏置({best_label})叠加Dither和补偿 ---")
    r2 = run_mc(N_MC, bias_fwd_deg=best_bf, bias_aft_deg=best_ba,
                dither_enable=True, gain_comp_enable=False)
    print_result(f'{best_label}+Dither', N_MC, *r2)
    r3 = run_mc(N_MC, bias_fwd_deg=best_bf, bias_aft_deg=best_ba,
                dither_enable=True, gain_comp_enable=True)
    print_result(f'{best_label}+Dither+补偿', N_MC, *r3)

    print(f"\n  最终最优: Landing={max(r2[0], r3[0], best[1])*100:.0f}%")
