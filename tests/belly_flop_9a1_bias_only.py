"""
Phase 9.0 Step 9a-1: 自适应偏置Trim验证
=========================================
理论方案 6.0 Step 9a-1.

核心目标: 验证前后翼反向偏置能否打破"暗礁27"(死区在配平点).

测试流程:
  1. 偏置方向扫描 (4种配置, 各跑1次, 选最优)
  2. 最优配置单次仿真 (验证theta有界)
  3. 最优配置10次MC (初步统计)

验收标准:
  - theta在30s内保持有界 (|theta-85°| < 10°)
  - Safe Landing > 0% (至少有1次成功)
  - 目标: Safe Landing > 30%
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


def run_single(seed, bias_fwd_deg, bias_aft_deg,
               dither_enable=False, gain_comp_enable=False,
               verbose=False):
    """单次仿真."""
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
    )

    actuators = NonIdealActuatorSuite(rng=rng)
    actuators.randomize(rng)
    # 固定死区为最坏工况 (0.8°)
    actuators.fwd_flap.dead_zone = np.deg2rad(0.8)
    actuators.aft_flap.dead_zone = np.deg2rad(0.8)

    hsm = StarshipSafetyHSM()
    gps_blk = GPSBlackout()
    radar_mp = RadarMultipath(rng=rng)

    theta_history = []
    max_theta_err = 0.0
    outcome = 'UNKNOWN'
    kill_reason = ''

    for k in range(N):
        t = k * DT

        T_cmd, theta_cmd, d_fwd, d_aft, phase, kill, info = controller.update(state, DT)
        if kill:
            outcome = 'KILL'
            kill_reason = info.get('kill_reason', '')
            break

        T_act = actuators.update_thrust(T_cmd, DT, t)
        d_fwd_act, d_aft_act = actuators.update_flaps(d_fwd, d_aft, DT)

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
            kill_reason = hsm_r.get('abort_reason', '')
            break

        if hsm_r['abort']:
            T_act = 0.0
            if hsm_r['landing_prediction'] is not None:
                if hsm_r['landing_prediction']['in_safe_zone']:
                    outcome = 'SAFE_ABORT'
                else:
                    outcome = 'UNSAFE_CRASH'
                break

        state = rk4_step(state, T_act, theta_cmd, DT, d_fwd_act, d_aft_act)

        theta_err = abs(np.rad2deg(state[4]) - 85.0)
        if phase == 'BELLY' and theta_err > max_theta_err:
            max_theta_err = theta_err

        if k % 100 == 0:
            theta_history.append((t, np.rad2deg(state[4]), phase, state[1], state[3]))

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
        'outcome': outcome,
        'kill_reason': kill_reason,
        'max_theta_err': max_theta_err,
        'theta_bounded': max_theta_err < 10.0,
        'final_h': state[1],
        'final_vz': state[3],
        'final_theta': np.rad2deg(state[4]),
        'theta_history': theta_history,
        'safety_level': int(hsm.level),
    }


def scan_bias_direction():
    """扫描4种偏置方向, 选最优."""
    configs = [
        ('F+1.25/A-1.25',  1.25, -1.25),
        ('F-1.25/A+1.25', -1.25,  1.25),
        ('F+2.5/A0',       2.5,   0.0),
        ('F0/A+2.5',       0.0,   2.5),
    ]

    print("=" * 70)
    print("Phase 9.0 Step 9a-1: 偏置方向扫描 (死区=0.8°, seed=42)")
    print("=" * 70)

    best_config = None
    best_score = -999

    for name, fwd, aft in configs:
        t0 = time.time()
        r = run_single(seed=42, bias_fwd_deg=fwd, bias_aft_deg=aft)
        elapsed = time.time() - t0

        # 评分: theta有界+outcome
        score = 0
        if r['theta_bounded']:
            score += 100
        if r['outcome'] == 'SAFE_LANDING':
            score += 100
        elif r['outcome'] == 'SAFE_ABORT':
            score += 50
        score -= r['max_theta_err']  # theta误差越小越好

        print(f"\n  [{name}]  ({elapsed:.2f}s)")
        print(f"    outcome={r['outcome']}, max_theta_err={r['max_theta_err']:.1f}°, "
              f"theta_bounded={r['theta_bounded']}")
        print(f"    final: h={r['final_h']:.0f}m, vz={r['final_vz']:.1f}, "
              f"theta={r['final_theta']:.1f}°, L={r['safety_level']}")
        print(f"    score={score:.1f}")

        if score > best_score:
            best_score = score
            best_config = (name, fwd, aft)

    print(f"\n  >>> 最优配置: {best_config[0]} (score={best_score:.1f})")
    return best_config


def run_mc(n_mc, bias_fwd_deg, bias_aft_deg,
           dither_enable=False, gain_comp_enable=False):
    """小规模MC."""
    outcomes = {'SAFE_LANDING': 0, 'SAFE_ABORT': 0, 'UNSAFE_CRASH': 0, 'KILL': 0, 'UNKNOWN': 0}
    theta_errs = []

    for i in range(n_mc):
        r = run_single(seed=i + 100,
                       bias_fwd_deg=bias_fwd_deg, bias_aft_deg=bias_aft_deg,
                       dither_enable=dither_enable, gain_comp_enable=gain_comp_enable)
        outcomes[r['outcome']] += 1
        theta_errs.append(r['max_theta_err'])

    return outcomes, theta_errs


if __name__ == '__main__':
    # ============ Step 1: 偏置方向扫描 ============
    best = scan_bias_direction()
    best_name, best_fwd, best_aft = best

    # ============ Step 2: 最优配置10次MC ============
    print("\n" + "=" * 70)
    print(f"Step 9a-1 MC: {best_name}, 10次, 死区=0.8°")
    print("=" * 70)

    t0 = time.time()
    outcomes, theta_errs = run_mc(10, best_fwd, best_aft)
    elapsed = time.time() - t0

    print(f"  耗时: {elapsed:.1f}s")
    print(f"  结果: {outcomes}")
    print(f"  Safe Landing: {outcomes['SAFE_LANDING']}/10 = {outcomes['SAFE_LANDING']*10}%")
    print(f"  theta_err: mean={np.mean(theta_errs):.1f}°, max={np.max(theta_errs):.1f}°")
    print(f"  theta_bounded (<10°): {sum(1 for e in theta_errs if e < 10.0)}/10")

    # ============ 验收判定 ============
    print("\n  验收:")
    landing_rate = outcomes['SAFE_LANDING'] / 10.0
    if landing_rate > 0.30:
        print(f"    Safe Landing {landing_rate*100:.0f}% > 30% → PASS, 进入9a-2")
    elif landing_rate > 0.0:
        print(f"    Safe Landing {landing_rate*100:.0f}% > 0% 但 < 30% → 部分有效, 仍进入9a-2")
    else:
        print(f"    Safe Landing 0% → 偏置无效, 需检查方向或增大偏置")
