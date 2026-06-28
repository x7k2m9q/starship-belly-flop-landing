"""Phase 9.0 FLIP阶段诊断: 追踪单次仿真的theta/phase时间历程."""
import sys, os
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


def run_trace(seed=100):
    rng = np.random.default_rng(seed)
    m_fuel_init = M_FUEL_INIT * 0.7 * (1.0 + rng.uniform(-0.05, 0.05))
    vz_init = VZ_INIT * (1.0 + rng.uniform(-0.10, 0.10))
    vx_init = VX_INIT * (1.0 + rng.uniform(-0.20, 0.20))
    state = np.array([0.0, H_INIT, vx_init, vz_init, THETA_INIT, 0.0, m_fuel_init])

    controller = Phase9BellyFlopController(
        bias_enable=False, dither_enable=False, gain_comp_enable=False)
    actuators = NonIdealActuatorSuite(rng=rng)
    actuators.randomize(rng)
    hsm = StarshipSafetyHSM()
    gps_blk = GPSBlackout()
    radar_mp = RadarMultipath(rng=rng)

    print(f"seed={seed}, m_fuel={m_fuel_init:.0f}kg, vz0={vz_init:.1f}, vx0={vx_init:.1f}")
    print(f"死区: fwd={np.rad2deg(actuators.fwd_flap.dead_zone):.2f}°, aft={np.rad2deg(actuators.aft_flap.dead_zone):.2f}°")
    print(f"{'t':>6} {'phase':>8} {'h':>8} {'V':>7} {'theta':>8} {'q':>8} {'d_fwd':>8} {'d_aft':>8} {'d_fwd_act':>10} {'L':>3}")

    prev_phase = 'BELLY'
    l3_printed = False
    for k in range(N):
        t = k * DT
        T_cmd, theta_cmd, d_fwd, d_aft, phase, kill, info = controller.update(state, DT)
        if kill:
            print(f"  KILL at t={t:.1f}s: {info.get('kill_reason','?')}")
            break

        T_act = actuators.update_thrust(T_cmd, DT, t)
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

        phase_changed = phase != prev_phase
        if phase_changed:
            print(f"  >>> {prev_phase}->{phase} at t={t:.2f}s, theta={np.rad2deg(state[4]):.1f}°, h={state[1]:.0f}m, V={V:.1f}m/s, q={np.rad2deg(state[5]):.1f}°/s")
            prev_phase = phase

        if hsm_r['abort'] and not l3_printed:
            print(f"  !!! L3 at t={t:.2f}s, phase={phase}, theta={np.rad2deg(state[4]):.1f}°, h={state[1]:.0f}m, q={np.rad2deg(state[5]):.1f}°/s")
            l3_printed = True

        # 打印: 阶段切换后10步 + 每5秒
        print_now = phase_changed or (k % 500 == 0) or (hsm_r['abort'] and not l3_printed)
        # FLIP阶段打印更频繁
        if phase == 'FLIP' and k % 50 == 0:
            print_now = True
        if print_now:
            print(f"{t:6.1f} {phase:>8} {state[1]:8.0f} {V:7.1f} {np.rad2deg(state[4]):8.1f} {np.rad2deg(state[5]):8.1f} {np.rad2deg(d_fwd):8.2f} {np.rad2deg(d_aft):8.2f} {np.rad2deg(d_fwd_act):10.2f} {int(hsm.level):3d}")

        if hsm_r['kill']:
            print(f"  HSM KILL at t={t:.1f}s")
            break
        if hsm_r['abort']:
            T_act = 0.0
            if hsm_r['landing_prediction'] is not None:
                if hsm_r['landing_prediction']['in_safe_zone']:
                    print(f"  SAFE_ABORT at t={t:.1f}s, h={state[1]:.0f}m")
                else:
                    print(f"  UNSAFE_CRASH at t={t:.1f}s")
                break

        state = rk4_step(state, T_act, theta_cmd, DT, d_fwd_act, d_aft_act)
        if state[1] <= 0:
            state[1] = 0.0
            vz_f = state[3]
            th_f = abs(np.rad2deg(state[4]))
            print(f"  触地: vz={vz_f:.1f}, theta={th_f:.1f}° -> {'SAFE_LANDING' if abs(vz_f)<6 and th_f<15 else 'UNSAFE_CRASH'}")
            break


if __name__ == '__main__':
    run_trace(seed=100)
