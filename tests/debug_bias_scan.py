"""Quick debug: check why bias doesn't stabilize BELLY phase."""
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
N = 3000  # 30s

H_INIT = 10000.0
VZ_INIT = 300.0
VX_INIT = 50.0
THETA_INIT = np.deg2rad(85.0)

# Test different bias magnitudes
configs = [
    ('No bias (7E)', 0.0, 0.0),
    ('F-1.25/A+1.25', -1.25, 1.25),
    ('F-3.0/A-3.0', -3.0, 3.0),
    ('F+3.0/A-3.0', 3.0, -3.0),
    ('F+5.0/A-5.0', 5.0, -5.0),
    ('F-5.0/A+5.0', -5.0, 5.0),
]

for name, fwd, aft in configs:
    rng = np.random.default_rng(42)
    state = np.array([0.0, H_INIT, VX_INIT, VZ_INIT, THETA_INIT, 0.0, M_FUEL_INIT * 0.7])

    controller = Phase9BellyFlopController(
        bias_enable=(fwd != 0 or aft != 0),
        dither_enable=False,
        gain_comp_enable=False,
        delta_fwd_bias_deg=fwd,
        delta_aft_bias_deg=aft,
    )

    actuators = NonIdealActuatorSuite(rng=rng)
    actuators.randomize(rng)
    # 最坏死区
    actuators.fwd_flap.dead_zone = np.deg2rad(0.8)
    actuators.aft_flap.dead_zone = np.deg2rad(0.8)

    hsm = StarshipSafetyHSM()
    gps_blk = GPSBlackout()
    radar_mp = RadarMultipath(rng=rng)

    max_theta_err = 0.0
    l3_triggered = False

    for k in range(N):
        t = k * DT
        T_cmd, theta_cmd, d_fwd, d_aft, phase, kill, info = controller.update(state, DT)
        if kill:
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

        if hsm_r['abort'] and not l3_triggered:
            l3_triggered = True
            print(f"  [{name}] L3 @ t={t:.1f}s, phase={phase}, "
                  f"theta={np.rad2deg(state[4]):.1f}°")

        if hsm_r['kill']:
            break

        if hsm_r['abort']:
            T_act = 0.0

        state = rk4_step(state, T_act, theta_cmd, DT, d_fwd_act, d_aft_act)

        if phase == 'BELLY':
            err = abs(np.rad2deg(state[4]) - 85.0)
            if err > max_theta_err:
                max_theta_err = err

        if state[1] <= 0:
            break

    print(f"  [{name:16s}] theta_err={max_theta_err:5.1f}°, L3={l3_triggered}, "
          f"final_theta={np.rad2deg(state[4]):5.1f}°, phase={phase}")
