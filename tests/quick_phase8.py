"""Quick smoke test for Phase 8.0 MC - runs 1 simulation with progress."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.belly_flop.integrated_controller import IntegratedBellyFlopController
from src.belly_flop.dynamics import rk4_step
from src.belly_flop.aero_model import M_FUEL_INIT, atmosphere
from src.starship_non_ideal import NonIdealActuatorSuite
from src.starship_safety_hsm import StarshipSafetyHSM
from src.sensors import GPSBlackout, RadarMultipath

DT = 0.01
T_END = 90.0
H_INIT = 10000.0
VZ_INIT = 300.0
VX_INIT = 50.0
THETA_INIT = np.deg2rad(85.0)
M_FUEL_SIM = M_FUEL_INIT * 0.7

rng = np.random.default_rng(42)
state = np.array([0.0, H_INIT, VX_INIT, VZ_INIT, THETA_INIT, 0.0, M_FUEL_SIM])

actuators = NonIdealActuatorSuite(rng=rng)
actuators.randomize(rng)
hsm = StarshipSafetyHSM()
controller = IntegratedBellyFlopController()
gps_blk = GPSBlackout()
radar_mp = RadarMultipath(rng=rng)

N = int(T_END / DT)
t0 = time.time()

print(f"Starting sim: N={N} steps, dt={DT}s")
print(f"Initial: h={H_INIT}m, vz={VZ_INIT}m/s, theta=85deg")

for k in range(N):
    t = k * DT
    
    T_cmd, theta_cmd, d_fwd, d_aft, phase, kill, info = controller.update(state, DT)
    
    if kill:
        print(f"  CTRL KILL @ t={t:.1f}s, phase={phase}")
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
    
    hsm_r = hsm.update(state, d_fwd, d_aft, d_fwd_act, d_aft_act, radar_jump, t, DT, phase=phase)
    
    if hsm_r['kill']:
        print(f"  HSM KILL @ t={t:.1f}s: {hsm_r['abort_reason']}")
        break
    
    if hsm_r['abort']:
        T_act = 0.0
    
    state = rk4_step(state, T_act, theta_cmd, DT, d_fwd_act, d_aft_act)
    
    if k % 1000 == 0:
        print(f"  t={t:.1f}s: phase={phase}, h={state[1]:.0f}m, vz={state[3]:.1f}m/s, "
              f"theta={np.rad2deg(state[4]):.1f}deg, L={hsm_r['level']}")
    
    if state[1] <= 0:
        print(f"  TOUCHDOWN @ t={t:.1f}s: h={state[1]:.1f}m, vz={state[3]:.1f}m/s, "
              f"vx={state[2]:.1f}m/s, theta={np.rad2deg(state[4]):.1f}deg")
        print(f"  phase={phase}, safety_level={hsm_r['level']}")
        print(f"  L1={hsm.l1_trigger_count}, L2={hsm.l2_trigger_count}, L3={hsm.l3_trigger_count}")
        break

elapsed = time.time() - t0
print(f"\nElapsed: {elapsed:.2f}s ({N} steps, {elapsed/max(k,1)*1000:.2f}ms/step)")
