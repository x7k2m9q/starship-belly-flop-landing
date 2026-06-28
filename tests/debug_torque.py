"""调试G-FOLD段姿态环力矩平衡."""
import sys, os; sys.path.insert(0, '.')
import numpy as np
import src.quaternion_utils as qu
import src.rocket_params as rp
from src.dynamics import DynamicsEngine, make_state
from src.actuators import TVC, GridFin, RCS
from src.attitude_control import AttitudeController
from src.guidance import LandingGuidance
from src.wind import DrydenWind
from src.atmosphere import atmosphere

DT = 0.01
rng = np.random.default_rng(42)
dyn = DynamicsEngine(dt=DT)
tvc = TVC(tau=0.5); gf = GridFin(); rcs = RCS()
att = AttitudeController(wn=2 * np.pi * 0.5, zeta=0.9)
guidance = LandingGuidance(gfold_N=20, dt=DT)
wind = DrydenWind(rng, sigma_base=1.0)
state = make_state(pos_n=[0,0,-2000.0], vel_n=[0,0,80.0], q=qu.Q_VERT.copy())
fuel = 15000.0; t = 0.0; M_aero_prev = np.zeros(3)

print("t    h    vz   tilt  qd_tilt phase   M_cmd_y   M_gf_y    M_aero_y  qdyn   mach  gf_sat")
for i in range(5000):
    h = -state[2]
    m, cg_x, I_body = rp.mass_properties(fuel)
    throttle, q_des, omega_des, _, phase = guidance.update(state, fuel, t, DT)
    if phase == 'LANDED': break
    rho, a, _, _ = atmosphere(h)
    v_mag = np.linalg.norm(state[3:6])
    mach = v_mag / a if a > 0 else 0.0
    qdyn = 0.5 * rho * v_mag * v_mag
    gf_cmd, rcs_cmd, _ = att.update(q_des, omega_des, state, mach, qdyn, cg_x, I_body, gf, rcs, DT, M_disturbance=M_aero_prev)
    M_gf = gf.update(gf_cmd, mach, qdyn, cg_x, DT)
    M_rcs = rcs.update(rcs_cmd, cg_x, DT)
    thrust_actual, gp, gy = tvc.update(throttle, 0.0, 0.0, phase, h, DT)
    w = wind.update(h, state[3:6], DT)
    state, fuel, info = dyn.step(state, fuel, thrust_actual, gp, gy, w, extra_moment_b=M_gf + M_rcs)
    M_aero_prev = info['M_aero_b'].copy()
    t += DT
    if i % 50 == 0 and phase == 'G-FOLD':
        tilt = np.degrees(qu.tilt_angle_from_vertical(state[6:10]))
        qd_tilt = np.degrees(qu.tilt_angle_from_vertical(q_des))
        M_cmd_y = att.last_M_cmd[1]
        M_gf_y = M_gf[1]
        M_aero_y = M_aero_prev[1]
        gf_max = gf.max_torque_estimate(mach, qdyn, cg_x)[1]
        sat = "SAT" if abs(M_cmd_y) > gf_max else ""
        print("%4.1f %5.0f %5.1f %5.1f %5.1f %-6s %9.0f %9.0f %9.0f %7.0f %5.2f %s" %
              (t, h, state[5], tilt, qd_tilt, phase, M_cmd_y, M_gf_y, M_aero_y, qdyn, mach, sat))
    if h < 0.0: break