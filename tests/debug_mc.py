"""调试单个蒙特卡洛失败案例."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# 挑战案例: h=2500, vz=150, px=50, vx=10, fuel=10000
pos0 = np.array([50.0, 0.0, -2500.0])
vel0 = np.array([10.0, 0.0, 150.0])
fuel_init = 10000.0

rng = np.random.default_rng(12345)
dyn = DynamicsEngine(dt=DT)
tvc = TVC(tau=0.05)
gf = GridFin()
rcs = RCS()
att = AttitudeController(wn=2 * np.pi * 0.3, zeta=0.9)
guidance = LandingGuidance(gfold_N=50, dt=DT)
wind = DrydenWind(rng, sigma_base=2.0)

state = make_state(pos_n=pos0, vel_n=vel0, q=qu.Q_VERT.copy())
fuel = fuel_init
t = 0.0
M_aero_prev = np.zeros(3)

print("t      h      vz     vx     tilt   phase     throttle  thrust   TVC_p  TVC_y  qd_tilt  h_pos")
for i in range(15000):
    h = -state[2]
    m, cg_x, I_body = rp.mass_properties(fuel)

    throttle, q_des, omega_des, tvc_gimbal_cmd, phase = guidance.update(state, fuel, t, DT)
    if phase == 'LANDED':
        break

    rho, a, _, _ = atmosphere(h)
    v_mag = np.linalg.norm(state[3:6])
    mach = v_mag / a if a > 0 else 0.0
    qdyn = 0.5 * rho * v_mag * v_mag

    gf_cmd, rcs_cmd, tvc_gimbal_att = att.update(
        q_des, omega_des, state, mach, qdyn, cg_x, I_body, gf, rcs, DT,
        M_disturbance=M_aero_prev, phase=phase, tvc=tvc, thrust_actual=tvc.thrust)

    M_gf = gf.update(gf_cmd, mach, qdyn, cg_x, DT)
    M_rcs = rcs.update(rcs_cmd, cg_x, DT)

    if phase in ('G-FOLD', 'DEADBAND'):
        tvc_p, tvc_y = tvc_gimbal_att[0], tvc_gimbal_att[1]
    else:
        tvc_p, tvc_y = 0.0, 0.0
    thrust_actual, gp, gy = tvc.update(throttle, tvc_p, tvc_y, phase, h, DT)

    w = wind.update(h, state[3:6], DT)
    state, fuel, info = dyn.step(state, fuel, thrust_actual, gp, gy, w,
                                  extra_moment_b=M_gf + M_rcs)
    M_aero_prev = info['M_aero_b'].copy()
    t += DT

    if i % 100 == 0:
        tilt = np.degrees(qu.tilt_angle_from_vertical(state[6:10]))
        qd_tilt = np.degrees(qu.tilt_angle_from_vertical(q_des))
        h_pos = np.hypot(state[0], state[1])
        print("%5.1f %6.1f %5.1f %5.1f %6.2f %-8s %6.2f %7.0f %5.2f %5.2f %5.2f %6.1f" %
              (t, h, state[5], state[0], tilt, phase, throttle, thrust_actual,
               np.degrees(gp), np.degrees(gy), qd_tilt, h_pos))
    if h < 0.0:
        break

print("\n终态: h=%.2f  vz=%.2f  tilt=%.2f°  水平偏差=%.2fm  t=%.1fs  剩余燃料=%.0fkg" %
      (-state[2], state[5], np.degrees(qu.tilt_angle_from_vertical(state[6:10])),
       np.hypot(state[0], state[1]), t, fuel))
