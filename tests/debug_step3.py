"""临时调试: 诊断Step3跟踪慢的本质原因."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import src.quaternion_utils as qu
import src.rocket_params as rp
from src.dynamics import DynamicsEngine, make_state
from src.actuators import TVC, GridFin, RCS
from src.attitude_control import AttitudeController, q_des_from_tilt
from src.atmosphere import atmosphere

DT = 0.01
G = rp.G0

def debug():
    rng = np.random.default_rng(0)
    dyn = DynamicsEngine(dt=DT)
    tvc = TVC(tau=0.5)
    gf = GridFin()
    rcs = RCS()
    wn = 2 * np.pi * 0.5
    att = AttitudeController(wn=wn, zeta=0.9)
    state = make_state(pos_n=[0, 0, -1000.0],
                       vel_n=[0, 0, 120.0], q=qu.Q_VERT.copy())
    fuel = 30000.0
    q_des = q_des_from_tilt(np.radians(3.0), np.array([0, 1, 0]))
    omega_des = np.zeros(3)
    n = int(3.0 / DT)
    M_aero_prev = np.zeros(3)

    print("wn=%.2f rad/s, zeta=%.2f" % (wn, 0.9))
    m, cg_x, I_body = rp.mass_properties(fuel)
    print("初始: m=%.0f, cg_x=%.3f, Iyy=%.0f" % (m, cg_x, I_body[1,1]))
    rho, a, _, _ = atmosphere(1000.0)
    v = 120.0
    mach = v / a
    qdyn = 0.5 * rho * v * v
    print("h=1000m: rho=%.3f, a=%.1f, Mach=%.3f, qdyn=%.0f Pa" % (rho, a, mach, qdyn))
    gf_max = gf.max_torque_estimate(mach, qdyn, cg_x)
    print("栅格舵max力矩: [roll=%.0f, pitch=%.0f, yaw=%.0f]" % tuple(gf_max))
    print()

    print("%6s %8s %8s %8s %10s %10s %10s %10s %10s %10s" % (
        "t", "tilt°", "omega°/s", "e_vec_y", "M_cmd_y", "gf_cmd_y", "M_gf_y", "M_rcs_y", "M_aero_y", "M_net_y"))
    for i in range(n):
        h = -state[2]
        m, cg_x, I_body = rp.mass_properties(fuel)
        thrust = m * G
        if i > 0:
            mach = info['mach']; qdyn = info['qdyn']
        else:
            rho, a, _, _ = atmosphere(h)
            mach = 120.0 / a; qdyn = 0.5 * rho * 120.0 ** 2
        gf_cmd, rcs_cmd, tvc_gimbal = att.update(
            q_des, omega_des, state, mach, qdyn, cg_x, I_body, gf, rcs, DT,
            M_disturbance=M_aero_prev)
        M_gf = gf.update(gf_cmd, mach, qdyn, cg_x, DT)
        M_rcs = rcs.update(rcs_cmd, cg_x, DT)
        throttle = 0.5  # 悬停油门 ([0,1]映射)
        thrust_actual, gp, gy = tvc.update(throttle, 0.0, 0.0, 'CRUISE', h, DT)
        w = np.zeros(3)
        state, fuel, info = dyn.step(
            state, fuel, thrust_actual, 0.0, 0.0, w,
            extra_moment_b=M_gf + M_rcs)
        M_aero = info['M_aero_b']
        tilt = np.degrees(qu.tilt_angle_from_vertical(state[6:10]))
        omega = state[10:13]
        e_q = qu.quat_error(q_des, state[6:10])
        M_aero_prev = M_aero.copy()
        if i % 10 == 0 or i < 20:
            M_net = M_gf[1] + M_rcs[1] + M_aero[1]
            print("%6.2f %8.3f %8.3f %8.5f %10.0f %10.3f %10.0f %10.0f %10.0f %10.0f" % (
                i*DT, tilt, np.degrees(omega[1]), e_q[2],
                att.last_M_cmd[1], gf_cmd[1],
                M_gf[1], M_rcs[1], M_aero[1], M_net))

if __name__ == "__main__":
    debug()
