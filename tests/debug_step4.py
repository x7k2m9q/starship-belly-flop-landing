"""快速调试: 单次着陆, 打印关键时序. 姿态环每步都执行."""
import sys, os, time
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
rng = np.random.default_rng(42)
dyn = DynamicsEngine(dt=DT)
tvc = TVC(tau=0.05)
gf = GridFin()
rcs = RCS()
att = AttitudeController(wn=2 * np.pi * 0.3, zeta=0.9)
guidance = LandingGuidance(gfold_N=50, dt=DT)
wind = DrydenWind(rng, sigma_base=1.0)

state = make_state(pos_n=[0, 0, -2000.0], vel_n=[0, 0, 80.0], q=qu.Q_VERT.copy())
fuel = 15000.0
t = 0.0
M_aero_prev = np.zeros(3)
t0 = time.time()

print("t      h      vz     tilt   phase     throttle  thrust   TVC_p  TVC_y  q_des_tilt")
for i in range(15000):
    h = -state[2]
    m, cg_x, I_body = rp.mass_properties(fuel)

    # 制导 (每步都调用, 返回q_des和throttle)
    throttle, q_des, omega_des, tvc_gimbal_cmd, phase = guidance.update(state, fuel, t, DT)
    if phase == 'LANDED':
        break

    # 气动信息
    rho, a, _, _ = atmosphere(h)
    v_mag = np.linalg.norm(state[3:6])
    mach = v_mag / a if a > 0 else 0.0
    qdyn = 0.5 * rho * v_mag * v_mag

    # 姿态环 (每步都执行! G-FOLD段输出TVC指令)
    gf_cmd, rcs_cmd, tvc_gimbal_att = att.update(
        q_des, omega_des, state, mach, qdyn, cg_x, I_body, gf, rcs, DT,
        M_disturbance=M_aero_prev, phase=phase, tvc=tvc, thrust_actual=tvc.thrust if hasattr(tvc,'thrust') else 0.0)

    # 执行器 (每步都更新!)
    M_gf = gf.update(gf_cmd, mach, qdyn, cg_x, DT)
    M_rcs = rcs.update(rcs_cmd, cg_x, DT)

    # TVC: G-FOLD/DEADBAND段用姿态环输出的gimbal, 其他段gimbal=0
    if phase in ('G-FOLD', 'DEADBAND'):
        tvc_p, tvc_y = tvc_gimbal_att[0], tvc_gimbal_att[1]
    else:
        tvc_p, tvc_y = 0.0, 0.0
    thrust_actual, gp, gy = tvc.update(throttle, tvc_p, tvc_y, phase, h, DT)

    # 风
    w = wind.update(h, state[3:6], DT)

    # 动力学
    state, fuel, info = dyn.step(state, fuel, thrust_actual, gp, gy, w,
                                  extra_moment_b=M_gf + M_rcs)
    M_aero_prev = info['M_aero_b'].copy()
    t += DT

    if i % 100 == 0:
        tilt = np.degrees(qu.tilt_angle_from_vertical(state[6:10]))
        qd_tilt = np.degrees(qu.tilt_angle_from_vertical(q_des))
        e_q = qu.quat_error(q_des, state[6:10])
        e_vec_deg = np.degrees(e_q[1:4])
        omega = state[10:13]
        M_aero = M_aero_prev
        safe = "SAFE" if tilt > 5.0 else ""
        print("%5.1f %6.1f %5.1f %6.2f %-8s %6.2f %7.0f %5.2f %5.2f %5.2f ev=[%5.2f,%5.2f,%5.2f] om=[%5.1f,%5.1f,%5.1f] Maero=[%6.0f,%6.0f,%6.0f] %s" %
              (t, h, state[5], tilt, phase, throttle, thrust_actual,
               np.degrees(gp), np.degrees(gy), qd_tilt, e_vec_deg[0], e_vec_deg[1], e_vec_deg[2],
               np.degrees(omega[0]), np.degrees(omega[1]), np.degrees(omega[2]),
               M_aero[0], M_aero[1], M_aero[2], safe))
    if h < 0.0:
        break

print("\n终态: h=%.2f  vz=%.2f  tilt=%.2f°  水平偏差=%.2fm  t=%.1fs  耗时=%.1fs" %
      (-state[2], state[5], np.degrees(qu.tilt_angle_from_vertical(state[6:10])),
       np.hypot(state[0], state[1]), t, time.time() - t0))