"""调试7C-3解析Jacobian精度问题."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from src.belly_flop.scvx import dynamics_7c2, jacobian_7c2, analytical_jacobian_7c3
from src.belly_flop.aero_model import (
    aero_coefficients, angle_of_attack, atmosphere, get_mass,
    S_REF, M_FUEL_INIT, T_IDLE
)

# 配平点附近状态
state = np.array([100.0, 8000.0, 50.0, 300.0, np.deg2rad(85.0), 0.0])
U = np.array([T_IDLE, np.deg2rad(85.0)])
m_fuel = M_FUEL_INIT * 0.8

x, h, vx, vz, theta, q = state
T, theta_cmd = U
m = get_mass(m_fuel)

V = np.sqrt(vx**2 + vz**2)
gamma = np.arctan2(vx, vz)
alpha = theta - gamma

print(f"V={V:.4f}, gamma={np.degrees(gamma):.4f}deg, alpha={np.degrees(alpha):.4f}deg")
print(f"theta={np.degrees(theta):.4f}deg, m={m:.1f}kg")

rho, a_sound, p, T_air = atmosphere(h)
M = V / a_sound
print(f"rho={rho:.6f}, a_sound={a_sound:.4f}, M={M:.4f}")

CD0, CDa, CLa, Cma = aero_coefficients(M)
print(f"CD0={CD0:.4f}, CDa={CDa:.4f}, CLa={CLa:.4f}, Cma={Cma:.4f}")

sin_alpha = np.sin(alpha)
sin_2alpha = np.sin(2*alpha)
cos_2alpha = np.cos(2*alpha)
CD = CD0 + CDa * sin_alpha**2
CL = CLa * sin_2alpha * 0.5
Q = 0.5 * rho * V**2 * S_REF
D = Q * CD
L = Q * CL

print(f"CD={CD:.4f}, CL={CL:.4f}, Q={Q:.4e}")
print(f"D={D:.4e}, L={L:.4e}")

sin_gamma = np.sin(gamma)
cos_gamma = np.cos(gamma)
Fx_aero = -D * sin_gamma + L * cos_gamma
Fz_aero = -D * cos_gamma - L * sin_gamma
print(f"Fx_aero={Fx_aero:.4e}, Fz_aero={Fz_aero:.4e}")

# 解析偏导
dCD_dalpha = CDa * sin_2alpha
dCL_dalpha = CLa * cos_2alpha * 0.5
dD_dtheta = Q * dCD_dalpha
dL_dtheta = Q * dCL_dalpha
print(f"dCD_dalpha={dCD_dalpha:.4f}, dCL_dalpha={dCL_dalpha:.4f}")
print(f"dD_dtheta={dD_dtheta:.4e}, dL_dtheta={dL_dtheta:.4e}")

dFx_dtheta = -dD_dtheta * sin_gamma + dL_dtheta * cos_gamma
dFz_dtheta = -dD_dtheta * cos_gamma - dL_dtheta * sin_gamma
print(f"dFx_dtheta={dFx_dtheta:.4e}, dFz_dtheta={dFz_dtheta:.4e}")

sin_theta = np.sin(theta)
cos_theta = np.cos(theta)
dax_dtheta = (dFx_dtheta + T * cos_theta) / m
daz_dtheta = (dFz_dtheta + T * sin_theta) / m
print(f"dax_dtheta={dax_dtheta:.6f}, daz_dtheta={daz_dtheta:.6f}")

# 数值Jacobian
A_num, B_num = jacobian_7c2(state, U, m_fuel)
print(f"\n数值Jacobian:")
print(f"A_num[2,4]={A_num[2,4]:.6f} (dax/dtheta)")
print(f"A_num[3,4]={A_num[3,4]:.6f} (daz/dtheta)")

# 解析Jacobian
A_ana, B_ana = analytical_jacobian_7c3(state, U, m_fuel)
print(f"\n解析Jacobian:")
print(f"A_ana[2,4]={A_ana[2,4]:.6f} (dax/dtheta)")
print(f"A_ana[3,4]={A_ana[3,4]:.6f} (daz/dtheta)")

# 手动数值差分验证
eps = 1e-6
state_plus = state.copy()
state_plus[4] += eps
state_minus = state.copy()
state_minus[4] -= eps
f_plus = dynamics_7c2(state_plus, U, m_fuel)
f_minus = dynamics_7c2(state_minus, U, m_fuel)
dax_dtheta_num = (f_plus[2] - f_minus[2]) / (2*eps)
daz_dtheta_num = (f_plus[3] - f_minus[3]) / (2*eps)
print(f"\n手动数值差分:")
print(f"dax_dtheta={dax_dtheta_num:.6f}")
print(f"daz_dtheta={daz_dtheta_num:.6f}")

# 检查aero_forces_and_moments的输出
from src.belly_flop.aero_model import aero_forces_and_moments
result = aero_forces_and_moments(vx, vz, theta, h, 0.0, 0.0)
D2, L2, Fx2, Fz2, M_aero2, M_flap2, M_total2, Q2, alpha2, gamma2, M2, rho2, a2 = result
print(f"\naero_forces_and_moments输出:")
print(f"D={D2:.4e}, L={L2:.4e}, Fx={Fx2:.4e}, Fz={Fz2:.4e}")
print(f"alpha={np.degrees(alpha2):.4f}deg, gamma={np.degrees(gamma2):.4f}deg")

# 数值差分aero_forces_and_moments
result_plus = aero_forces_and_moments(vx, vz, theta+eps, h, 0.0, 0.0)
result_minus = aero_forces_and_moments(vx, vz, theta-eps, h, 0.0, 0.0)
dFx_dtheta_aero = (result_plus[2] - result_minus[2]) / (2*eps)
dFz_dtheta_aero = (result_plus[3] - result_minus[3]) / (2*eps)
print(f"\naero_forces_and_moments数值差分:")
print(f"dFx_dtheta={dFx_dtheta_aero:.4e} (解析: {dFx_dtheta:.4e})")
print(f"dFz_dtheta={dFz_dtheta_aero:.4e} (解析: {dFz_dtheta:.4e})")

# 数值验证D和L对theta的偏导
D_plus = result_plus[0]
D_minus = result_minus[0]
L_plus = result_plus[1]
L_minus = result_minus[1]
dD_dtheta_num = (D_plus - D_minus) / (2*eps)
dL_dtheta_num = (L_plus - L_minus) / (2*eps)
print(f"\nD和L对theta的偏导验证:")
print(f"dD_dtheta: 数值={dD_dtheta_num:.4e}, 解析={dD_dtheta:.4e}")
print(f"dL_dtheta: 数值={dL_dtheta_num:.4e}, 解析={dL_dtheta:.4e}")

# 检查alpha在theta扰动下的变化
alpha_plus = result_plus[8]
alpha_minus = result_minus[8]
dalpha_dtheta_num = (alpha_plus - alpha_minus) / (2*eps)
print(f"\nalpha对theta的偏导: 数值={dalpha_dtheta_num:.6f} (应为1.0)")

# 检查alpha值
print(f"alpha(theta-eps)={np.degrees(alpha_minus):.6f}deg")
print(f"alpha(theta)     ={np.degrees(alpha2):.6f}deg")
print(f"alpha(theta+eps)={np.degrees(alpha_plus):.6f}deg")

# 检查归一化是否导致跳变
alpha_raw_plus = (theta + eps) - gamma
alpha_raw_minus = (theta - eps) - gamma
print(f"\nalpha_raw(theta-eps)={np.degrees(alpha_raw_minus):.6f}deg")
print(f"alpha_raw(theta+eps)={np.degrees(alpha_raw_plus):.6f}deg")
print(f"归一化后:")
alpha_norm_plus = (alpha_raw_plus + np.pi) % (2*np.pi) - np.pi
alpha_norm_minus = (alpha_raw_minus + np.pi) % (2*np.pi) - np.pi
print(f"alpha_norm(theta-eps)={np.degrees(alpha_norm_minus):.6f}deg")
print(f"alpha_norm(theta+eps)={np.degrees(alpha_norm_plus):.6f}deg")
