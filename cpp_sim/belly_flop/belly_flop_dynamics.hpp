// =============================================================================
// belly_flop_dynamics.hpp - Belly-Flop 2D 动力学 (C++翻译, Step 7E)
// =============================================================================
// 状态: X = [x, h, vx, vz, θ, q, m_fuel] (7维)
// 控制: T, θ_cmd, δ_extra_fwd, δ_extra_aft
// 积分: RK4, dt=0.01s
//
// 对应 Python: src/belly_flop/dynamics.py
// =============================================================================
#ifndef FALCON9_BELLY_FLOP_DYNAMICS_HPP
#define FALCON9_BELLY_FLOP_DYNAMICS_HPP

#include "belly_flop_aero.hpp"
#include <cmath>

namespace falcon9 {
namespace belly_flop {

// =============================================================================
// 状态结构体 (缺陷23: 统一状态结构体, 全阶段一致)
// =============================================================================
struct State {
    float x;       // 水平位置 [m]
    float h;       // 高度 [m] (向上为正)
    float vx;      // 水平速度 [m/s]
    float vz;      // 垂直速度 [m/s] (正向下)
    float theta;   // 俯仰角 [rad] (0=垂直, π/2=水平腹部朝下)
    float q;       // 俯仰角速度 [rad/s]
    float m_fuel;  // 燃料质量 [kg]

    void reset() {
        x = 0.0f; h = 0.0f; vx = 0.0f; vz = 0.0f;
        theta = 0.0f; q = 0.0f; m_fuel = 0.0f;
    }
};

// =============================================================================
// 状态导数
// =============================================================================
inline State state_derivative(const State& s, float T, float theta_cmd,
                               float delta_extra_fwd, float delta_extra_aft) {
    State d{};

    float m = get_mass(s.m_fuel);
    float Iyy = get_Iyy(s.m_fuel);
    float g = gravity(s.h);

    float V = std::sqrt(s.vx * s.vx + s.vz * s.vz);
    AlphaGamma ag = angle_of_attack(s.theta, s.vx, s.vz);

    AtmosphereResult atm = atmosphere(s.h);
    float M = (atm.a_sound > 0.0f) ? (V / atm.a_sound) : 0.0f;

    // 配平襟翼: 在 α_cmd = θ_cmd - γ 处配平
    float alpha_cmd = theta_cmd - ag.gamma;
    alpha_cmd = std::fmod(alpha_cmd + 3.14159265358979f, 2.0f * 3.14159265358979f) - 3.14159265358979f;

    float delta_trim_fwd, delta_trim_aft;
    trim_flaps(alpha_cmd, M, delta_trim_fwd, delta_trim_aft);

    // 总襟翼 = 配平 + 额外
    float delta_fwd = delta_trim_fwd + delta_extra_fwd;
    float delta_aft = delta_trim_aft + delta_extra_aft;
    if (delta_fwd > DELTA_MAX) delta_fwd = DELTA_MAX;
    if (delta_fwd < -DELTA_MAX) delta_fwd = -DELTA_MAX;
    if (delta_aft > DELTA_MAX) delta_aft = DELTA_MAX;
    if (delta_aft < -DELTA_MAX) delta_aft = -DELTA_MAX;

    // 气动力/力矩
    AeroResult aero = aero_forces_and_moments(s.vx, s.vz, s.theta, s.h, delta_fwd, delta_aft);

    // 状态导数
    d.x = s.vx;
    d.h = -s.vz;  // vz 正向下, h 正向上
    d.vx = (aero.Fx_aero + T * std::sin(s.theta)) / m;
    d.vz = (aero.Fz_aero - T * std::cos(s.theta)) / m + g;
    d.theta = s.q;
    d.q = aero.M_total / Iyy;
    d.m_fuel = -T / (ISP * G0_ISP);

    return d;
}

// =============================================================================
// RK4 单步积分
// =============================================================================
inline State rk4_step(const State& s, float T, float theta_cmd, float dt,
                       float delta_extra_fwd = 0.0f, float delta_extra_aft = 0.0f) {
    State k1 = state_derivative(s, T, theta_cmd, delta_extra_fwd, delta_extra_aft);

    State s2 = s;
    s2.x += 0.5f * dt * k1.x;
    s2.h += 0.5f * dt * k1.h;
    s2.vx += 0.5f * dt * k1.vx;
    s2.vz += 0.5f * dt * k1.vz;
    s2.theta += 0.5f * dt * k1.theta;
    s2.q += 0.5f * dt * k1.q;
    s2.m_fuel += 0.5f * dt * k1.m_fuel;
    State k2 = state_derivative(s2, T, theta_cmd, delta_extra_fwd, delta_extra_aft);

    State s3 = s;
    s3.x += 0.5f * dt * k2.x;
    s3.h += 0.5f * dt * k2.h;
    s3.vx += 0.5f * dt * k2.vx;
    s3.vz += 0.5f * dt * k2.vz;
    s3.theta += 0.5f * dt * k2.theta;
    s3.q += 0.5f * dt * k2.q;
    s3.m_fuel += 0.5f * dt * k2.m_fuel;
    State k3 = state_derivative(s3, T, theta_cmd, delta_extra_fwd, delta_extra_aft);

    State s4 = s;
    s4.x += dt * k3.x;
    s4.h += dt * k3.h;
    s4.vx += dt * k3.vx;
    s4.vz += dt * k3.vz;
    s4.theta += dt * k3.theta;
    s4.q += dt * k3.q;
    s4.m_fuel += dt * k3.m_fuel;
    State k4 = state_derivative(s4, T, theta_cmd, delta_extra_fwd, delta_extra_aft);

    State result;
    result.x = s.x + (dt / 6.0f) * (k1.x + 2.0f * k2.x + 2.0f * k3.x + k4.x);
    result.h = s.h + (dt / 6.0f) * (k1.h + 2.0f * k2.h + 2.0f * k3.h + k4.h);
    result.vx = s.vx + (dt / 6.0f) * (k1.vx + 2.0f * k2.vx + 2.0f * k3.vx + k4.vx);
    result.vz = s.vz + (dt / 6.0f) * (k1.vz + 2.0f * k2.vz + 2.0f * k3.vz + k4.vz);
    result.theta = s.theta + (dt / 6.0f) * (k1.theta + 2.0f * k2.theta + 2.0f * k3.theta + k4.theta);
    result.q = s.q + (dt / 6.0f) * (k1.q + 2.0f * k2.q + 2.0f * k3.q + k4.q);
    result.m_fuel = s.m_fuel + (dt / 6.0f) * (k1.m_fuel + 2.0f * k2.m_fuel + 2.0f * k3.m_fuel + k4.m_fuel);

    // 燃料非负
    if (result.m_fuel < 0.0f) result.m_fuel = 0.0f;

    return result;
}

}  // namespace belly_flop
}  // namespace falcon9

#endif  // FALCON9_BELLY_FLOP_DYNAMICS_HPP
