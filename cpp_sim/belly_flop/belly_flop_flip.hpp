// =============================================================================
// belly_flop_flip.hpp - 翻转段控制器 (C++翻译, Step 7D/7E)
// =============================================================================
// bang-bang + 标称前馈 + PD + 前馈力矩补偿
//
// 暗礁21: t_switch = t_target/2 (目标翻转时间反推)
// 暗礁22: M_nominal + M_pd + M_ff 三项力矩叠加
//
// 对应 Python: src/belly_flop/flip_controller.py
// =============================================================================
#ifndef FALCON9_BELLY_FLOP_FLIP_HPP
#define FALCON9_BELLY_FLOP_FLIP_HPP

#include "belly_flop_aero.hpp"
#include "belly_flop_dynamics.hpp"
#include <cmath>

namespace falcon9 {
namespace belly_flop {

// =============================================================================
// 翻转段参数
// =============================================================================
constexpr float THETA_BELLY_F    = 1.48352986419518f;  // 85° in rad
constexpr float THETA_LAND_F     = 0.0f;
constexpr float T_FLIP_MAX       = 8.0f;               // s, Kill阈值
constexpr float T_FLIP_TARGET    = 3.5f;               // s, 目标翻转时间
constexpr float M_MARGIN         = 0.8f;               // 力矩裕度
constexpr float WN_TRACK         = 3.0f;               // PD自然频率
constexpr float ZETA_TRACK       = 0.7f;               // 阻尼比
constexpr float PI_F             = 3.14159265358979f;
constexpr float DEG2RAD_F        = 0.017453292519943f;

// =============================================================================
// bang-bang期望轨迹
// =============================================================================
struct BangBangTraj {
    float theta;      // [rad]
    float theta_dot;  // [rad/s]
};

inline BangBangTraj bangbang_theta_trajectory(float t, float theta0, float thetaf,
                                                float t_switch, float t_total) {
    BangBangTraj traj{};

    if (t_total < 1e-6f) {
        traj.theta = thetaf;
        traj.theta_dot = 0.0f;
        return traj;
    }

    float delta_theta = theta0 - thetaf;
    float alpha_max = (t_switch > 0.0f) ? (delta_theta / (t_switch * t_switch)) : 0.0f;

    if (t < 0.0f) {
        traj.theta = theta0;
        traj.theta_dot = 0.0f;
    } else if (t < t_switch) {
        // 加速阶段
        traj.theta = theta0 - 0.5f * alpha_max * t * t;
        traj.theta_dot = -alpha_max * t;
    } else if (t < t_total) {
        // 减速阶段
        float t_dec = t - t_switch;
        float theta_mid = theta0 - 0.5f * alpha_max * t_switch * t_switch;
        traj.theta = theta_mid - alpha_max * t_switch * t_dec + 0.5f * alpha_max * t_dec * t_dec;
        traj.theta_dot = -alpha_max * t_switch + alpha_max * t_dec;
    } else {
        traj.theta = thetaf;
        traj.theta_dot = 0.0f;
    }

    return traj;
}

inline float bangbang_theta_acceleration(float t, float t_switch, float t_total, float alpha_max) {
    if (t < 0.0f || t >= t_total) return 0.0f;
    if (t < t_switch) return -alpha_max;
    return alpha_max;
}

// =============================================================================
// 前馈力矩补偿
// =============================================================================
inline float compute_feedforward_torque(const State& s, float theta_ref) {
    float V = std::sqrt(s.vx * s.vx + s.vz * s.vz);
    if (V < 1e-6f) return 0.0f;

    AtmosphereResult atm = atmosphere(s.h);
    float M = V / atm.a_sound;
    AlphaGamma ag = angle_of_attack(s.theta, s.vx, s.vz);

    float alpha_ref = theta_ref - ag.gamma;
    alpha_ref = std::fmod(alpha_ref + PI_F, 2.0f * PI_F) - PI_F;

    AeroCoeffs ac = aero_coefficients(M);
    float Cm_actual = -ac.Cma * std::sin(ag.alpha - THETA_BELLY_F);
    float Cm_ref = -ac.Cma * std::sin(alpha_ref - THETA_BELLY_F);

    float Q = 0.5f * atm.rho * V * V * S_REF;
    float M_residual = Q * S_REF * L_REF * (Cm_actual - Cm_ref);

    return -M_residual;  // 补偿残余
}

// =============================================================================
// 翻转控制器
// =============================================================================
class FlipController {
public:
    float theta0;
    float thetaf;
    float t_switch;
    float t_total;
    float alpha_max;
    float M_max;
    bool planned;
    float flip_t;

    FlipController(float theta0_ = THETA_BELLY_F, float thetaf_ = THETA_LAND_F)
        : theta0(theta0_), thetaf(thetaf_), t_switch(0.0f), t_total(0.0f),
          alpha_max(0.0f), M_max(0.0f), planned(false), flip_t(0.0f) {}

    // 规划bang-bang轨迹
    void plan(const State& s) {
        float V = std::sqrt(s.vx * s.vx + s.vz * s.vz);
        if (V < 1e-6f) {
            M_max = 0.0f;
            t_switch = T_FLIP_TARGET / 2.0f;
            t_total = T_FLIP_TARGET;
            alpha_max = 0.0f;
            planned = true;
            flip_t = 0.0f;
            return;
        }

        AtmosphereResult atm = atmosphere(s.h);
        float M_mach = V / atm.a_sound;
        float Q = 0.5f * atm.rho * V * V * S_REF;
        M_max = Q * S_REF * L_REF * (C_DELTA_FWD + C_DELTA_AFT) * DELTA_MAX;

        float Iyy = get_Iyy(s.m_fuel);

        t_total = T_FLIP_TARGET;
        t_switch = t_total / 2.0f;

        float delta_theta = std::fabs(theta0 - thetaf);
        alpha_max = delta_theta / (t_switch * t_switch);

        float M_needed = alpha_max * Iyy;
        if (M_needed > M_max * M_MARGIN) {
            t_switch = std::sqrt(delta_theta / (M_max * M_MARGIN / Iyy));
            t_total = 2.0f * t_switch;
            alpha_max = delta_theta / (t_switch * t_switch);
        }

        planned = true;
        flip_t = 0.0f;
    }

    // 控制更新
    // 返回: (T, theta_cmd, delta_extra_fwd, delta_extra_aft)
    void control(const State& s, float dt,
                 float& T, float& theta_cmd,
                 float& delta_extra_fwd, float& delta_extra_aft) {
        if (!planned) return;

        flip_t += dt;

        // bang-bang期望轨迹
        BangBangTraj traj = bangbang_theta_trajectory(flip_t, theta0, thetaf, t_switch, t_total);
        float theta_ref = traj.theta;
        float theta_ref_dot = traj.theta_dot;

        // bang-bang参考加速度
        float theta_ref_ddot = bangbang_theta_acceleration(flip_t, t_switch, t_total, alpha_max);

        T = T_IDLE;

        float V = std::sqrt(s.vx * s.vx + s.vz * s.vz);
        if (V > 1e-6f) {
            float Iyy = get_Iyy(s.m_fuel);

            // 1. 标称前馈力矩
            float M_nominal = Iyy * theta_ref_ddot;

            // 2. PD反馈力矩
            float e_theta = theta_ref - s.theta;
            e_theta = std::fmod(e_theta + PI_F, 2.0f * PI_F) - PI_F;
            float e_q = theta_ref_dot - s.q;
            float M_pd = Iyy * (WN_TRACK * WN_TRACK * e_theta + 2.0f * ZETA_TRACK * WN_TRACK * e_q);

            // 3. 前馈力矩补偿
            float M_ff = compute_feedforward_torque(s, theta_ref);

            // 总额外襟翼
            float M_extra = M_nominal + M_pd + M_ff;
            AtmosphereResult atm = atmosphere(s.h);
            float Q = 0.5f * atm.rho * V * V * S_REF;
            float denom = Q * S_REF * L_REF * (C_DELTA_FWD + C_DELTA_AFT);
            float delta_extra = (denom > 0.0f) ? (M_extra / denom) : 0.0f;

            if (delta_extra > DELTA_MAX) delta_extra = DELTA_MAX;
            if (delta_extra < -DELTA_MAX) delta_extra = -DELTA_MAX;

            delta_extra_fwd = delta_extra;
            delta_extra_aft = delta_extra;
        } else {
            delta_extra_fwd = 0.0f;
            delta_extra_aft = 0.0f;
        }

        theta_cmd = theta_ref;
    }

    // 翻转完成判断 (theta归一化)
    bool is_complete(const State& s) const {
        float theta_norm = std::fmod(s.theta + PI_F, 2.0f * PI_F) - PI_F;
        float thetaf_norm = std::fmod(thetaf + PI_F, 2.0f * PI_F) - PI_F;
        float theta_err = std::fabs(theta_norm - thetaf_norm);
        theta_err = std::fmin(theta_err, 2.0f * PI_F - theta_err);

        bool q_small = std::fabs(s.q) < 0.05f;

        if (flip_t > t_total + 2.0f) return true;
        if (theta_err < 5.0f * DEG2RAD_F && q_small) return true;
        return false;
    }

    bool is_timeout() const {
        return flip_t > T_FLIP_MAX;
    }
};

}  // namespace belly_flop
}  // namespace falcon9

#endif  // FALCON9_BELLY_FLOP_FLIP_HPP
