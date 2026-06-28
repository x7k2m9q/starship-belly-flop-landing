// =============================================================================
// belly_flop_integrated.hpp - 全程集成控制器 (C++翻译, Step 7E)
// =============================================================================
// 暗礁23: 统一状态结构体, 三阶段切换只改控制器不重置状态
// BELLY → FLIP → LANDING
//
// 对应 Python: src/belly_flop/integrated_controller.py
// =============================================================================
#ifndef FALCON9_BELLY_FLOP_INTEGRATED_HPP
#define FALCON9_BELLY_FLOP_INTEGRATED_HPP

#include "belly_flop_aero.hpp"
#include "belly_flop_dynamics.hpp"
#include "belly_flop_flip.hpp"
#include <cmath>
#include <cstdio>

namespace falcon9 {
namespace belly_flop {

// =============================================================================
// 阶段切换阈值 (与 Python controller.py 一致)
// =============================================================================
constexpr float TGO_FLIP_TRIGGER    = 15.0f;   // s
constexpr float V_FLIP_TRIGGER      = 250.0f;  // m/s
constexpr float H_FLIP_MIN          = 3000.0f; // m
constexpr float ALPHA_LAND_TRIGGER  = 0.174532925199f;  // 10° in rad
constexpr float H_LAND_MIN          = 500.0f;  // m
constexpr float H_FLIP_KILL         = 800.0f;  // m
constexpr float ENERGY_KILL_RATIO   = 0.7f;
constexpr float VZ_LAND_TARGET      = 3.0f;    // m/s
constexpr float RAMP_TRANSITION     = 2.0f;    // s

// =============================================================================
// 控制输出
// =============================================================================
struct ControlOutput7E {
    float T;
    float theta_cmd;
    float delta_extra_fwd;
    float delta_extra_aft;
    const char* phase;  // "BELLY", "FLIP", "LANDING"
    bool kill;
    char kill_reason[128];
};

// =============================================================================
// PD增益调度 (Mach分3段)
// =============================================================================
inline void pd_gains(float M, float& Kp, float& Kd) {
    if (M > 1.2f) {
        Kp = 0.80f; Kd = 0.40f;
    } else if (M > 0.8f) {
        Kp = 1.20f; Kd = 0.60f;
    } else {
        Kp = 0.80f; Kd = 0.40f;
    }
}

// =============================================================================
// 全程集成控制器
// =============================================================================
class IntegratedBellyFlopController {
public:
    const char* phase;
    float phase_t;
    float theta_cmd_current;
    float theta_cmd_target;
    bool ramp_active;
    float ramp_start;
    float ramp_end;
    float ramp_t;

    FlipController flip_ctrl;
    bool flip_initialized;
    float landing_t;

    IntegratedBellyFlopController()
        : phase("BELLY"), phase_t(0.0f),
          theta_cmd_current(THETA_BELLY_F), theta_cmd_target(THETA_BELLY_F),
          ramp_active(false), ramp_start(THETA_BELLY_F), ramp_end(THETA_BELLY_F),
          ramp_t(0.0f), flip_initialized(false), landing_t(0.0f) {}

    float compute_tgo(const State& s) {
        if (std::fabs(s.vz) < 1.0f) return 999.0f;
        float h_to_land = s.h - H_LAND_MIN;
        if (h_to_land <= 0.0f) return 0.0f;
        return h_to_land / std::fabs(s.vz);
    }

    bool energy_check(const State& s, char* reason) {
        float V = std::sqrt(s.vx * s.vx + s.vz * s.vz);
        float m = get_mass(s.m_fuel);
        float g = gravity(s.h);

        if (s.h < 1.0f) return false;

        float a_needed = V * V / (2.0f * s.h);
        float a_avail = T_MAX / m - g;
        if (a_avail <= 0.0f) {
            snprintf(reason, 128, "no_thrust_authority (a_avail=%.1f<=0)", a_avail);
            return true;
        }

        float ratio = a_needed / a_avail;
        if (ratio > ENERGY_KILL_RATIO) {
            snprintf(reason, 128, "insufficient_thrust (ratio=%.2f>%d)", ratio, ENERGY_KILL_RATIO);
            return true;
        }
        return false;
    }

    void start_ramp(float new_target) {
        if (std::fabs(new_target - theta_cmd_current) < 1e-6f) {
            theta_cmd_current = new_target;
            theta_cmd_target = new_target;
            ramp_active = false;
        } else {
            ramp_start = theta_cmd_current;
            ramp_end = new_target;
            theta_cmd_target = new_target;
            ramp_t = 0.0f;
            ramp_active = true;
        }
    }

    void ramp_update(float dt) {
        if (ramp_active) {
            ramp_t += dt;
            float alpha = std::fmin(1.0f, ramp_t / RAMP_TRANSITION);
            theta_cmd_current = ramp_start + alpha * (ramp_end - ramp_start);
            if (alpha >= 1.0f) {
                ramp_active = false;
                theta_cmd_current = ramp_end;
            }
        } else {
            theta_cmd_current = theta_cmd_target;
        }
    }

    ControlOutput7E update(const State& s, float dt) {
        ControlOutput7E out{};
        out.kill = false;
        out.kill_reason[0] = '\0';

        float V = std::sqrt(s.vx * s.vx + s.vz * s.vz);
        AlphaGamma ag = angle_of_attack(s.theta, s.vx, s.vz);
        AtmosphereResult atm = atmosphere(s.h);
        float M = (atm.a_sound > 0.0f) ? (V / atm.a_sound) : 0.0f;

        phase_t += dt;

        // ============ 阶段切换 ============
        if (phase == "BELLY") {
            float tgo = compute_tgo(s);
            if (tgo <= TGO_FLIP_TRIGGER && V < V_FLIP_TRIGGER && s.h > H_FLIP_MIN) {
                phase = "FLIP";
                phase_t = 0.0f;
                // 暗礁23: 从当前theta初始化FlipController
                flip_ctrl = FlipController(s.theta, THETA_LAND_F);
                flip_ctrl.plan(s);
                flip_initialized = true;
            }
        } else if (phase == "FLIP") {
            if (flip_ctrl.is_complete(s)) {
                phase = "LANDING";
                phase_t = 0.0f;
                landing_t = 0.0f;
                start_ramp(THETA_LAND_F);
            }

            if (s.h < H_FLIP_KILL) {
                out.kill = true;
                snprintf(out.kill_reason, 128, "flip_too_low (h=%.0fm<%.0fm)", s.h, H_FLIP_KILL);
                out.phase = phase;
                return out;
            }

            if (flip_ctrl.is_timeout()) {
                out.kill = true;
                snprintf(out.kill_reason, 128, "flip_timeout (t=%.1fs>%.1fs)", flip_ctrl.flip_t, T_FLIP_MAX);
                out.phase = phase;
                return out;
            }
        } else if (phase == "LANDING") {
            landing_t += dt;
        }

        // ============ 能量检查 ============
        if (phase == "FLIP" || (phase == "LANDING" && s.h > 200.0f)) {
            char reason[128];
            if (energy_check(s, reason)) {
                out.kill = true;
                snprintf(out.kill_reason, 128, "%s", reason);
                out.phase = phase;
                return out;
            }
        }

        // ============ 各阶段控制律 ============
        if (phase == "BELLY") {
            out.T = 0.0f;
            float theta_cmd_target = THETA_BELLY_F;

            if (std::fabs(theta_cmd_target - theta_cmd_target) > 1e-6f && !ramp_active) {
                start_ramp(theta_cmd_target);
            }
            ramp_update(dt);
            out.theta_cmd = theta_cmd_current;

            // PD阻尼
            float Kp, Kd;
            pd_gains(M, Kp, Kd);
            float e_theta = out.theta_cmd - s.theta;
            e_theta = std::fmod(e_theta + PI_F, 2.0f * PI_F) - PI_F;
            float delta_extra = Kp * e_theta - Kd * s.q;
            out.delta_extra_fwd = delta_extra;
            out.delta_extra_aft = delta_extra;

        } else if (phase == "FLIP") {
            flip_ctrl.control(s, dt, out.T, out.theta_cmd, out.delta_extra_fwd, out.delta_extra_aft);
            theta_cmd_current = out.theta_cmd;
            theta_cmd_target = out.theta_cmd;
            ramp_active = false;

        } else {  // LANDING
            float m = get_mass(s.m_fuel);
            float g = gravity(s.h);

            if (s.vz < -1.0f) {
                out.T = 0.0f;
                out.theta_cmd = THETA_LAND_F;
            } else {
                float h_eff = (s.h > 1.0f) ? s.h : 1.0f;
                float a_brake = (s.vz * s.vz - VZ_LAND_TARGET * VZ_LAND_TARGET) / (2.0f * h_eff);
                float T_needed = m * (g + a_brake);
                out.T = (T_needed > T_MAX) ? T_MAX : ((T_needed < 0.0f) ? 0.0f : T_needed);

                if (std::fabs(s.vz) > 1.0f) {
                    float theta_cmd_target = -0.5f * std::atan2(s.vx, std::fabs(s.vz));
                    if (theta_cmd_target > 10.0f * DEG2RAD_F) theta_cmd_target = 10.0f * DEG2RAD_F;
                    if (theta_cmd_target < -10.0f * DEG2RAD_F) theta_cmd_target = -10.0f * DEG2RAD_F;
                    out.theta_cmd = theta_cmd_target;
                } else {
                    out.theta_cmd = THETA_LAND_F;
                }
            }

            if (std::fabs(out.theta_cmd - theta_cmd_target) > 1e-6f && !ramp_active) {
                start_ramp(out.theta_cmd);
            }
            ramp_update(dt);
            out.theta_cmd = theta_cmd_current;

            // PD阻尼
            float Kp, Kd;
            pd_gains(M, Kp, Kd);
            float e_theta = out.theta_cmd - s.theta;
            e_theta = std::fmod(e_theta + PI_F, 2.0f * PI_F) - PI_F;
            float delta_extra = Kp * e_theta - Kd * s.q;
            out.delta_extra_fwd = delta_extra;
            out.delta_extra_aft = delta_extra;
        }

        out.phase = phase;
        return out;
    }
};

}  // namespace belly_flop
}  // namespace falcon9

#endif  // FALCON9_BELLY_FLOP_INTEGRATED_HPP
