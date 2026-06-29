// =============================================================================
// phase_controller_6dof.hpp - 星舰 6-DOF 三阶段控制器 (Phase 11 移植)
// =============================================================================
// 对应 Python: src/belly_flop/phase_controller_6dof.py
//
// 理论方案 7.0 问题 23: 状态机切换逻辑 BELLY → FLIP → LANDING
//
// 三阶段状态机 (6DOF 版, 14 维状态):
//   BELLY:   θ_cmd=85°(腹部朝下气动减速), T=0, 四元数 PD 维持 BELLY 姿态
//            切换条件: tgo≤15s AND V<250m/s AND h>3km
//   FLIP:    bang-bang+PD+前馈, θ 从 85°→0°, T=T_idle(保持 TVC 有效)
//            切换条件: θ<10° AND |q|<0.05rad/s (或翻转超时)
//   LANDING: θ_cmd=0°(垂直), T=bang-bang 匀减速, 四元数 PD 维持垂直
//            切换条件: h≤0 (着陆)
//
// 6DOF 关键设计 (与 3DOF 的区别):
//   1. 状态向量 14 维: [px,py,pz, vx,vy,vz, qw,qx,qy,qz, p,q,r, m_fuel]
//   2. 姿态用四元数, 期望姿态通过 euler_angle_to_quat 转换
//   3. 控制分配 4 片襟翼 (FL/FR/RL/RR), 物理推导禁 pinv
//   4. 非理想执行器: Bouc-Wen + 死区补偿 + 速率限制 + TVC 延迟
//   5. AttitudeController6DOF 计算力矩, allocate_flaps 分配襟翼
//
// 阶段切换平滑过渡 (问题 7):
//   - BELLY→FLIP: bang-bang 轨迹本身平滑, 无需额外斜坡
//   - FLIP→LANDING: 2 秒斜坡过渡 θ_cmd (防力矩跳变)
//
// Kill 条件:
//   - FLIP 后 h<800m: 着陆段不可行
//   - 翻转超时 > 8s
//   - 能量检查: a_needed > 0.7·a_avail (推力不足撞地)
// =============================================================================
#ifndef STARSHIP_BELLY_FLOP_6DOF_PHASE_CONTROLLER_HPP
#define STARSHIP_BELLY_FLOP_6DOF_PHASE_CONTROLLER_HPP

#include <cmath>
#include "aero_6dof.hpp"
#include "dynamics_6dof.hpp"
#include "control_allocation_6dof.hpp"
#include "attitude_control_6dof.hpp"
#include "actuator_nonideal_6dof.hpp"
#include "../core/fixed_matrix.hpp"
#include "../core/quaternion.hpp"

namespace starship {
namespace belly_flop_6dof {

// =============================================================================
// 阶段切换阈值 (复用 3DOF, 问题 23)
// =============================================================================
constexpr float TGO_FLIP_TRIGGER    = 15.0f;   // s, belly→flip 的 tgo 阈值
constexpr float V_FLIP_TRIGGER      = 250.0f;  // m/s, belly→flip 的速度阈值
constexpr float H_FLIP_MIN          = 3000.0f; // m, belly→flip 的最低高度
constexpr float THETA_LAND_TRIGGER  = 0.17453292519943295f;  // 10° in rad
constexpr float H_LAND_MIN          = 500.0f;  // m, flip→landing 的最低高度
constexpr float H_FLIP_KILL         = 800.0f;  // m, 翻转后 h<此值→Kill

// 翻转参数 (复用 3DOF)
constexpr float THETA_BELLY_DEG     = 85.0f;   // belly 阶段俯仰角 (度)
constexpr float THETA_LAND_DEG      = 0.0f;    // landing 阶段俯仰角 (度)
constexpr float T_FLIP_MAX          = 8.0f;    // s, 翻转超时 Kill 阈值
constexpr float T_FLIP_TARGET       = 3.5f;    // s, 目标翻转时间
constexpr float M_MARGIN            = 0.8f;    // 力矩裕度

// 斜坡过渡
constexpr float RAMP_TRANSITION     = 2.0f;    // s, 阶段切换斜坡过渡时间

// 能量检查
constexpr float ENERGY_KILL_RATIO   = 0.7f;    // a_needed > 0.7·a_avail → Kill

// 着陆参数
constexpr float VZ_LAND_TARGET      = 3.0f;    // m/s, 着陆目标垂直速度
constexpr float VZ_LAND_MAX         = 10.0f;   // m/s, 着陆最大垂直速度

// 角度转换
constexpr float DEG2RAD = 0.017453292519943295f;
constexpr float RAD2DEG = 57.29577951308232f;

// =============================================================================
// 阶段枚举
// =============================================================================
enum class Phase : int {
    BELLY   = 0,
    FLIP    = 1,
    LANDING = 2
};

// =============================================================================
// 诊断信息结构体 (替代 Python info dict, 零动态内存)
// =============================================================================
struct ControllerInfo {
    Phase phase;
    float V;             // 总速度 (m/s)
    float h;             // 高度 (m)
    float Mach;          // 马赫数
    float theta_deg;     // 当前俯仰角 (度)
    float theta_cmd_deg; // 指令俯仰角 (度)
    float t;             // 总时间 (s)
    float Q_dyn;         // 动压 (Pa)
    float T_cmd;         // 推力指令 (N)
    float tgo;           // 时间到着陆 (s, BELLY 阶段)
    float flip_t;        // 翻转时间 (s, FLIP 阶段)
    float M_cmd[3];      // 力矩指令 (N·m)
    float delta_flaps_cmd[4];   // 襟翼指令 (rad)
    float delta_flaps_actual[4]; // 襟翼实际 (rad)
    float tvc_gimbal_cmd[2];    // TVC 指令 (rad)
    float tvc_gimbal_actual[2]; // TVC 实际 (rad)
    float q_des[4];      // 期望四元数 [w,x,y,z]
    float omega_des[3];  // 期望角速度 (rad/s)
    bool  kill;          // 是否触发 Kill
    char  kill_reason[64]; // Kill 原因
    char  phase_transition[32]; // 阶段切换描述

    void reset() {
        phase = Phase::BELLY;
        V = h = Mach = theta_deg = theta_cmd_deg = t = Q_dyn = T_cmd = 0.0f;
        tgo = flip_t = 0.0f;
        for (int i = 0; i < 3; ++i) M_cmd[i] = omega_des[i] = 0.0f;
        for (int i = 0; i < 4; ++i) { delta_flaps_cmd[i] = delta_flaps_actual[i] = 0.0f; q_des[i] = 0.0f; }
        for (int i = 0; i < 2; ++i) { tvc_gimbal_cmd[i] = tvc_gimbal_actual[i] = 0.0f; }
        kill = false;
        kill_reason[0] = phase_transition[0] = '\0';
    }
};

// =============================================================================
// bang-bang 翻转轨迹辅助函数 (6DOF 版, 复用 3DOF 逻辑)
// =============================================================================

// 计算 6DOF 翻转段最大可用力矩
// M_max = Q·L·(Cδf+Cδa)·δ_max
inline float compute_max_flip_torque(const State6DOF& state, float& Q_out) {
    Vec3f vel_n = state.vel_n();
    float V = vel_n.norm();
    float h = -state[2];

    Q_out = 0.0f;
    if (V < 1e-6f || h < 0.0f) return 0.0f;

    AtmosphereResult atm = atmosphere(h);
    float Q = 0.5f * atm.rho * V * V * S_REF;
    Q_out = Q;
    return Q * L_REF * (C_DELTA_FWD + C_DELTA_AFT) * DELTA_MAX;
}

// 6DOF bang-bang 切换时间解析公式 (缺陷 21)
// 返回: (t_switch, t_total, alpha_max, M_max)
inline void compute_t_switch(float theta0_deg, float thetaf_deg,
                              const State6DOF& state, float m_fuel,
                              float t_target,
                              float& t_switch, float& t_total,
                              float& alpha_max, float& M_max) {
    float Q;
    M_max = compute_max_flip_torque(state, Q);
    InertiaTensor I = get_inertia_tensor(m_fuel);
    float Iyy = I.Iyy;  // 俯仰惯量

    t_total = t_target;
    t_switch = t_total * 0.5f;

    float delta_theta = std::fabs((theta0_deg - thetaf_deg) * DEG2RAD);
    alpha_max = (t_switch > 0.0f) ? (delta_theta / (t_switch * t_switch)) : 0.0f;

    float M_needed = alpha_max * Iyy;
    if (M_needed > M_max * M_MARGIN) {
        // 力矩不足, 延长翻转时间
        t_switch = std::sqrt(delta_theta / (M_max * M_MARGIN / Iyy));
        t_total = 2.0f * t_switch;
        alpha_max = delta_theta / (t_switch * t_switch);
    }
}

// 6DOF bang-bang 期望轨迹 (梯形角速度剖面)
// 返回: (theta_ref_deg, theta_ref_dot_rad_s)
inline void bangbang_theta_trajectory(float t, float theta0_deg, float thetaf_deg,
                                       float t_switch, float t_total,
                                       float& theta_ref_deg, float& theta_ref_dot) {
    float delta_theta = (theta0_deg - thetaf_deg) * DEG2RAD;  // 正值

    if (t_total < 1e-6f) {
        theta_ref_deg = thetaf_deg;
        theta_ref_dot = 0.0f;
        return;
    }

    float alpha = (t_switch > 0.0f) ? (delta_theta / (t_switch * t_switch)) : 0.0f;

    if (t < 0.0f) {
        theta_ref_deg = theta0_deg;
        theta_ref_dot = 0.0f;
    } else if (t < t_switch) {
        // 加速阶段
        float theta = theta0_deg * DEG2RAD - 0.5f * alpha * t * t;
        theta_ref_deg = theta * RAD2DEG;
        theta_ref_dot = -alpha * t;
    } else if (t < t_total) {
        // 减速阶段
        float t_dec = t - t_switch;
        float theta_mid = theta0_deg * DEG2RAD - 0.5f * alpha * t_switch * t_switch;
        float theta = theta_mid - alpha * t_switch * t_dec + 0.5f * alpha * t_dec * t_dec;
        theta_ref_deg = theta * RAD2DEG;
        theta_ref_dot = -alpha * t_switch + alpha * t_dec;
    } else {
        theta_ref_deg = thetaf_deg;
        theta_ref_dot = 0.0f;
    }
}

// 6DOF bang-bang 参考角加速度 (标称前馈)
inline float bangbang_theta_acceleration(float t, float t_switch,
                                          float t_total, float alpha_max) {
    if (t < 0.0f || t >= t_total) return 0.0f;
    if (t < t_switch) return -alpha_max;
    return alpha_max;
}

// =============================================================================
// 6DOF 三阶段控制器: BELLY → FLIP → LANDING
// =============================================================================
class PhaseController6DOF {
public:
    // 子系统
    AttitudeController6DOF attitude_ctrl;
    FlapActuatorSuite6DOF flap_actuator;
    TVCActuator6DOF tvc_actuator;
    bool use_nonideal_actuator;

    // 阶段状态
    Phase phase;
    float phase_t;
    float total_t;

    // 斜坡过渡
    float theta_cmd_current_deg;
    float theta_cmd_target_deg;
    bool  ramp_active;
    float ramp_start_deg;
    float ramp_end_deg;
    float ramp_t;

    // FLIP 轨迹规划
    bool  flip_planned;
    float flip_t;
    float flip_t_switch;
    float flip_t_total;
    float flip_alpha_max;
    float flip_M_max;
    float flip_theta0_deg;
    float flip_thetaf_deg;

    // LANDING 计时
    float landing_t;

    // 诊断
    ControllerInfo last_info;

    PhaseController6DOF(bool use_nonideal_actuator_ = true,
                        bool use_notch = true,
                        float wn = 2.0f * 3.14159265358979f * 0.5f,
                        float zeta = 0.9f)
        : attitude_ctrl(wn, zeta, 100.0f, use_notch),
          flap_actuator(),
          tvc_actuator(0.08f, 20.0f * DEG2RAD, 10.0f * DEG2RAD, 0.01f),
          use_nonideal_actuator(use_nonideal_actuator_),
          phase(Phase::BELLY), phase_t(0.0f), total_t(0.0f),
          theta_cmd_current_deg(THETA_BELLY_DEG),
          theta_cmd_target_deg(THETA_BELLY_DEG),
          ramp_active(false),
          ramp_start_deg(THETA_BELLY_DEG), ramp_end_deg(THETA_BELLY_DEG),
          ramp_t(0.0f),
          flip_planned(false), flip_t(0.0f),
          flip_t_switch(0.0f), flip_t_total(0.0f),
          flip_alpha_max(0.0f), flip_M_max(0.0f),
          flip_theta0_deg(THETA_BELLY_DEG), flip_thetaf_deg(THETA_LAND_DEG),
          landing_t(0.0f) {
        last_info.reset();
    }

    void reset() {
        phase = Phase::BELLY;
        phase_t = 0.0f;
        total_t = 0.0f;
        theta_cmd_current_deg = THETA_BELLY_DEG;
        theta_cmd_target_deg = THETA_BELLY_DEG;
        ramp_active = false;
        ramp_t = 0.0f;
        flip_planned = false;
        flip_t = 0.0f;
        landing_t = 0.0f;
        attitude_ctrl.reset();
        if (use_nonideal_actuator) {
            flap_actuator.reset();
            tvc_actuator.reset();
        }
        last_info.reset();
    }

    // ============ 辅助函数 ============

    // tgo = (h - H_LAND_MIN) / |vz|
    float compute_tgo(const State6DOF& state) const {
        float h = -state[2];
        float vz = state[5];
        if (std::fabs(vz) < 1.0f) return 999.0f;
        float h_to_land = h - H_LAND_MIN;
        if (h_to_land <= 0.0f) return 0.0f;
        return h_to_land / std::fabs(vz);
    }

    // 能量检查 (缺陷 9)
    bool energy_check(const State6DOF& state, char* reason_buf, int buf_size) const {
        Vec3f vel_n = state.vel_n();
        float V = vel_n.norm();
        float h = -state[2];
        float m_fuel = state.m_fuel();
        float m = get_mass(m_fuel);
        float g = gravity(h);

        if (h < 1.0f) return false;

        float a_needed = V * V / (2.0f * h);
        float a_avail = T_MAX / m - g;
        if (a_avail <= 0.0f) {
            if (reason_buf) snprintf(reason_buf, buf_size,
                "no_thrust_authority (a_avail=%.1f<=0)", a_avail);
            return true;
        }

        float ratio = a_needed / a_avail;
        if (ratio > ENERGY_KILL_RATIO) {
            if (reason_buf) snprintf(reason_buf, buf_size,
                "insufficient_thrust (a_needed=%.1f, a_avail=%.1f, ratio=%.2f>%d%%)",
                a_needed, a_avail, ratio, (int)(ENERGY_KILL_RATIO * 100));
            return true;
        }
        return false;
    }

    // 2 秒斜坡过渡 (缺陷 7)
    void ramp_update(float dt) {
        if (ramp_active) {
            ramp_t += dt;
            float alpha = ramp_t / RAMP_TRANSITION;
            if (alpha > 1.0f) alpha = 1.0f;
            theta_cmd_current_deg = ramp_start_deg +
                alpha * (ramp_end_deg - ramp_start_deg);
            if (alpha >= 1.0f) {
                ramp_active = false;
                theta_cmd_current_deg = ramp_end_deg;
            }
        } else {
            theta_cmd_current_deg = theta_cmd_target_deg;
        }
    }

    void start_ramp(float new_target_deg) {
        if (std::fabs(new_target_deg - theta_cmd_current_deg) < 1e-6f) {
            theta_cmd_current_deg = new_target_deg;
            theta_cmd_target_deg = new_target_deg;
            ramp_active = false;
        } else {
            ramp_start_deg = theta_cmd_current_deg;
            ramp_end_deg = new_target_deg;
            theta_cmd_target_deg = new_target_deg;
            ramp_t = 0.0f;
            ramp_active = true;
        }
    }

    float get_pitch_angle_deg(const State6DOF& state) const {
        Quaternion q = state.q();
        float theta_rad = get_pitch_angle_from_quat(q);
        return theta_rad * RAD2DEG;
    }

    // 计算动压和马赫数
    void compute_Q_dyn(const State6DOF& state, float& Q, float& M_mach) const {
        Vec3f vel_n = state.vel_n();
        float V = vel_n.norm();
        float h = -state[2];
        Q = 0.0f; M_mach = 0.0f;
        if (V < 1e-6f || h < 0.0f) return;
        AtmosphereResult atm = atmosphere(h);
        Q = 0.5f * atm.rho * V * V * S_REF;
        M_mach = (atm.a_sound > 0.0f) ? (V / atm.a_sound) : 0.0f;
    }

    // ============ 各阶段控制律 ============

    // BELLY 阶段: θ_cmd=85°, T=0
    void belly_control(float& T_cmd, float& theta_cmd_target_deg) const {
        T_cmd = 0.0f;
        theta_cmd_target_deg = THETA_BELLY_DEG;
    }

    // 规划 FLIP bang-bang 轨迹
    void flip_plan(const State6DOF& state) {
        float m_fuel = state.m_fuel();
        float theta_current = get_pitch_angle_deg(state);
        flip_theta0_deg = theta_current;
        flip_thetaf_deg = THETA_LAND_DEG;

        compute_t_switch(flip_theta0_deg, flip_thetaf_deg,
                         state, m_fuel, T_FLIP_TARGET,
                         flip_t_switch, flip_t_total,
                         flip_alpha_max, flip_M_max);
        flip_planned = true;
        flip_t = 0.0f;
    }

    // FLIP 阶段: bang-bang+PD+前馈, T=T_idle
    void flip_control(float dt, float& T_cmd, float& theta_cmd_deg,
                      float& theta_ref_dot, float& theta_ref_ddot) {
        flip_t += dt;

        bangbang_theta_trajectory(flip_t, flip_theta0_deg, flip_thetaf_deg,
                                   flip_t_switch, flip_t_total,
                                   theta_cmd_deg, theta_ref_dot);
        theta_ref_ddot = bangbang_theta_acceleration(
            flip_t, flip_t_switch, flip_t_total, flip_alpha_max);

        T_cmd = T_IDLE;
    }

    // LANDING 阶段: θ_cmd=0°, T=bang-bang 匀减速
    void landing_control(const State6DOF& state,
                          float& T_cmd, float& theta_cmd_target_deg) const {
        Vec3f vel_n = state.vel_n();
        float vx = vel_n[0];
        float vz = vel_n[2];
        float h = -state[2];
        float m_fuel = state.m_fuel();
        float m = get_mass(m_fuel);
        float g = gravity(h);

        // 上升时关机
        if (vz < -1.0f) {
            T_cmd = 0.0f;
            theta_cmd_target_deg = THETA_LAND_DEG;
            return;
        }

        // 下降时匀减速剖面
        float h_eff = (h > 1.0f) ? h : 1.0f;
        float a_brake = (vz * vz - VZ_LAND_TARGET * VZ_LAND_TARGET) / (2.0f * h_eff);
        float T_needed = m * (g + a_brake);
        T_cmd = T_needed;
        if (T_cmd < 0.0f) T_cmd = 0.0f;
        if (T_cmd > T_MAX) T_cmd = T_MAX;

        // 水平速度阻尼: θ_cmd 向 vx 反方向倾斜, 限幅 ±10°
        if (std::fabs(vz) > 1.0f) {
            float tilt = -0.5f * std::atan2(vx, std::fabs(vz));
            if (tilt >  10.0f * DEG2RAD) tilt =  10.0f * DEG2RAD;
            if (tilt < -10.0f * DEG2RAD) tilt = -10.0f * DEG2RAD;
            theta_cmd_target_deg = tilt * RAD2DEG;
        } else {
            theta_cmd_target_deg = THETA_LAND_DEG;
        }
    }

    // 翻转完成判断: θ<10° AND |q|<0.05rad/s
    bool is_flip_complete(const State6DOF& state) const {
        float theta_deg = get_pitch_angle_deg(state);
        Vec3f omega_b = state.omega_b();
        float q_rate = omega_b[1];  // 俯仰角速度

        float theta_err = std::fabs((theta_deg - THETA_LAND_DEG) * DEG2RAD);
        bool q_small = std::fabs(q_rate) < 0.05f;

        if (flip_t > flip_t_total + 2.0f) return true;
        if (theta_err < THETA_LAND_TRIGGER && q_small) return true;
        return false;
    }

    // =========================================================================
    // 主控制循环
    // =========================================================================
    // state: 14 维状态
    // dt: 时间步长 (s)
    // info: 诊断信息输出
    // 返回: (T_cmd, delta_flaps_actual[4], tvc_gimbal_actual[2], kill)
    // =========================================================================
    bool update(const State6DOF& state, float dt,
                 float& T_cmd_out, float delta_flaps_actual[4],
                 float tvc_gimbal_actual[2], ControllerInfo& info) {
        total_t += dt;
        phase_t += dt;

        // 提取状态
        Vec3f vel_n = state.vel_n();
        float V = vel_n.norm();
        float h = -state[2];
        float m_fuel = state.m_fuel();
        float theta_deg = get_pitch_angle_deg(state);
        float Q_dyn, M_mach;
        compute_Q_dyn(state, Q_dyn, M_mach);

        // 初始化 info
        info.reset();
        info.phase = phase;
        info.V = V;
        info.h = h;
        info.Mach = M_mach;
        info.theta_deg = theta_deg;
        info.t = total_t;
        info.Q_dyn = Q_dyn;
        info.kill = false;

        // ============ 阶段切换 ============
        if (phase == Phase::BELLY) {
            float tgo = compute_tgo(state);
            if (tgo <= TGO_FLIP_TRIGGER && V < V_FLIP_TRIGGER && h > H_FLIP_MIN) {
                phase = Phase::FLIP;
                phase_t = 0.0f;
                flip_plan(state);
                snprintf(info.phase_transition, sizeof(info.phase_transition),
                         "BELLY->FLIP");
            }
        } else if (phase == Phase::FLIP) {
            // 翻转完成检查
            if (flip_planned && is_flip_complete(state)) {
                phase = Phase::LANDING;
                phase_t = 0.0f;
                landing_t = 0.0f;
                start_ramp(THETA_LAND_DEG);
                snprintf(info.phase_transition, sizeof(info.phase_transition),
                         "FLIP->LANDING");
            }

            // Kill: 翻转后 h < 800m
            if (h < H_FLIP_KILL) {
                T_cmd_out = 0.0f;
                for (int i = 0; i < 4; ++i) delta_flaps_actual[i] = 0.0f;
                for (int i = 0; i < 2; ++i) tvc_gimbal_actual[i] = 0.0f;
                info.kill = true;
                snprintf(info.kill_reason, sizeof(info.kill_reason),
                         "flip_too_low (h=%.0fm<%.0fm)", h, H_FLIP_KILL);
                last_info = info;
                return true;
            }

            // Kill: 翻转超时
            if (flip_t > T_FLIP_MAX) {
                T_cmd_out = 0.0f;
                for (int i = 0; i < 4; ++i) delta_flaps_actual[i] = 0.0f;
                for (int i = 0; i < 2; ++i) tvc_gimbal_actual[i] = 0.0f;
                info.kill = true;
                snprintf(info.kill_reason, sizeof(info.kill_reason),
                         "flip_timeout (t=%.1fs>%.1fs)", flip_t, T_FLIP_MAX);
                last_info = info;
                return true;
            }
        } else if (phase == Phase::LANDING) {
            landing_t += dt;
        }

        // ============ 能量检查 ============
        if (phase == Phase::FLIP ||
            (phase == Phase::LANDING && h > 200.0f)) {
            char reason[64];
            if (energy_check(state, reason, sizeof(reason))) {
                T_cmd_out = 0.0f;
                for (int i = 0; i < 4; ++i) delta_flaps_actual[i] = 0.0f;
                for (int i = 0; i < 2; ++i) tvc_gimbal_actual[i] = 0.0f;
                info.kill = true;
                snprintf(info.kill_reason, sizeof(info.kill_reason), "%s", reason);
                last_info = info;
                return true;
            }
        }

        // ============ 各阶段控制律 ============
        float T_cmd = 0.0f;
        float theta_cmd_deg = THETA_BELLY_DEG;
        float omega_des_arr[3] = {0.0f, 0.0f, 0.0f};
        float omega_des_dot_arr[3] = {0.0f, 0.0f, 0.0f};

        if (phase == Phase::BELLY) {
            float theta_cmd_target;
            belly_control(T_cmd, theta_cmd_target);
            if (std::fabs(theta_cmd_target - theta_cmd_target_deg) > 1e-6f &&
                !ramp_active) {
                start_ramp(theta_cmd_target);
            }
            ramp_update(dt);
            theta_cmd_deg = theta_cmd_current_deg;
            // omega_des = 0 (已初始化)

        } else if (phase == Phase::FLIP) {
            float theta_ref_dot, theta_ref_ddot;
            flip_control(dt, T_cmd, theta_cmd_deg, theta_ref_dot, theta_ref_ddot);
            // 期望角速度: 只有俯仰分量
            omega_des_arr[1] = theta_ref_dot;
            // 期望角加速度 (前馈): 只有俯仰分量
            omega_des_dot_arr[1] = theta_ref_ddot;
            theta_cmd_current_deg = theta_cmd_deg;
            theta_cmd_target_deg = theta_cmd_deg;
            ramp_active = false;

        } else {  // LANDING
            float theta_cmd_target;
            landing_control(state, T_cmd, theta_cmd_target);
            if (std::fabs(theta_cmd_target - theta_cmd_target_deg) > 1e-6f &&
                !ramp_active) {
                start_ramp(theta_cmd_target);
            }
            ramp_update(dt);
            theta_cmd_deg = theta_cmd_current_deg;
            // omega_des = 0 (已初始化)
        }

        // ============ 姿态控制器 ============
        Quaternion q_actual = state.q();
        Vec3f omega_actual = state.omega_b();   // ← 前项目教训: 从 state 读取
        InertiaTensor I = get_inertia_tensor(m_fuel);

        // 期望四元数
        Quaternion q_des = euler_angle_to_quat(theta_cmd_deg);

        // 期望角速度/角加速度向量
        Vec3f omega_des;
        omega_des[0] = omega_des_arr[0]; omega_des[1] = omega_des_arr[1]; omega_des[2] = omega_des_arr[2];
        Vec3f omega_des_dot;
        omega_des_dot[0] = omega_des_dot_arr[0];
        omega_des_dot[1] = omega_des_dot_arr[1];
        omega_des_dot[2] = omega_des_dot_arr[2];

        // 计算力矩指令
        float M_cmd[3], delta_flaps_cmd[4], tvc_gimbal_cmd[2];
        attitude_ctrl.compute_torque(
            q_des, omega_des, q_actual, omega_actual,
            I, Q_dyn, &omega_des_dot,
            M_cmd, delta_flaps_cmd, tvc_gimbal_cmd);

        // ============ 非理想执行器 ============
        if (use_nonideal_actuator) {
            flap_actuator.update(delta_flaps_cmd, dt, delta_flaps_actual);
            tvc_actuator.update(tvc_gimbal_cmd, dt, tvc_gimbal_actual);
        } else {
            for (int i = 0; i < 4; ++i) delta_flaps_actual[i] = delta_flaps_cmd[i];
            for (int i = 0; i < 2; ++i) tvc_gimbal_actual[i] = tvc_gimbal_cmd[i];
        }

        // ============ 诊断信息 ============
        info.T_cmd = T_cmd;
        info.theta_cmd_deg = theta_cmd_deg;
        for (int i = 0; i < 3; ++i) {
            info.M_cmd[i] = M_cmd[i];
            info.omega_des[i] = omega_des[i];
        }
        for (int i = 0; i < 4; ++i) {
            info.delta_flaps_cmd[i] = delta_flaps_cmd[i];
            info.delta_flaps_actual[i] = delta_flaps_actual[i];
        }
        for (int i = 0; i < 2; ++i) {
            info.tvc_gimbal_cmd[i] = tvc_gimbal_cmd[i];
            info.tvc_gimbal_actual[i] = tvc_gimbal_actual[i];
        }
        info.q_des[0] = q_des.w; info.q_des[1] = q_des.x;
        info.q_des[2] = q_des.y; info.q_des[3] = q_des.z;
        info.tgo = (phase == Phase::BELLY) ? compute_tgo(state) : 0.0f;
        info.flip_t = (phase == Phase::FLIP) ? flip_t : 0.0f;

        T_cmd_out = T_cmd;
        last_info = info;
        return false;  // kill = false
    }
};

}  // namespace belly_flop_6dof
}  // namespace starship

#endif  // STARSHIP_BELLY_FLOP_6DOF_PHASE_CONTROLLER_HPP
