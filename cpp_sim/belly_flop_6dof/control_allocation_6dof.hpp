// =============================================================================
// control_allocation_6dof.hpp - 星舰 6-DOF 4 片襟翼控制分配 (Phase 11 移植)
// =============================================================================
// 对应 Python: src/belly_flop/control_allocation_6dof.py
//
// 理论方案 7.0 问题 19: 控制分配矩阵病态
//   - 4 片襟翼控制 3 个力矩通道 (俯仰/偏航/滚转), 存在 1 维零空间
//   - 严禁使用 pinv 伪逆 (猎鹰 9 号 Bug 教训: 37 倍力矩低估)
//   - 必须基于物理推导直接分配矩阵
//
// 物理推导 (从 aero_6dof.hpp 的力矩公式反推):
//   气动力矩公式 (与 aero_forces_and_moments 一致):
//     Mx = k_roll · (d_FL - d_FR - d_RL + d_RR)        // 滚转
//     My = k_fwd  · (d_FL + d_FR) - k_aft · (d_RL + d_RR)  // 俯仰
//     Mz = k_yaw  · (d_FL - d_FR + d_RL - d_RR)        // 偏航
//   其中:
//     k_fwd  = Q·L·C_DELTA_FWD   (前翼效率)
//     k_aft  = Q·L·C_DELTA_AFT   (后翼效率)
//     k_roll = Q·L·C_DELTA_ROLL  (滚转效率)
//     k_yaw  = Q·L·C_DELTA_YAW   (偏航效率)
//
//   设: a=d_FL+d_FR, b=d_RL+d_RR, c=d_FL-d_FR, d=d_RL-d_RR
//   则:
//     Mx/k_roll = c - d
//     Mz/k_yaw  = c + d
//     My = k_fwd·a - k_aft·b
//
//   解得 (等力矩分配: 前后翼各承担 My/2):
//     c = (Mx/k_roll + Mz/k_yaw) / 2
//     d = (Mz/k_yaw - Mx/k_roll) / 2
//     a = My / (2·k_fwd)
//     b = -My / (2·k_aft)
//
//   最终:
//     d_FL =  My/(4·k_fwd)  + Mx/(4·k_roll) + Mz/(4·k_yaw)
//     d_FR =  My/(4·k_fwd)  - Mx/(4·k_roll) - Mz/(4·k_yaw)
//     d_RL = -My/(4·k_aft)  - Mx/(4·k_roll) + Mz/(4·k_yaw)
//     d_RR = -My/(4·k_aft)  + Mx/(4·k_roll) - Mz/(4·k_yaw)
//
// 符号验证 (与方案 7.0 的 D 矩阵一致):
//   FL: My+, Mx+, Mz+ (俯仰+, 滚转+, 偏航+)
//   FR: My+, Mx-, Mz- (俯仰+, 滚转-, 偏航-)
//   RL: My-, Mx-, Mz+ (俯仰-, 滚转-, 偏航+)
//   RR: My-, Mx+, Mz- (俯仰-, 滚转+, 偏航-)
// =============================================================================
#ifndef FALCON9_BELLY_FLOP_6DOF_CONTROL_ALLOCATION_HPP
#define FALCON9_BELLY_FLOP_6DOF_CONTROL_ALLOCATION_HPP

#include <cmath>
#include "aero_6dof.hpp"

namespace falcon9 {
namespace belly_flop_6dof {

// =============================================================================
// 4 片襟翼控制分配 (物理推导, 禁 pinv)
// =============================================================================
// 参数:
//   M_cmd: [Mx, My, Mz] 期望力矩 (N·m), body 系
//     Mx: 滚转力矩 (绕 Xb)
//     My: 俯仰力矩 (绕 Yb)
//     Mz: 偏航力矩 (绕 Zb)
//   Q: 动压 (Pa) = 0.5·ρ·V²·S_REF
//   delta_out: [d_FL, d_FR, d_RL, d_RR] (rad), 已钳位到 ±DELTA_MAX
// =============================================================================
inline void allocate_flaps(const float M_cmd[3], float Q,
                            float delta_out[4]) {
    float Mx = M_cmd[0], My = M_cmd[1], Mz = M_cmd[2];

    // 效率系数 (力矩 = k · δ)
    float k_fwd  = Q * L_REF * C_DELTA_FWD;
    float k_aft  = Q * L_REF * C_DELTA_AFT;
    float k_roll = Q * L_REF * C_DELTA_ROLL;
    float k_yaw  = Q * L_REF * C_DELTA_YAW;

    // 防除零 (动压极低时仍保持数值稳定)
    const float eps = 1e-10f;
    if (std::fabs(k_fwd)  < eps) k_fwd  = eps;
    if (std::fabs(k_aft)  < eps) k_aft  = eps;
    if (std::fabs(k_roll) < eps) k_roll = eps;
    if (std::fabs(k_yaw)  < eps) k_yaw  = eps;

    // 物理推导的分配公式
    float d_FL =  My / (4.0f * k_fwd)  + Mx / (4.0f * k_roll) + Mz / (4.0f * k_yaw);
    float d_FR =  My / (4.0f * k_fwd)  - Mx / (4.0f * k_roll) - Mz / (4.0f * k_yaw);
    float d_RL = -My / (4.0f * k_aft)  - Mx / (4.0f * k_roll) + Mz / (4.0f * k_yaw);
    float d_RR = -My / (4.0f * k_aft)  + Mx / (4.0f * k_roll) - Mz / (4.0f * k_yaw);

    // 钳位到 ±DELTA_MAX
    if (d_FL >  DELTA_MAX) d_FL =  DELTA_MAX;
    if (d_FL < -DELTA_MAX) d_FL = -DELTA_MAX;
    if (d_FR >  DELTA_MAX) d_FR =  DELTA_MAX;
    if (d_FR < -DELTA_MAX) d_FR = -DELTA_MAX;
    if (d_RL >  DELTA_MAX) d_RL =  DELTA_MAX;
    if (d_RL < -DELTA_MAX) d_RL = -DELTA_MAX;
    if (d_RR >  DELTA_MAX) d_RR =  DELTA_MAX;
    if (d_RR < -DELTA_MAX) d_RR = -DELTA_MAX;

    delta_out[0] = d_FL;
    delta_out[1] = d_FR;
    delta_out[2] = d_RL;
    delta_out[3] = d_RR;
}

// =============================================================================
// 归一化分配: 如果任一襟翼超限, 等比缩小所有襟翼保持方向
// =============================================================================
// 用于避免钳位导致的力矩方向偏转.
// 物理意义: 4 片襟翼同步缩放, 保持合力矩方向不变, 仅幅度减小.
// =============================================================================
inline void allocate_flaps_normalized(const float M_cmd[3], float Q,
                                       float delta_out[4]) {
    float Mx = M_cmd[0], My = M_cmd[1], Mz = M_cmd[2];

    float k_fwd  = Q * L_REF * C_DELTA_FWD;
    float k_aft  = Q * L_REF * C_DELTA_AFT;
    float k_roll = Q * L_REF * C_DELTA_ROLL;
    float k_yaw  = Q * L_REF * C_DELTA_YAW;

    // 防除零
    const float eps = 1e-10f;
    if (std::fabs(k_fwd)  < eps) k_fwd  = eps;
    if (std::fabs(k_aft)  < eps) k_aft  = eps;
    if (std::fabs(k_roll) < eps) k_roll = eps;
    if (std::fabs(k_yaw)  < eps) k_yaw  = eps;

    float d_FL =  My / (4.0f * k_fwd)  + Mx / (4.0f * k_roll) + Mz / (4.0f * k_yaw);
    float d_FR =  My / (4.0f * k_fwd)  - Mx / (4.0f * k_roll) - Mz / (4.0f * k_yaw);
    float d_RL = -My / (4.0f * k_aft)  - Mx / (4.0f * k_roll) + Mz / (4.0f * k_yaw);
    float d_RR = -My / (4.0f * k_aft)  + Mx / (4.0f * k_roll) - Mz / (4.0f * k_yaw);

    // 归一化: 如果超限, 等比缩小保持方向
    float max_abs = d_FL;
    if (std::fabs(d_FR) > max_abs) max_abs = std::fabs(d_FR);
    if (std::fabs(d_RL) > max_abs) max_abs = std::fabs(d_RL);
    if (std::fabs(d_RR) > max_abs) max_abs = std::fabs(d_RR);
    if (std::fabs(d_FL) > max_abs) max_abs = std::fabs(d_FL);  // 确保非负

    if (max_abs > DELTA_MAX) {
        float scale = DELTA_MAX / max_abs;
        d_FL *= scale;
        d_FR *= scale;
        d_RL *= scale;
        d_RR *= scale;
    }

    delta_out[0] = d_FL;
    delta_out[1] = d_FR;
    delta_out[2] = d_RL;
    delta_out[3] = d_RR;
}

// =============================================================================
// 验证分配: 给定襟翼偏转, 计算实际力矩, 返回误差 (用于开环测试)
// =============================================================================
inline void verify_allocation(const float M_cmd[3], const float delta[4],
                               float Q,
                               float M_actual[3], float M_error[3]) {
    float d_FL = delta[0], d_FR = delta[1], d_RL = delta[2], d_RR = delta[3];

    // 正向计算力矩 (与 aero_6dof.hpp 一致)
    M_actual[0] = Q * L_REF * C_DELTA_ROLL * (d_FL - d_FR - d_RL + d_RR);
    M_actual[1] = Q * L_REF * (C_DELTA_FWD * (d_FL + d_FR) -
                                C_DELTA_AFT * (d_RL + d_RR));
    M_actual[2] = Q * L_REF * C_DELTA_YAW  * (d_FL - d_FR + d_RL - d_RR);

    M_error[0] = M_actual[0] - M_cmd[0];
    M_error[1] = M_actual[1] - M_cmd[1];
    M_error[2] = M_actual[2] - M_cmd[2];
}

}  // namespace belly_flop_6dof
}  // namespace falcon9

#endif  // FALCON9_BELLY_FLOP_6DOF_CONTROL_ALLOCATION_HPP
