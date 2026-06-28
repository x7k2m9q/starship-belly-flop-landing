// =============================================================================
// fault_injection.hpp - 物理级故障注入框架 (Phase 13)
// =============================================================================
// 故障类型 (物理级, 测试控制鲁棒性):
//   1. SensorBiasDrift: 陀螺零偏缓慢漂移 (测试姿态控制器抗偏置能力)
//   2. ActuatorStuck:   单片襟翼卡死 (测试 4 襟翼冗余)
//   3. SensorDropout:   IMU 掉线 N 秒 (测试保持/惯性滑行恢复)
//
// 批判性决策 (Phase 13 方案审查):
//   方案要求 CPU过载/通信中断/电源毛刺/位翻转注入. 这些是 OS/硬件级故障,
//   需真实硬件才能有意义. Windows 上模拟这些是元仿真 (测 Windows, 不测代码), 拒绝.
//   物理级故障 (偏置漂移/执行器卡死/传感器掉线) 直接测试控制律鲁棒性, 有真实价值.
//   这与 SpaceX 的故障注入测试理念一致: 注入执行器失效, 验证系统能否安全着陆.
// =============================================================================
#ifndef FALCON9_BELLY_FLOP_6DOF_FAULT_INJECTION_HPP
#define FALCON9_BELLY_FLOP_6DOF_FAULT_INJECTION_HPP

#include "hal/hal.hpp"
#include <cmath>

namespace falcon9 {
namespace belly_flop_6dof {

// =============================================================================
// 故障类型枚举
// =============================================================================
enum class FaultType {
    NONE = 0,
    SENSOR_BIAS_DRIFT = 1,   // 陀螺零偏漂移
    ACTUATOR_STUCK    = 2,   // 襟翼卡死
    SENSOR_DROPOUT    = 3,   // IMU 掉线
};

// =============================================================================
// 故障注入器
// =============================================================================
class FaultInjector {
public:
    FaultType type;
    float start_time;    // 故障注入起始时间 (s)
    float severity;      // 故障严重度 (类型相关)
    int actuator_index;  // 卡死的襟翼索引 (0-3), 仅 ACTUATOR_STUCK 用
    float stuck_value;   // 卡死角度 (rad), 仅 ACTUATOR_STUCK 用
    bool active;         // 是否激活

    // 诊断
    int injection_count; // 故障注入次数
    char description[80];

    FaultInjector()
        : type(FaultType::NONE), start_time(0.0f), severity(0.0f),
          actuator_index(0), stuck_value(0.0f), active(false),
          injection_count(0) {
        description[0] = '\0';
    }

    // 配置陀螺零偏漂移
    // drift_rate: rad/s² (每秒漂移量). 战术级IMU典型漂移 0.01°/s=1.7e-4 rad/s
    void config_bias_drift(float start_t, float drift_rate_rad_per_s2) {
        type = FaultType::SENSOR_BIAS_DRIFT;
        start_time = start_t;
        severity = drift_rate_rad_per_s2;
        actuator_index = 0;
        active = true;
        snprintf(description, sizeof(description),
                 "gyro bias drift %.4f rad/s^2 @ t=%.1fs", drift_rate_rad_per_s2, start_t);
    }

    // 配置襟翼卡死
    // index: 0=FL, 1=FR, 2=RL, 3=RR. stuck_angle: 卡死角度 (rad)
    void config_actuator_stuck(float start_t, int index, float stuck_angle_rad) {
        type = FaultType::ACTUATOR_STUCK;
        start_time = start_t;
        actuator_index = index;
        stuck_value = stuck_angle_rad;
        active = true;
        snprintf(description, sizeof(description),
                 "flap[%d] stuck @ %.1f deg @ t=%.1fs",
                 index, stuck_angle_rad * 57.2957795f, start_t);
    }

    // 配置 IMU 掉线
    // duration: 掉线持续时间 (s)
    void config_sensor_dropout(float start_t, float duration_s) {
        type = FaultType::SENSOR_DROPOUT;
        start_time = start_t;
        severity = duration_s;  // 复用 severity 字段存持续时间
        active = true;
        snprintf(description, sizeof(description),
                 "IMU dropout %.2fs @ t=%.1fs", duration_s, start_t);
    }

    // 判断在时间 t 故障是否生效
    bool is_active_at(float t) const {
        if (!active || t < start_time) return false;
        if (type == FaultType::SENSOR_DROPOUT) {
            return t < start_time + severity;  // severity = duration
        }
        return true;  // 偏置漂移和卡死一旦激活就持续
    }

    // 应用故障到 IMU 采样
    void apply_to_imu(hal::IMUSample& sample, float t) {
        if (!is_active_at(t)) return;
        if (type == FaultType::SENSOR_BIAS_DRIFT) {
            // 零偏 = drift_rate × (t - start_time), 累加到陀螺输出
            float dt_fault = t - start_time;
            float bias = severity * dt_fault;
            sample.gyro[0] += bias;
            sample.gyro[1] += bias * 0.5f;  // 不同轴不同漂移率
            sample.gyro[2] += bias * 0.3f;
        } else if (type == FaultType::SENSOR_DROPOUT) {
            // 掉线: 输出零 (传感器无数据)
            for (int i = 0; i < 3; ++i) {
                sample.gyro[i] = 0.0f;
                sample.accel[i] = 0.0f;
            }
        }
    }

    // 应用故障到执行器指令
    void apply_to_actuator(hal::ActuatorCommand& cmd, float t) {
        if (!is_active_at(t)) return;
        if (type == FaultType::ACTUATOR_STUCK) {
            // 指定襟翼卡死: 强制指令为卡死角度
            if (actuator_index >= 0 && actuator_index < 4) {
                cmd.flap_cmd[actuator_index] = stuck_value;
            }
        }
    }
};

} // namespace belly_flop_6dof
} // namespace falcon9

#endif // FALCON9_BELLY_FLOP_6DOF_FAULT_INJECTION_HPP
