// =============================================================================
// hal.hpp - 最小硬件抽象层 (Phase 14)
// =============================================================================
// 传感器/执行器抽象基类, 为未来接真实硬件预留.
//
// 批判性决策 (Phase 14 方案审查):
//   方案要求 SensorInterface/ActuatorInterface + UDP 实时通信.
//   UDP 在单机无价值 (只增延迟), 拒绝.
//   HAL 接口本身有价值 (SimulatedSensor vs RealSensor 切换), 但不过度设计:
//     - 传感器: read() 返回 IMU 采样 (角速度+比力)
//     - 执行器: write() 接收襟翼+TVC 指令
//   当前仅 SimulatedSensor/SimulatedActuator 实现, 未来 RealSensor 直接替换.
// =============================================================================
#ifndef FALCON9_HAL_HAL_HPP
#define FALCON9_HAL_HAL_HPP

#include "../core/fixed_matrix.hpp"

namespace falcon9 {
namespace hal {

// IMU 采样数据 (body 系)
struct IMUSample {
    float gyro[3];   // 角速度 (rad/s), body 系
    float accel[3];  // 比力 (m/s²), body 系 (不含重力)
    float timestamp; // 时间戳 (s)
};

// 执行器指令
struct ActuatorCommand {
    float flap_cmd[4];    // 襟翼指令 (rad): FL, FR, RL, RR
    float tvc_cmd[2];     // TVC 指令 (rad): pitch, yaw
    float thrust_cmd;     // 推力指令 (N)
};

// =============================================================================
// 传感器接口 (抽象基类)
// =============================================================================
class SensorInterface {
public:
    virtual ~SensorInterface() = default;
    // 从真实/仿真状态读取传感器数据. state14 = 14维状态.
    virtual IMUSample read(const float state14[14], float t) = 0;
};

// 仿真传感器: 从动力学状态提取 IMU 数据, 含可选噪声
class SimulatedSensor : public SensorInterface {
    float gyro_bias_[3];   // 陀螺零偏 (rad/s)
    float accel_bias_[3];  // 加计零偏 (m/s²)
public:
    SimulatedSensor() {
        for (int i = 0; i < 3; ++i) { gyro_bias_[i] = 0.0f; accel_bias_[i] = 0.0f; }
    }

    void set_gyro_bias(float bx, float by, float bz) {
        gyro_bias_[0] = bx; gyro_bias_[1] = by; gyro_bias_[2] = bz;
    }

    IMUSample read(const float state14[14], float t) override;
};

// =============================================================================
// 执行器接口 (抽象基类)
// =============================================================================
class ActuatorInterface {
public:
    virtual ~ActuatorInterface() = default;
    // 写入指令, 返回实际输出 (含非理想特性)
    virtual void write(const ActuatorCommand& cmd, float dt,
                       float flap_actual[4], float tvc_actual[2],
                       float& thrust_actual) = 0;
};

} // namespace hal
} // namespace falcon9

#endif // FALCON9_HAL_HAL_HPP
