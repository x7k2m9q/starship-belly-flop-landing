// =============================================================================
// types.hpp - 公共类型定义与定长数据结构
// 理论方案3.0 Phase 2: ARM Cortex-R5F + FreeRTOS
// 对应 Phase 1.3 接口隔离: 传感器/状态估计/控制输出/安全状态
//
// 约法三章:
//   1. 零动态内存 (无 vector/string/new/malloc, 全部定长数组)
//   2. float32 默认, double 仅用于 EKF 协方差 P
//   3. 所有矩阵使用 FixedMatrix 模板
// =============================================================================
#ifndef STARSHIP_CORE_TYPES_HPP
#define STARSHIP_CORE_TYPES_HPP

#include <cstdint>
#include "fixed_matrix.hpp"

namespace starship {

// ---------------------------------------------------------------------------
// 安全状态机 (与 Python safety_monitor.py SafetyMonitor 一致)
// 优先级: EMERGENCY_RECOVER > ABORT_MAX_BRAKE > FALLBACK_PD > MIN_FUEL_GLIDE > SAFE_MODE > NOMINAL
//   1. 姿态超限 (tilt > 30°): EMERGENCY_RECOVER — RCS 全力回正
//   2. 可达集违反 (vz² > 2*a_brake*h): ABORT_MAX_BRAKE — 全发满推力刹车
//   3. SOCP 不可行 (last_solve_status=infeasible): FALLBACK_PD — 退化 PD 控制
//   4. 燃料告警 (fuel < FUEL_RESERVE): MIN_FUEL_GLIDE — 最省油滑翔
//   5. 全部正常: NOMINAL
// ---------------------------------------------------------------------------
enum class SafetyStatus : uint8_t {
    NOMINAL           = 0,  // 全部正常
    EMERGENCY_RECOVER = 1,  // 姿态超限 (>30°): RCS 全力回正
    ABORT_MAX_BRAKE   = 2,  // 可达集违反: 全发满推力刹车
    FALLBACK_PD       = 3,  // SOCP 不可行: 退化 PD 控制
    MIN_FUEL_GLIDE    = 4,  // 燃料告警: 最省油滑翔
    SAFE_MODE         = 5,  // 通用安全模式 (多重故障)
};

// ---------------------------------------------------------------------------
// 发动机构型 (单发/三发)
// 对应 Python flight_controller.py n_engines (1 或 3)
// ---------------------------------------------------------------------------
enum class EngineConfig : uint8_t {
    SINGLE = 1,  // 单台 Merlin (回收段默认)
    TRIPLE = 3,  // 三台 Merlin (Octaweb 中心三角)
};

// ---------------------------------------------------------------------------
// 传感器数据 (Phase 1.3 接口隔离: 传感器任务 -> EKF/控制任务)
// 全部 float32; NED 系; 体系 IMU
// 对应 Python sensors.py 传感器输出
//
// 注意: 使用 #pragma pack(1) 紧凑对齐, 与 Python struct.pack('<3f3f3f3f?f?I')
//       严格一致 (58字节). 避免 C++ 默认对齐导致的 padding 不匹配.
// ---------------------------------------------------------------------------
#pragma pack(push, 1)
struct SensorData {
    // IMU (体系 b)
    float gyro[3];       // 陀螺角速度 [rad/s], b系
    float accel[3];      // 加速度计比力 [m/s²], b系

    // GPS (NED 系)
    float gps_pos[3];    // 位置 [m], NED
    float gps_vel[3];    // 速度 [m/s], NED
    bool  gps_valid;     // GPS 定位有效标志

    // 雷达高度计
    float radar_alt;     // 雷达高度 [m] (相对地面)
    bool  radar_valid;   // 雷达有效标志

    // 时间戳
    uint32_t timestamp_us;  // 时间戳 [us] (FreeRTOS tick 或硬件定时器)

    // 初始化为零/无效
    void reset() {
        for (int i = 0; i < 3; ++i) {
            gyro[i]    = 0.0f;
            accel[i]   = 0.0f;
            gps_pos[i] = 0.0f;
            gps_vel[i] = 0.0f;
        }
        gps_valid    = false;
        radar_alt    = 0.0f;
        radar_valid  = false;
        timestamp_us = 0;
    }
};
#pragma pack(pop)

// ---------------------------------------------------------------------------
// 状态估计 (EKF 输出, 15 维误差状态)
// 状态顺序: [δp(3), δv(3), δθ(3), δbg(3), δba(3)]
// 全部 float32, 协方差 P 除外 (double, 数值稳定)
// 对应 Python ekf.py EKF 状态: p, v, q, omega, bg, ba, P(15×15)
// ---------------------------------------------------------------------------
struct StateEstimate {
    // 导航解算结果 (NED 系)
    float p[3];          // 位置 [m], NED
    float v[3];          // 速度 [m/s], NED

    // 姿态 (b->n, Hamilton, q=[w,x,y,z])
    float q[4];          // 四元数
    float omega[3];      // 体系角速度 [rad/s] (= gyro - bg)

    // IMU 零偏估计
    float bg[3];         // 陀螺零偏 [rad/s]
    float ba[3];         // 加速度计零偏 [m/s²]

    // EKF 协方差矩阵 (15×15, 双精度, 行主序一维存储)
    // 对应 Python ekf.py self.P = np.eye(15)
    // 状态顺序: [δp, δv, δθ, δbg, δba]
    double P[15 * 15];

    // 估计有效性
    bool     valid;         // 估计有效标志
    uint32_t timestamp_us;

    // 初始化为零/单位四元数
    void reset() {
        for (int i = 0; i < 3; ++i) {
            p[i]     = 0.0f;
            v[i]     = 0.0f;
            omega[i] = 0.0f;
            bg[i]    = 0.0f;
            ba[i]    = 0.0f;
        }
        q[0] = 1.0f;  // 单位四元数 [w=1, x=y=z=0]
        q[1] = 0.0f;
        q[2] = 0.0f;
        q[3] = 0.0f;
        for (int i = 0; i < 15 * 15; ++i) {
            P[i] = 0.0;
        }
        valid        = false;
        timestamp_us = 0;
    }
};

// ---------------------------------------------------------------------------
// 控制输出 (控制任务 -> 执行器任务)
// 对应 Python flight_controller.py ControlOutput
//
// 注意: 使用 #pragma pack(1) 紧凑对齐, 与 Python struct.pack('<f4f3f2f3f3fBBfI')
//       严格一致 (74字节). 避免 C++ 默认对齐导致的 padding 不匹配.
// ---------------------------------------------------------------------------
#pragma pack(push, 1)
struct ControlOutput {
    float throttle;        // 油门 [0, 1] (归一化推力比例)

    // 期望姿态 (b->n, Hamilton)
    float q_des[4];        // 期望四元数 [w,x,y,z]
    float omega_des[3];    // 期望角速度 [rad/s], b系

    // 执行器指令
    float tvc_gimbal[2];   // TVC 万向架 [pitch, yaw] [rad]
    float gf_cmd[3];       // 燃气舵指令 [roll, pitch, yaw] (归一化 [-1,1])
    float rcs_cmd[3];      // RCS 力矩指令 [Mx, My, Mz] [N·m]

    // 状态与推力
    SafetyStatus  status;      // 安全状态机输出
    EngineConfig  n_engines;   // 投入发动机数量 (SINGLE/TRIPLE)
    float         total_thrust;// 总推力 [N] (throttle * 单发最大推力 * n_engines)

    uint32_t timestamp_us;

    // 初始化为零/默认
    void reset() {
        throttle = 0.0f;
        q_des[0] = 1.0f;  // 单位四元数
        q_des[1] = 0.0f;
        q_des[2] = 0.0f;
        q_des[3] = 0.0f;
        for (int i = 0; i < 3; ++i) {
            omega_des[i] = 0.0f;
            gf_cmd[i]    = 0.0f;
            rcs_cmd[i]   = 0.0f;
        }
        tvc_gimbal[0] = 0.0f;
        tvc_gimbal[1] = 0.0f;
        status        = SafetyStatus::NOMINAL;
        n_engines     = EngineConfig::SINGLE;
        total_thrust  = 0.0f;
        timestamp_us  = 0;
    }
};
#pragma pack(pop)

}  // namespace starship

#endif  // STARSHIP_CORE_TYPES_HPP
