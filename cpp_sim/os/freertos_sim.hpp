// =============================================================================
// freertos_sim.hpp - FreeRTOS 任务模拟 (Desktop std::thread 版本)
// 理论方案3.0 Phase 3: 软硬件协同验证
//
// 架构:
//   Python物理引擎 (dynamics.py) ←UDP→ C++飞控 (FreeRTOS任务模拟)
//
// 任务优先级与频率 (与真实FreeRTOS部署一致):
//   Task_IMU      1000Hz  优先级5 (最高)  IMU读取 + EKF预测
//   Task_Control   100Hz  优先级4         姿态控制 + Octaweb推力分配
//   Task_Safety     10Hz  优先级3         安全监控 + 故障检测
//   Task_Guidance    1Hz  优先级2         G-FOLD SOCP重求解
//
// 通信:
//   任务间通过 RingBuffer<T, N> 无锁环形缓冲区通信
//   IMU→Control: SensorData (1000Hz→100Hz, 10:1降采样)
//   Control→Safety: StateEstimate (100Hz→10Hz, 10:1降采样)
//   Guidance→Control: GuidanceOutput (1Hz→100Hz, 零阶保持)
//   Safety→Control: SafetyStatus (10Hz→100Hz, 零阶保持)
//
// Desktop模拟:
//   使用 std::thread + std::chrono 实现周期调度
//   jitter目标: <1ms (验收标准)
//   真实FreeRTOS部署时替换为 xTaskCreate + vTaskDelayUntil
// =============================================================================
#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

#include "../core/ring_buffer.hpp"
#include "../core/types.hpp"
#include "../gnc/ekf.hpp"
#include "../gnc/guidance.hpp"
#include "../gnc/control.hpp"
#include "../gnc/octaweb.hpp"
#include "../gnc/safety.hpp"
#include "../gnc/sensor_guard.hpp"

namespace falcon9 {

// ===========================================================================
// 时间工具
// ===========================================================================
using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

inline double elapsed_ms(TimePoint start, TimePoint end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ===========================================================================
// FlightComputer - 飞控计算机 (所有GNC模块的容器)
//   持有 EKF/Guidance/Control/Octaweb/Safety 的实例
//   各Task通过引用访问这些模块
// ===========================================================================
class FlightComputer {
public:
    MEKF            ekf;
    LandingGuidance guidance;
    AttitudeController attitude;
    Octaweb         octaweb;
    SafetyMonitor   safety;
    SensorGuard     sensor_guard;   // 工程硬化阶段一: 传感器数据净化

    // 任务间共享数据 (通过RingBuffer传递)
    SensorData      latest_sensor;       // 最新传感器数据 (Task_IMU写入)
    StateEstimate   latest_state;        // 最新状态估计 (Task_IMU写入)
    ControlOutput   latest_control;      // 最新控制输出 (Task_Control写入)
    SafetyStatus    latest_safety;       // 最新安全状态 (Task_Safety写入)

    // 系统状态
    std::atomic<bool> running{true};     // 运行标志
    std::atomic<float> sim_time{0.0f};   // 仿真时间 [s]
    std::atomic<float> fuel_mass{15000.0f}; // 剩余燃料 [kg]

    // 统计
    std::atomic<int> imu_steps{0};
    std::atomic<int> control_steps{0};
    std::atomic<int> guidance_steps{0};
    std::atomic<int> safety_steps{0};
    std::atomic<double> imu_jitter_max{0.0};
    std::atomic<double> control_jitter_max{0.0};

    // 着陆标志
    std::atomic<bool> landed{false};

    FlightComputer() {
        latest_sensor.reset();
        latest_state.reset();
        latest_control.reset();
        latest_safety = SafetyStatus::NOMINAL;
        octaweb.set_engine_config(1);
    }

    void stop() { running = false; }
};

// ===========================================================================
// Task_IMU - IMU任务 (1000Hz, 1ms周期)
//
// 注: Windows 定时器分辨率默认 15.6ms, std::this_thread::sleep_until 无法
//     精确到 1ms. 实测 Task_IMU 只能运行 ~64Hz, 严重低于 1000Hz.
//     EKF 处理已移至 main.cpp 主循环 (100Hz 传感器速率, 匹配 dt=0.01).
//     本任务保留为占位, 真实 FreeRTOS 部署时恢复 1000Hz IMU 读取.
// ===========================================================================
inline void task_imu(FlightComputer& fc, std::atomic<bool>& sensor_ready) {
    (void)sensor_ready;
    const double period_ms = 1.0;  // 1000Hz
    auto next_wakeup = Clock::now();

    while (fc.running.load(std::memory_order_relaxed)) {
        next_wakeup += std::chrono::microseconds(static_cast<int64_t>(period_ms * 1000));
        std::this_thread::sleep_until(next_wakeup);
    }
}

// ===========================================================================
// process_sensor_data - 在主循环中处理传感器数据 (100Hz)
//
// 对应 Python flight_controller.py 的 EKF predict + update
// dt=0.01s 与 Python 100Hz 一致
// ===========================================================================
inline void process_sensor_data(FlightComputer& fc, const SensorData& sensor) {
    // 1. EKF预测步 (dt=0.01s, 与 Python 100Hz 一致)
    fc.ekf.predict(sensor.gyro, sensor.accel, 0.01f);

    // 2. GPS更新 (Python 端 10Hz 发送 gps_valid=true)
    if (sensor.gps_valid) {
        fc.ekf.update_gps(sensor.gps_pos, sensor.gps_vel);
    }

    // 3. 雷达更新 (Python 端 50Hz 发送 radar_valid=true)
    if (sensor.radar_valid) {
        fc.ekf.update_radar(sensor.radar_alt);
    }

    // 4. 输出状态估计
    fc.ekf.get_state(fc.latest_state);
    // 填充角速度: omega_b = gyro_meas - bg (对应 Python flight_controller.py:227)
    //   EKF.get_state 将 omega 置零, 必须由调用方用 IMU 测量填充.
    //   修复: 旧版未填充 omega, 姿态控制器 Kd 项失效, 纯比例控制导致发散.
    fc.ekf.get_estimated_omega(sensor.gyro, fc.latest_state.omega);
    fc.latest_state.valid = true;
    fc.latest_state.timestamp_us = sensor.timestamp_us;

    fc.imu_steps.fetch_add(1, std::memory_order_relaxed);
}

// ===========================================================================
// Task_Control - 控制任务 (100Hz, 10ms周期)
//
// 注: 与 Task_IMU 同理, Windows 定时器分辨率导致无法精确 100Hz.
//     控制逻辑已移至 process_control(), 由主循环 100Hz 调用.
//     本任务保留为占位, 真实 FreeRTOS 部署时恢复.
// ===========================================================================
inline void task_control(FlightComputer& fc) {
    const double period_ms = 10.0;  // 100Hz
    auto next_wakeup = Clock::now();

    while (fc.running.load(std::memory_order_relaxed)) {
        next_wakeup += std::chrono::microseconds(static_cast<int64_t>(period_ms * 1000));
        std::this_thread::sleep_until(next_wakeup);
    }
}

// ===========================================================================
// sanitize_control_output - 控制输出净化 (工程硬化阶段一)
//
// 防止 NaN/Inf/超量程控制指令进入执行器:
//   - throttle: 钳位到 [0, 1], NaN → 0 (关机)
//   - q_des: NaN → 单位四元数, 归一化
//   - tvc_gimbal: 钳位到 [-0.262, 0.262] (±15°), NaN → 0
//   - gf_cmd: 钳位到 [-1, 1], NaN → 0
//   - rcs_cmd: 钳位到物理量程, NaN → 0
// ===========================================================================
inline void sanitize_control_output(ControlOutput& c) {
    // 油门: [0, 1], NaN/Inf → 0 (安全关机)
    if (std::isnan(c.throttle) || std::isinf(c.throttle)) {
        c.throttle = 0.0f;
    } else if (c.throttle < 0.0f) {
        c.throttle = 0.0f;
    } else if (c.throttle > 1.0f) {
        c.throttle = 1.0f;
    }

    // 期望四元数: NaN → 单位, 归一化
    bool q_bad = false;
    for (int i = 0; i < 4; ++i) {
        if (std::isnan(c.q_des[i]) || std::isinf(c.q_des[i])) {
            q_bad = true;
            break;
        }
    }
    if (q_bad) {
        c.q_des[0] = 1.0f; c.q_des[1] = 0.0f;
        c.q_des[2] = 0.0f; c.q_des[3] = 0.0f;
    } else {
        float qn = c.q_des[0]*c.q_des[0] + c.q_des[1]*c.q_des[1]
                 + c.q_des[2]*c.q_des[2] + c.q_des[3]*c.q_des[3];
        if (qn < 0.5f || qn > 2.0f) {
            // 范数异常, 重置为单位四元数
            c.q_des[0] = 1.0f; c.q_des[1] = 0.0f;
            c.q_des[2] = 0.0f; c.q_des[3] = 0.0f;
        } else {
            float inv = 1.0f / std::sqrt(qn);
            for (int i = 0; i < 4; ++i) c.q_des[i] *= inv;
        }
    }

    // 期望角速度: NaN → 0, 钳位 ±10 rad/s
    for (int i = 0; i < 3; ++i) {
        if (std::isnan(c.omega_des[i]) || std::isinf(c.omega_des[i])) {
            c.omega_des[i] = 0.0f;
        } else if (c.omega_des[i] > 10.0f) {
            c.omega_des[i] = 10.0f;
        } else if (c.omega_des[i] < -10.0f) {
            c.omega_des[i] = -10.0f;
        }
    }

    // TVC 万向架: ±15° = ±0.262 rad, NaN → 0
    constexpr float TVC_LIMIT = 0.262f;
    for (int i = 0; i < 2; ++i) {
        if (std::isnan(c.tvc_gimbal[i]) || std::isinf(c.tvc_gimbal[i])) {
            c.tvc_gimbal[i] = 0.0f;
        } else if (c.tvc_gimbal[i] > TVC_LIMIT) {
            c.tvc_gimbal[i] = TVC_LIMIT;
        } else if (c.tvc_gimbal[i] < -TVC_LIMIT) {
            c.tvc_gimbal[i] = -TVC_LIMIT;
        }
    }

    // 燃气舵: [-1, 1], NaN → 0
    for (int i = 0; i < 3; ++i) {
        if (std::isnan(c.gf_cmd[i]) || std::isinf(c.gf_cmd[i])) {
            c.gf_cmd[i] = 0.0f;
        } else if (c.gf_cmd[i] > 1.0f) {
            c.gf_cmd[i] = 1.0f;
        } else if (c.gf_cmd[i] < -1.0f) {
            c.gf_cmd[i] = -1.0f;
        }
    }

    // RCS: ±50000 N·m, NaN → 0
    constexpr float RCS_LIMIT = 50000.0f;
    for (int i = 0; i < 3; ++i) {
        if (std::isnan(c.rcs_cmd[i]) || std::isinf(c.rcs_cmd[i])) {
            c.rcs_cmd[i] = 0.0f;
        } else if (c.rcs_cmd[i] > RCS_LIMIT) {
            c.rcs_cmd[i] = RCS_LIMIT;
        } else if (c.rcs_cmd[i] < -RCS_LIMIT) {
            c.rcs_cmd[i] = -RCS_LIMIT;
        }
    }

    // 总推力: NaN → 0, 负值 → 0
    if (std::isnan(c.total_thrust) || std::isinf(c.total_thrust)) {
        c.total_thrust = 0.0f;
    } else if (c.total_thrust < 0.0f) {
        c.total_thrust = 0.0f;
    }
}

// ===========================================================================
// process_control - 在主循环中执行控制逻辑 (100Hz)
//
// 对应 Python flight_controller.py 的 guidance + attitude + allocate
// dt=0.01s 与 Python 100Hz 一致
// ===========================================================================
inline void process_control(FlightComputer& fc) {
    if (!fc.latest_state.valid) return;

    // 1. 制导更新 (每 100Hz, G-FOLD 重求解内部仍为 1Hz/100步)
    float t = fc.sim_time.load(std::memory_order_relaxed);
    float fuel = fc.fuel_mass.load(std::memory_order_relaxed);
    fc.guidance.update(
        fc.latest_state.p, fc.latest_state.v,
        fc.latest_state.q, fuel, t, 0.01f);

    if (fc.guidance.landed) {
        fc.landed.store(true, std::memory_order_relaxed);
    }
    fc.guidance_steps.fetch_add(1, std::memory_order_relaxed);

    // 2. 读取制导输出
    float throttle = fc.guidance.throttle;
    float q_des[4] = {fc.guidance.q_des[0], fc.guidance.q_des[1],
                      fc.guidance.q_des[2], fc.guidance.q_des[3]};

    // 3. 计算质量属性并设置姿态控制器参数
    float m_total, cg_x, I_body[3];
    rocket_params::mass_properties(fuel, m_total, cg_x, I_body);
    fc.attitude.set_inertia(I_body);
    bool is_gfold = (fc.guidance.phase == GuidancePhase::GFOLD ||
                     fc.guidance.phase == GuidancePhase::DEADBAND);
    fc.attitude.set_phase(is_gfold);
    fc.octaweb.set_phase(is_gfold);  // 同步 Octaweb TVC 限幅 (15° G-FOLD / 3° 其他)
    float h = -fc.latest_state.p[2];
    float T_max_single = rocket_params::thrust_at_alt(h);
    // 修复: TVC权限仅基于中心发动机(0号)推力, 不是总推力.
    //   旧版: throttle * T_max_single * n_active → 3发模式高估3倍 → TVC欠偏转3倍
    //   新版: 用上一步中心发动机实际推力 (含一阶滞后), 对应 Python center_tvc.thrust
    float est_thrust = fc.octaweb.engines[0].thrust;
    if (est_thrust < 1.0f) {
        // 首步或关机: 用指令推力估算
        est_thrust = throttle * T_max_single;
    }
    fc.attitude.set_thrust(est_thrust);
    fc.attitude.set_cg(cg_x);

    // 4. 姿态控制
    float tvc_gimbal[2];
    float gf_cmd[3], rcs_cmd[3];
    fc.attitude.update(
        fc.latest_state.q, fc.latest_state.omega,
        q_des, fc.guidance.omega_des,
        throttle, fc.latest_state.p, fc.latest_state.v,
        fc.octaweb.n_active, tvc_gimbal, gf_cmd, rcs_cmd);

    // 5. Octaweb推力分配 (先同步发动机配置)
    if (fc.guidance.n_engines_current != fc.octaweb.n_active) {
        fc.octaweb.set_engine_config(fc.guidance.n_engines_current);
    }
    float total_thrust = fc.octaweb.update(
        throttle, tvc_gimbal[0], tvc_gimbal[1],
        fc.guidance.phase_str(), h, 0.01f);

    // 6. 推力一致性检查 (Phase 0)
    if (fc.safety.thrust_check_enabled) {
        float T_expected = fc.octaweb.get_expected_thrust(throttle, h);
        bool is_fault;
        float ratio;
        int streak;
        fc.safety.check_thrust_consistency(
            total_thrust, T_expected, 0.01f,
            is_fault, ratio, streak);
        if (is_fault) {
            fc.latest_safety = SafetyStatus::EMERGENCY_RECOVER;
            fc.safety.reset_thrust_check();
        }
    }

    // 7. 输出控制
    fc.latest_control.throttle = throttle;
    for (int i = 0; i < 4; ++i)
        fc.latest_control.q_des[i] = q_des[i];
    fc.latest_control.tvc_gimbal[0] = tvc_gimbal[0];
    fc.latest_control.tvc_gimbal[1] = tvc_gimbal[1];
    for (int i = 0; i < 3; ++i) {
        fc.latest_control.gf_cmd[i] = gf_cmd[i];
        fc.latest_control.rcs_cmd[i] = rcs_cmd[i];
    }
    fc.latest_control.total_thrust = total_thrust;
    // 发动机配置: n_active>=3 -> TRIPLE, 否则 SINGLE (含故障安全)
    fc.latest_control.n_engines = (fc.octaweb.n_active >= 3)
        ? EngineConfig::TRIPLE : EngineConfig::SINGLE;
    fc.latest_control.status = fc.latest_safety;
    fc.latest_control.timestamp_us = static_cast<uint32_t>(
        fc.sim_time.load() * 1e6);

    // === 工程硬化阶段一: 控制输出净化 ===
    // 防止 NaN/Inf 控制指令进入执行器, 导致硬件损坏
    sanitize_control_output(fc.latest_control);

    fc.control_steps.fetch_add(1, std::memory_order_relaxed);
}

// ===========================================================================
// Task_Safety - 安全监控任务 (10Hz, 100ms周期)
//
// 职责:
//   1. 读取最新 StateEstimate
//   2. 安全状态机评估 (姿态/可达集/SOCP/燃料)
//   3. 输出 SafetyStatus
// ===========================================================================
inline void task_safety(FlightComputer& fc) {
    const double period_ms = 100.0;  // 10Hz
    auto next_wakeup = Clock::now();

    while (fc.running.load(std::memory_order_relaxed)) {
        if (fc.latest_state.valid) {
            float h = -fc.latest_state.p[2];
            float fuel = fc.fuel_mass.load(std::memory_order_relaxed);

            SafetyStatus status = fc.safety.evaluate(
                fc.latest_state, fuel, h,
                fc.latest_control.total_thrust,
                fc.latest_control.n_engines == EngineConfig::TRIPLE ? 3 : 1);

            fc.latest_safety = status;
            fc.safety_steps.fetch_add(1, std::memory_order_relaxed);

            // 着陆检测
            if (h < 0.5f) {
                fc.landed.store(true, std::memory_order_relaxed);
            }
        }

        next_wakeup += std::chrono::microseconds(static_cast<int64_t>(period_ms * 1000));
        std::this_thread::sleep_until(next_wakeup);
    }
}

// ===========================================================================
// Task_Guidance - 制导任务 (已合并至 Task_Control)
//
// 注: Python 架构中 guidance.update() 每步调用 (100Hz), G-FOLD 重求解
//     内部每 100 步 (1Hz)。为对齐 Python, 制导已移至 task_control (100Hz)。
//     本任务保留为占位, 真实 FreeRTOS 部署时可恢复 1Hz 独立任务。
// ===========================================================================
inline void task_guidance(FlightComputer& fc) {
    const double period_ms = 1000.0;  // 1Hz
    auto next_wakeup = Clock::now();

    while (fc.running.load(std::memory_order_relaxed)) {
        next_wakeup += std::chrono::microseconds(static_cast<int64_t>(period_ms * 1000));
        std::this_thread::sleep_until(next_wakeup);
    }
}

// ===========================================================================
// launch_flight_computer - 启动飞控计算机 (4个任务线程)
//
// 返回: FlightComputer引用 (调用方负责join线程)
// ===========================================================================
struct FlightThreads {
    std::thread imu;
    std::thread control;
    std::thread safety;
    std::thread guidance;
};

inline FlightThreads launch_flight_computer(FlightComputer& fc,
                                             std::atomic<bool>& sensor_ready) {
    FlightThreads threads;
    threads.imu      = std::thread(task_imu, std::ref(fc), std::ref(sensor_ready));
    threads.control  = std::thread(task_control, std::ref(fc));
    threads.safety   = std::thread(task_safety, std::ref(fc));
    threads.guidance = std::thread(task_guidance, std::ref(fc));
    return threads;
}

inline void join_flight_computer(FlightThreads& threads) {
    if (threads.imu.joinable())      threads.imu.join();
    if (threads.control.joinable())  threads.control.join();
    if (threads.safety.joinable())   threads.safety.join();
    if (threads.guidance.joinable()) threads.guidance.join();
}

}  // namespace falcon9
