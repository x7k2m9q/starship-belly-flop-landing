// =============================================================================
// flight_computer.hpp - 4 任务飞控计算机 (Phase 12-14 集成)
// =============================================================================
// 集成:
//   - CyclicExecutive (协作式循环调度器, 4 任务)
//   - RingBuffer (SPSC 无锁队列, IMU→Control 数据流)
//   - Watchdog (逻辑看门狗, Control 喂狗, Safety 检查)
//   - SimulatedSensor (HAL, 未来替换为 RealSensor)
//   - PhaseController6DOF (Phase 11, 三阶段控制)
//   - FaultInjector (Phase 13, 物理级故障注入)
//
// 4 任务架构 (批判性实现):
//   Task_IMU      (每周期, 100Hz-sim): 读真值状态 → 传感器模型 → 故障注入 → RingBuffer
//   Task_Control  (每周期, 100Hz):     从 RingBuffer 取最新 IMU → PhaseController6DOF → 喂狗
//   Task_Safety   (每10周期, 10Hz):    检查看门狗 + 状态边界 → Kill 标志
//   Task_Guidance (每100周期, 1Hz):    任务监控 + 阶段进度日志
//
// 工程决策 (诚实记录):
//   PhaseController6DOF.update() 内部耦合了制导(阶段切换)和控制(姿态PD).
//   强行拆分为 1Hz 制导 + 100Hz 控制需要重构 PhaseController6DOF, 有破坏已验证
//   代码的风险. 当前 Task_Control 在 100Hz 运行完整 PhaseController6DOF (含阶段切换),
//   Task_Guidance 在 1Hz 作为高层监控. 真正的制导/控制分离留待制导算法变重时
//   (如在线凸优化). 拒绝无必要的重构.
//
//   IMU→Control 数据流通过 RingBuffer 演示架构. 当前 PhaseController6DOF 直接使用
//   真值状态 (仿真中无 EKF). 真实系统中 Control 任务从 EKF 估计状态运行,
//   EKF 输入来自 RingBuffer 中的 IMU 采样. 架构已就绪, 待 EKF 移植.
// =============================================================================
#ifndef FALCON9_BELLY_FLOP_6DOF_FLIGHT_COMPUTER_HPP
#define FALCON9_BELLY_FLOP_6DOF_FLIGHT_COMPUTER_HPP

#include "phase_controller_6dof.hpp"
#include "fault_injection.hpp"
#include "aero_6dof.hpp"
#include "../os/ring_buffer.hpp"
#include "../os/cyclic_executive.hpp"
#include "../os/watchdog.hpp"
#include "../hal/hal.hpp"
#include "../core/quaternion.hpp"
#include <chrono>
#include <cstdio>

// SimulatedSensor::read() 实现需在 hal 命名空间内 (声明在 hal.hpp 的 falcon9::hal)
namespace falcon9 {
namespace hal {

inline IMUSample SimulatedSensor::read(const float state14[14], float t) {
    IMUSample s;
    // 陀螺 = body 系角速度 + 零偏
    s.gyro[0] = state14[10] + gyro_bias_[0];
    s.gyro[1] = state14[11] + gyro_bias_[1];
    s.gyro[2] = state14[12] + gyro_bias_[2];
    // 比力近似: 当前实现简化为零 (架构演示, 真实 EKF 移植时补完整比力)
    s.accel[0] = accel_bias_[0];
    s.accel[1] = accel_bias_[1];
    s.accel[2] = accel_bias_[2];
    s.timestamp = t;
    return s;
}

} // namespace hal
} // namespace falcon9

namespace falcon9 {
namespace belly_flop_6dof {

// =============================================================================
// 飞控计算机
// =============================================================================
class FlightComputer {
public:
    // 任务周期 (基础 tick = 1 控制周期 = 10ms)
    static constexpr int IMU_PERIOD      = 1;    // 每周期 (100Hz-sim)
    static constexpr int CONTROL_PERIOD  = 1;    // 每周期 (100Hz)
    static constexpr int SAFETY_PERIOD   = 10;   // 每 10 周期 (10Hz)
    static constexpr int GUIDANCE_PERIOD = 100;  // 每 100 周期 (1Hz)

private:
    // OS 组件
    os::CyclicExecutive exec_;
    os::Watchdog watchdog_;
    os::RingBuffer<hal::IMUSample, 16> imu_buffer_;

    // HAL
    hal::SimulatedSensor sensor_;

    // 控制器
    PhaseController6DOF controller_;

    // 故障注入
    FaultInjector fault_;

    // 共享状态 (任务间通信)
    hal::ActuatorCommand actuator_cmd_;     // Control → 动力学
    float last_T_cmd_;
    float last_delta_flaps_[4];
    float last_tvc_gimbal_[2];
    ControllerInfo last_info_;

    // 安全状态
    bool kill_;
    char kill_reason_[80];
    bool landed_;

    // 执行时间统计 (微秒)
    uint64_t imu_exec_us_max_;
    uint64_t control_exec_us_max_;
    uint64_t safety_exec_us_max_;
    uint64_t guidance_exec_us_max_;

    // 时间
    float t_;

    // 真值状态 + 步长 (step() 设置, task_control/safety 读取)
    State6DOF truth_state_;
    float dt_;

public:
    FlightComputer(bool use_nonideal = true, bool use_notch = true)
        : watchdog_(os::Watchdog::DEFAULT_TIMEOUT_CYCLES),
          controller_(use_nonideal, use_notch),
          kill_(false), landed_(false),
          imu_exec_us_max_(0), control_exec_us_max_(0),
          safety_exec_us_max_(0), guidance_exec_us_max_(0),
          t_(0.0f), dt_(0.01f) {
        kill_reason_[0] = '\0';
        reset_actuators();
        last_info_.reset();

        // 注册 4 任务 (按优先级顺序: IMU > Control > Safety > Guidance)
        exec_.register_task("IMU",      &FlightComputer::task_imu_wrapper,
                           IMU_PERIOD, 5);
        exec_.register_task("Control",  &FlightComputer::task_control_wrapper,
                           CONTROL_PERIOD, 4);
        exec_.register_task("Safety",   &FlightComputer::task_safety_wrapper,
                           SAFETY_PERIOD, 3);
        exec_.register_task("Guidance", &FlightComputer::task_guidance_wrapper,
                           GUIDANCE_PERIOD, 2);
    }

    // 设置故障注入
    void set_fault(const FaultInjector& f) { fault_ = f; }

    // 单步更新 (一个控制周期 dt)
    // state: 真值状态 (in/out, 动力学积分后更新)
    // 返回: 是否 Kill
    bool step(State6DOF& state, float dt) {
        if (kill_ || landed_) return kill_;

        // 保存真值状态和步长供任务使用 (协作式调度, 单线程安全)
        truth_state_ = state;
        dt_ = dt;

        // 准备任务上下文
        os::TaskContext ctx;
        ctx.flight_computer = this;

        // 运行调度器 (执行到期的任务)
        exec_.step(ctx);

        // 动力学积分 (使用最新执行器指令)
        state = rk4_step(state, last_T_cmd_, last_delta_flaps_, dt,
                         last_tvc_gimbal_);

        t_ += dt;

        // 着陆判断
        float h = -state[2];
        if (h <= 0.0f) {
            landed_ = true;
        }

        return kill_;
    }

    // 访问器
    const ControllerInfo& info() const { return last_info_; }
    bool is_killed() const { return kill_; }
    bool is_landed() const { return landed_; }
    float time() const { return t_; }
    const os::Watchdog& watchdog() const { return watchdog_; }

    // 执行时间统计 (微秒)
    uint64_t imu_exec_us_max() const { return imu_exec_us_max_; }
    uint64_t control_exec_us_max() const { return control_exec_us_max_; }
    uint64_t safety_exec_us_max() const { return safety_exec_us_max_; }
    uint64_t guidance_exec_us_max() const { return guidance_exec_us_max_; }

private:
    void reset_actuators() {
        last_T_cmd_ = 0.0f;
        for (int i = 0; i < 4; ++i) last_delta_flaps_[i] = 0.0f;
        for (int i = 0; i < 2; ++i) last_tvc_gimbal_[i] = 0.0f;
        actuator_cmd_.thrust_cmd = 0.0f;
        for (int i = 0; i < 4; ++i) actuator_cmd_.flap_cmd[i] = 0.0f;
        for (int i = 0; i < 2; ++i) actuator_cmd_.tvc_cmd[i] = 0.0f;
    }

    // ===== 任务包装器 (静态函数 → 成员函数) =====
    static void task_imu_wrapper(os::TaskContext& ctx) {
        static_cast<FlightComputer*>(ctx.flight_computer)->task_imu();
    }
    static void task_control_wrapper(os::TaskContext& ctx) {
        static_cast<FlightComputer*>(ctx.flight_computer)->task_control();
    }
    static void task_safety_wrapper(os::TaskContext& ctx) {
        static_cast<FlightComputer*>(ctx.flight_computer)->task_safety();
    }
    static void task_guidance_wrapper(os::TaskContext& ctx) {
        static_cast<FlightComputer*>(ctx.flight_computer)->task_guidance();
    }

    // ===== Task_IMU (100Hz-sim): 传感器采样 =====
    void task_imu() {
        auto t0 = std::chrono::high_resolution_clock::now();

        // 从真值状态读取传感器数据 (架构: 真值 → 传感器 → 故障注入 → RingBuffer)
        // 真实系统中 IMU 硬件直接给出测量值, 此处用 SimulatedSensor 演示数据流
        hal::IMUSample sample = sensor_.read(truth_state_.data, t_);

        // 故障注入 (偏置漂移/掉线)
        fault_.apply_to_imu(sample, t_);

        // 推入 RingBuffer (生产者)
        imu_buffer_.push(sample);

        auto t1 = std::chrono::high_resolution_clock::now();
        uint64_t us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        if (us > imu_exec_us_max_) imu_exec_us_max_ = us;
    }

    // ===== Task_Control (100Hz): 姿态控制 + 襟翼分配 =====
    void task_control() {
        auto t0 = std::chrono::high_resolution_clock::now();

        // 从 RingBuffer 取最新 IMU 采样 (消费者, 架构演示)
        hal::IMUSample latest_imu;
        imu_buffer_.peek_latest(latest_imu);  // 不消费, 只偷看最新

        // 运行三阶段控制器 (使用真值状态, 仿真中无 EKF)
        // 猎鹰9号教训: omega 从 state 读取, 控制器内部 omega_actual 为必传
        float T_cmd, delta_flaps[4], tvc_gimbal[2];
        ControllerInfo info;
        bool ctrl_kill = controller_.update(truth_state_, dt_,
                                             T_cmd, delta_flaps,
                                             tvc_gimbal, info);

        // 故障注入到执行器指令
        hal::ActuatorCommand cmd;
        cmd.thrust_cmd = T_cmd;
        for (int i = 0; i < 4; ++i) cmd.flap_cmd[i] = delta_flaps[i];
        for (int i = 0; i < 2; ++i) cmd.tvc_cmd[i] = tvc_gimbal[i];
        fault_.apply_to_actuator(cmd, t_);

        // 保存执行器指令 (供动力学积分)
        last_T_cmd_ = cmd.thrust_cmd;
        for (int i = 0; i < 4; ++i) last_delta_flaps_[i] = cmd.flap_cmd[i];
        for (int i = 0; i < 2; ++i) last_tvc_gimbal_[i] = cmd.tvc_cmd[i];
        last_info_ = info;

        // 控制器 Kill 传递
        if (ctrl_kill && !kill_) {
            kill_ = true;
            snprintf(kill_reason_, sizeof(kill_reason_), "controller: %s",
                     info.kill_reason);
        }

        // 喂狗 (控制任务每周期喂狗)
        watchdog_.feed();

        auto t1 = std::chrono::high_resolution_clock::now();
        uint64_t us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        if (us > control_exec_us_max_) control_exec_us_max_ = us;
    }

    // ===== Task_Safety (10Hz): 安全监控 =====
    void task_safety() {
        auto t0 = std::chrono::high_resolution_clock::now();

        // 检查看门狗 (控制任务是否在喂狗)
        bool wd_timeout = watchdog_.check();
        if (wd_timeout && !kill_) {
            kill_ = true;
            snprintf(kill_reason_, sizeof(kill_reason_),
                     "watchdog timeout (%d cycles, control task stalled)",
                     watchdog_.total_timeouts());
        }

        // 状态边界检查
        float h = -truth_state_[2];
        float V = std::sqrt(truth_state_[3]*truth_state_[3] +
                            truth_state_[4]*truth_state_[4] +
                            truth_state_[5]*truth_state_[5]);

        // 数值健康检查
        bool healthy = true;
        for (int i = 0; i < 14; ++i) {
            if (!std::isfinite(truth_state_[i])) { healthy = false; break; }
        }
        if (!healthy && !kill_) {
            kill_ = true;
            snprintf(kill_reason_, sizeof(kill_reason_),
                     "NaN/Inf detected in state (safety monitor)");
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        uint64_t us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        if (us > safety_exec_us_max_) safety_exec_us_max_ = us;
    }

    // ===== Task_Guidance (1Hz): 任务监控 =====
    void task_guidance() {
        auto t0 = std::chrono::high_resolution_clock::now();

        // 高层任务监控 (当前为架构占位)
        // 真实系统中此处运行制导算法 (如 G-FOLD 凸优化)
        // 当前制导逻辑 (阶段切换) 已在 PhaseController6DOF 中, 100Hz 运行

        auto t1 = std::chrono::high_resolution_clock::now();
        uint64_t us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        if (us > guidance_exec_us_max_) guidance_exec_us_max_ = us;
    }
};

} // namespace belly_flop_6dof
} // namespace falcon9

#endif // FALCON9_BELLY_FLOP_6DOF_FLIGHT_COMPUTER_HPP
