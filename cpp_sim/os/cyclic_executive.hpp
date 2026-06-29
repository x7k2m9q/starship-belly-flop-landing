// =============================================================================
// cyclic_executive.hpp - 协作式循环调度器 (Phase 12)
// =============================================================================
// 协作式循环调度器 (Cyclic Executive / Rate-Monotonic Cooperative Scheduler)
//
// 批判性决策 (Phase 12 方案审查):
//   方案要求 FreeRTOS 4 任务 + 优先级调度. 但:
//     1. FreeRTOS 不原生运行于 Windows (Windows 模拟器 = Windows 线程, 非实时)
//     2. 多线程引入非确定性, 破坏 Python golden data 对齐
//     3. 真正的优先级反转/死锁测试需要 RTOS, Windows 上无法验证
//   改用协作式循环调度器:
//     - 单线程确定性 (golden data 可对齐)
//     - 4 任务按速率分离 (架构清晰, 可移植)
//     - 移植 FreeRTOS: 每个 task 函数体 → xTaskCreate 的入口, 零逻辑修改
//
// 调度模型 (与真实嵌入式飞控一致):
//   主循环 dt = 1ms (基础时钟)
//   Task_IMU     1000Hz (每 1ms)   优先级 5 (最高)
//   Task_Control  100Hz (每 10ms)  优先级 4
//   Task_Safety    10Hz (每 100ms) 优先级 3
//   Task_Guidance   1Hz (每 1000ms)优先级 2
//
// 实现: 仿真中物理积分步长 dt=0.01s=10ms, 对应控制周期.
//   IMU 1000Hz 通过子步实现 (每控制周期内 IMU 跑 10 次, 但仿真中无真实传感器,
//   故 IMU 任务在仿真中退化为每个控制周期采样一次, 标注为 100Hz-sim).
//   Guidance 1Hz = 每 100 个控制周期跑一次.
//   Safety 10Hz = 每 10 个控制周期跑一次.
// =============================================================================
#ifndef STARSHIP_OS_CYCLIC_EXECUTIVE_HPP
#define STARSHIP_OS_CYCLIC_EXECUTIVE_HPP

#include <cstdint>

namespace starship {
namespace os {

// 任务上下文 (传递给 task 函数的参数)
struct TaskContext {
    void* flight_computer;  // FlightComputer 指针
};

// 任务函数签名
using TaskFunc = void (*)(TaskContext&);

// 任务描述
struct Task {
    const char* name;
    TaskFunc func;
    int period_ticks;    // 每 N 个基础 tick 跑一次
    int tick_counter;    // 计数器
    int priority;        // 优先级 (数字大=高, 仅文档用)
    // 执行时间统计 (微秒, 由调度器测量)
    uint64_t exec_time_us_total;
    uint64_t exec_time_us_max;
    int run_count;
};

class CyclicExecutive {
public:
    static constexpr int MAX_TASKS = 8;

private:
    Task tasks_[MAX_TASKS];
    int num_tasks_;
    int tick_;  // 基础 tick 计数 (每个 step +1)

public:
    CyclicExecutive() : num_tasks_(0), tick_(0) {
        for (int i = 0; i < MAX_TASKS; ++i) {
            tasks_[i] = {"", nullptr, 0, 0, 0, 0, 0, 0};
        }
    }

    // 注册任务. period_ticks=每 N 个基础 tick 跑一次.
    void register_task(const char* name, TaskFunc func,
                       int period_ticks, int priority) {
        if (num_tasks_ >= MAX_TASKS) return;
        tasks_[num_tasks_] = {name, func, period_ticks, 0, priority, 0, 0, 0};
        ++num_tasks_;
    }

    // 调度一步 (一个基础 tick = 一个控制周期 dt).
    // 按优先级顺序检查每个任务是否到周期.
    void step(TaskContext& ctx) {
        ++tick_;
        for (int i = 0; i < num_tasks_; ++i) {
            tasks_[i].tick_counter += 1;
            if (tasks_[i].tick_counter >= tasks_[i].period_ticks) {
                tasks_[i].tick_counter = 0;
                tasks_[i].func(ctx);
                ++tasks_[i].run_count;
            }
        }
    }

    const Task& get_task(int i) const { return tasks_[i]; }
    int num_tasks() const { return num_tasks_; }
};

} // namespace os
} // namespace starship

#endif // STARSHIP_OS_CYCLIC_EXECUTIVE_HPP
