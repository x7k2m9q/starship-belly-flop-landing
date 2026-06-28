// =============================================================================
// watchdog.hpp - 逻辑看门狗 (Phase 12)
// =============================================================================
// 看门狗: 控制任务每周期喂狗, 安全任务检查. 超时则触发安全动作.
//
// 工作原理:
//   - Task_Control 每个周期调用 feed()
//   - Task_Safety 每个周期调用 check(dt)
//   - 若超过 N 个周期未喂狗, 触发 timeout, 系统进入安全模式
//
// 批判性决策:
//   方案要求"看门狗复位系统". 在 Windows 上无法真正复位进程,
//   改为逻辑复位: 设置 fault 标志, 由安全任务决定处置 (如切到安全模式/终止).
//   移植 FreeRTOS 时, 超时可直接调用 NVIC_SystemReset() 真复位.
// =============================================================================
#ifndef FALCON9_OS_WATCHDOG_HPP
#define FALCON9_OS_WATCHDOG_HPP

#include <cstdint>

namespace falcon9 {
namespace os {

class Watchdog {
public:
    static constexpr int DEFAULT_TIMEOUT_CYCLES = 5;  // 5 周期未喂狗 → 超时

private:
    int counter_;          // 剩余周期 (每周期 -1, 喂狗重置为 timeout)
    int timeout_cycles_;   // 超时阈值
    bool triggered_;       // 是否已触发
    int total_timeouts_;   // 累计超时次数

public:
    explicit Watchdog(int timeout_cycles = DEFAULT_TIMEOUT_CYCLES)
        : counter_(timeout_cycles), timeout_cycles_(timeout_cycles),
          triggered_(false), total_timeouts_(0) {}

    // 控制任务调用: 喂狗
    void feed() {
        counter_ = timeout_cycles_;
        triggered_ = false;
    }

    // 安全任务调用: 检查. 返回 true 表示本次检查触发了超时.
    bool check() {
        if (counter_ > 0) {
            --counter_;
        }
        if (counter_ <= 0 && !triggered_) {
            triggered_ = true;
            ++total_timeouts_;
            return true;
        }
        return triggered_;
    }

    bool is_triggered() const { return triggered_; }
    int remaining_cycles() const { return counter_; }
    int total_timeouts() const { return total_timeouts_; }

    void reset() {
        counter_ = timeout_cycles_;
        triggered_ = false;
        total_timeouts_ = 0;
    }
};

} // namespace os
} // namespace falcon9

#endif // FALCON9_OS_WATCHDOG_HPP
