// =============================================================================
// ring_buffer.hpp - SPSC 无锁环形缓冲区 (Phase 12)
// =============================================================================
// 单生产者-单消费者 (SPSC) 无锁环形缓冲区, 基于 std::atomic.
//
// 用途: IMU(生产者) → 控制(消费者) 的数据流, 无 mutex, 无数据撕裂.
//       未来移植 FreeRTOS 时, 此类可直接用于 task 间通信.
//
// 内存: 定长数组, 零动态分配. 容量 N-1 (一个 slot 用于区分满/空).
//
// 批判性决策 (Phase 12 方案审查):
//   方案要求 FreeRTOS 多任务 + SPSC. 但 FreeRTOS 不原生运行于 Windows,
//   Windows 模拟器是 Windows 线程 (非实时, 工程降级).
//   本类实现真实无锁队列, 在协作式调度器中作为速率缓冲 (IMU 1000Hz → 控制 100Hz),
//   移植 FreeRTOS 时生产者/消费者各为独立 task, 此类零修改复用.
// =============================================================================
#ifndef STARSHIP_OS_RING_BUFFER_HPP
#define STARSHIP_OS_RING_BUFFER_HPP

#include <atomic>
#include <cstddef>

namespace starship {
namespace os {

template <typename T, std::size_t N>
class RingBuffer {
    static_assert(N >= 2, "RingBuffer capacity must be >= 2");
    T buffer_[N];
    std::atomic<std::size_t> head_{0};  // 写索引 (生产者)
    std::atomic<std::size_t> tail_{0};  // 读索引 (消费者)

public:
    RingBuffer() = default;

    // 生产者: 写入一个元素. 满则丢弃并返回 false.
    bool push(const T& item) {
        const std::size_t h = head_.load(std::memory_order_relaxed);
        const std::size_t next = (h + 1) % N;
        if (next == tail_.load(std::memory_order_acquire)) {
            return false;  // 满
        }
        buffer_[h] = item;
        head_.store(next, std::memory_order_release);
        return true;
    }

    // 消费者: 读取一个元素. 空则返回 false.
    bool pop(T& out) {
        const std::size_t t = tail_.load(std::memory_order_relaxed);
        if (t == head_.load(std::memory_order_acquire)) {
            return false;  // 空
        }
        out = buffer_[t];
        tail_.store((t + 1) % N, std::memory_order_release);
        return true;
    }

    // 偷看最新数据 (不消费). 空则返回 false.
    bool peek_latest(T& out) const {
        const std::size_t h = head_.load(std::memory_order_acquire);
        const std::size_t t = tail_.load(std::memory_order_relaxed);
        if (h == t) return false;  // 空
        out = buffer_[(h == 0) ? (N - 1) : (h - 1)];
        return true;
    }

    bool empty() const {
        return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
    }

    std::size_t capacity() const { return N - 1; }
};

} // namespace os
} // namespace starship

#endif // STARSHIP_OS_RING_BUFFER_HPP
