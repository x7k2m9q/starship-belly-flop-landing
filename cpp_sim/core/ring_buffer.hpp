// =============================================================================
// ring_buffer.hpp - 无锁环形缓冲区 (FreeRTOS 任务间通信)
// 星舰回收算法 C++ 翻译项目
//
// 设计:
//   - SPSC (单生产者单消费者) 无锁环形缓冲
//   - head/tail 使用 std::atomic, 内存序: acquire/release 保证数据可见性
//   - 容量 = N - 1 (一个槽位浪费, 用于区分满/空)
//   - 不抛异常, 零动态内存 (使用 std::array)
// =============================================================================
#pragma once

#include <array>
#include <atomic>
#include <cstddef>

namespace starship {

// ===========================================================================
// RingBuffer<T, N> - 无锁环形缓冲区 (SPSC)
// T: 元素类型
// N: 缓冲槽数 (实际可用容量 = N - 1)
// ===========================================================================
template <typename T, std::size_t N>
class RingBuffer {
    static_assert(N > 1, "N 必须大于 1");

public:
    using value_type = T;
    using size_type  = std::size_t;

    // 构造: head/tail 初始化为 0
    RingBuffer() : head_(0), tail_(0) {}

    // 禁止拷贝/移动 (含原子量, 不可拷贝)
    RingBuffer(const RingBuffer&)            = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;
    RingBuffer(RingBuffer&&)                 = delete;
    RingBuffer& operator=(RingBuffer&&)      = delete;

    // ------- 生产者: 压入一个元素 (非阻塞) -------
    // 成功返回 true, 缓冲满返回 false
    // 内存序: acquire 读取 tail_ (同步消费者释放), release 写入 head_ (发布数据)
    bool push(const T& item) {
        const size_type h    = head_.load(std::memory_order_relaxed);
        const size_type next = (h + 1) % N;
        if (next == tail_.load(std::memory_order_acquire)) {
            return false;  // 缓冲满
        }
        buffer_[h] = item;  // 写数据
        head_.store(next, std::memory_order_release);  // 发布: 数据可见后再更新 head
        return true;
    }

    // ------- 消费者: 弹出一个元素 (非阻塞) -------
    // 成功返回 true 并写入 item, 缓冲空返回 false
    // 内存序: acquire 读取 head_ (同步生产者释放), release 写入 tail_ (发布槽位可用)
    bool pop(T& item) {
        const size_type t = tail_.load(std::memory_order_relaxed);
        if (t == head_.load(std::memory_order_acquire)) {
            return false;  // 缓冲空
        }
        item = buffer_[t];  // 读数据 (acquire 保证看到生产者写入)
        tail_.store((t + 1) % N, std::memory_order_release);  // 发布: 槽位可复用
        return true;
    }

    // ------- 状态查询 -------
    // 是否为空 (并发下为近似值)
    bool empty() const {
        return head_.load(std::memory_order_acquire) ==
               tail_.load(std::memory_order_acquire);
    }

    // 是否已满 (并发下为近似值)
    bool full() const {
        const size_type h    = head_.load(std::memory_order_acquire);
        const size_type next = (h + 1) % N;
        return next == tail_.load(std::memory_order_acquire);
    }

    // 当前元素个数 (近似值, 并发下可能略有偏差, 仅供监控)
    size_type size() const {
        const size_type h = head_.load(std::memory_order_acquire);
        const size_type t = tail_.load(std::memory_order_acquire);
        return (h + N - t) % N;
    }

private:
    std::array<T, N> buffer_{};      // 静态分配数组, 零动态内存
    std::atomic<size_type> head_;    // 写位置 (生产者写, 消费者读)
    std::atomic<size_type> tail_;    // 读位置 (消费者写, 生产者读)
};

}  // namespace starship
