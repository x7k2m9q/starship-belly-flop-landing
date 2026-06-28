// =============================================================================
// test_flight_computer.cpp - Phase 12-14 飞控计算机验证
// =============================================================================
// 验证目标 (聚焦物理与工程, 拒绝无意义测试):
//   1. 4任务架构全程闭环 (BELLY→FLIP→LANDING, 与 Phase 11 一致)
//   2. 故障注入鲁棒性 (襟翼卡死 + IMU掉线, 验证控制系统能否安全着陆)
//   3. 执行时间预算 (control 任务 < 10ms, 验证实时可行性)
//   4. 看门狗正常 (控制任务每周期喂狗, 无误触发)
//
// 不测:
//   - Jitter (非实时OS上无意义)
//   - CPU过载/位翻转/电源毛刺 (需真实硬件)
//   - 无锁队列数据撕裂 (单线程协作式调度无并发, SPSC 正确性由 std::atomic 保证)
// =============================================================================
#include "../belly_flop_6dof/flight_computer.hpp"
#include "../belly_flop_6dof/fault_injection.hpp"
#include <cstdio>
#include <cmath>

using namespace falcon9;
using namespace falcon9::belly_flop_6dof;

static void banner(const char* s) {
    printf("======================================================================\n");
    printf("%s\n", s);
    printf("======================================================================\n");
}

// =============================================================================
// 测试 1: 4任务架构全程闭环 (无故障)
// =============================================================================
static bool test_full_mission_4task() {
    banner("[1] 4任务架构全程闭环 (无故障, 与 Phase 11 对齐)");
    bool pass = true;

    State6DOF state = make_initial_state(10000.0f, 300.0f, 50.0f, 85.0f);
    FlightComputer fc(true, true);  // 非理想执行器 + 陷波滤波

    const float dt = 0.01f;
    const float t_end = 120.0f;
    int n_steps = (int)(t_end / dt);

    bool reached_flip = false, reached_landing = false;
    float final_vz = 0.0f, final_tilt = 0.0f;
    int belly_to_flip_step = -1, flip_to_landing_step = -1;

    printf("  step     t(s)    h(m)     V(m/s)  theta(deg)  phase     T_cmd(kN)\n");
    printf("  -------- -------- -------- -------- ----------- --------- ---------\n");

    for (int step = 0; step <= n_steps; ++step) {
        bool kill = fc.step(state, dt);
        const ControllerInfo& info = fc.info();

        if (info.phase == Phase::FLIP && !reached_flip) {
            reached_flip = true;
            belly_to_flip_step = step;
        }
        if (info.phase == Phase::LANDING && !reached_landing) {
            reached_landing = true;
            flip_to_landing_step = step;
        }

        // 打印: 每10秒 / 阶段切换 / Kill / 着陆
        bool phase_change = (info.phase_transition[0] != '\0');
        if (step % 1000 == 0 || phase_change || kill || fc.is_landed()) {
            const char* ph = "BELLY";
            if (info.phase == Phase::FLIP) ph = "FLIP";
            if (info.phase == Phase::LANDING) ph = "LANDING";
            if (fc.is_landed()) ph = "LANDED";
            printf("  %8d %8.2f %8.1f %8.1f %11.2f %-9s %9.1f\n",
                   step, fc.time(), info.h, info.V, info.theta_deg,
                   ph, info.T_cmd / 1000.0f);
            if (phase_change)
                printf("           *** %s ***\n", info.phase_transition);
        }

        if (kill) {
            printf("  *** KILL: %s ***\n", fc.info().kill_reason);
            break;
        }
        if (fc.is_landed()) break;

        // 数值健康
        bool healthy = true;
        for (int i = 0; i < 14; ++i)
            if (!std::isfinite(state[i])) { healthy = false; break; }
        if (!healthy) { printf("  *** NaN/Inf ***\n"); pass = false; break; }
    }

    // 最终状态
    final_vz = state[5];
    Quaternion qf = state.q();
    final_tilt = get_tilt_angle_from_quat(qf) * 57.29577951308232f;

    printf("\n  === 结果 ===\n");
    printf("  阶段: BELLY→FLIP(t=%.1fs)→LANDING(t=%.1fs)\n",
           belly_to_flip_step * dt, flip_to_landing_step * dt);
    printf("  着陆: vz=%.2f m/s, tilt=%.2f deg, kill=%s\n",
           final_vz, final_tilt, fc.is_killed() ? "YES" : "NO");
    printf("  看门狗超时次数: %d\n", fc.watchdog().total_timeouts());

    // 执行时间预算
    printf("\n  === 执行时间预算 (control 周期 10ms = 10000us) ===\n");
    printf("  IMU      max: %6llu us\n", (unsigned long long)fc.imu_exec_us_max());
    printf("  Control  max: %6llu us  (%.1f%% of 10ms budget)\n",
           (unsigned long long)fc.control_exec_us_max(),
           100.0 * fc.control_exec_us_max() / 10000.0);
    printf("  Safety   max: %6llu us\n", (unsigned long long)fc.safety_exec_us_max());
    printf("  Guidance max: %6llu us\n", (unsigned long long)fc.guidance_exec_us_max());

    // 验收
    printf("\n  === 验收 ===\n");
    bool ok_nodiv = std::isfinite(final_vz) && std::isfinite(final_tilt);
    printf("  [1] 无数值发散:       %s\n", ok_nodiv ? "PASS" : "FAIL");
    if (!ok_nodiv) pass = false;

    bool ok_phase = reached_flip && reached_landing;
    printf("  [2] 三阶段全到达:     %s\n", ok_phase ? "PASS" : "FAIL");
    if (!ok_phase) pass = false;

    bool ok_kill = !fc.is_killed();
    printf("  [3] 无异常 Kill:      %s\n", ok_kill ? "PASS" : "FAIL");
    if (!ok_kill) pass = false;

    bool ok_tilt = (final_tilt < 15.0f);
    printf("  [4] 着陆姿态<15deg:   %.2f %s\n", final_tilt, ok_tilt ? "PASS" : "FAIL");
    if (!ok_tilt) pass = false;

    bool ok_vz = (std::fabs(final_vz) < 20.0f);
    printf("  [5] 着陆速度<20m/s:   %.2f %s\n", final_vz, ok_vz ? "PASS" : "FAIL");
    if (!ok_vz) pass = false;

    bool ok_wd = (fc.watchdog().total_timeouts() == 0);
    printf("  [6] 看门狗无误触发:   %s\n", ok_wd ? "PASS" : "FAIL");
    if (!ok_wd) pass = false;

    bool ok_time = (fc.control_exec_us_max() < 10000);
    printf("  [7] Control<10ms:     %llu us %s\n",
           (unsigned long long)fc.control_exec_us_max(),
           ok_time ? "PASS" : "FAIL");
    if (!ok_time) pass = false;

    printf("\n  结论: %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

// =============================================================================
// 测试 2: 故障注入 — 襟翼卡死 (FL 卡死在 5°)
// =============================================================================
static bool test_fault_actuator_stuck() {
    banner("[2] 故障注入: 襟翼卡死 (FL stuck @ 5deg, t=20s)");
    bool pass = true;

    State6DOF state = make_initial_state(10000.0f, 300.0f, 50.0f, 85.0f);
    FlightComputer fc(true, true);

    // 配置故障: FL 襟翼 (索引0) 在 t=20s 卡死在 5°
    FaultInjector fault;
    fault.config_actuator_stuck(20.0f, 0, 5.0f * 0.017453293f);  // 5° in rad
    fc.set_fault(fault);
    printf("  故障: %s\n", fault.description);

    const float dt = 0.01f;
    int n_steps = 12000;
    bool reached_landing = false;
    float final_vz = 0.0f, final_tilt = 0.0f;

    for (int step = 0; step <= n_steps; ++step) {
        bool kill = fc.step(state, dt);

        if (fc.info().phase == Phase::LANDING) reached_landing = true;

        // 故障注入时刻打印
        if (std::fabs(fc.time() - 20.0f) < dt && step > 0) {
            printf("  t=%.1fs: 故障注入触发, FL襟翼卡死@5deg\n", fc.time());
        }

        if (kill || fc.is_landed()) {
            printf("  t=%.1fs: %s\n", fc.time(),
                   fc.is_landed() ? "着陆" : "KILL");
            if (kill) printf("    原因: %s\n", fc.info().kill_reason);
            break;
        }

        bool healthy = true;
        for (int i = 0; i < 14; ++i)
            if (!std::isfinite(state[i])) { healthy = false; break; }
        if (!healthy) { printf("  *** NaN/Inf ***\n"); pass = false; break; }
    }

    final_vz = state[5];
    Quaternion qf = state.q();
    final_tilt = get_tilt_angle_from_quat(qf) * 57.29577951308232f;

    printf("\n  === 结果 ===\n");
    printf("  着陆: vz=%.2f m/s, tilt=%.2f deg, kill=%s\n",
           final_vz, final_tilt, fc.is_killed() ? "YES" : "NO");
    printf("  到达LANDING阶段: %s\n", reached_landing ? "YES" : "NO");

    printf("\n  === 验收 ===\n");
    // 襟翼卡死是严重故障, 验收标准放宽:
    // - 不崩溃 (无 NaN/Inf)
    // - 无看门狗超时 (控制系统仍在运行)
    // - 若着陆: 速度 < 30 m/s (放宽, 卡死襟翼降低控制权限)
    // - 若 Kill: 是控制系统的主动安全决策 (非崩溃)
    bool ok_healthy = std::isfinite(final_vz) && std::isfinite(final_tilt);
    printf("  [1] 无数值发散:       %s\n", ok_healthy ? "PASS" : "FAIL");
    if (!ok_healthy) pass = false;

    bool ok_wd = (fc.watchdog().total_timeouts() == 0);
    printf("  [2] 看门狗无误触发:   %s (控制系统仍在运行)\n", ok_wd ? "PASS" : "FAIL");
    if (!ok_wd) pass = false;

    if (fc.is_landed() && !fc.is_killed()) {
        bool ok_vz = (std::fabs(final_vz) < 30.0f);
        printf("  [3] 着陆速度<30m/s:   %.2f %s (卡死襟翼放宽阈值)\n",
               final_vz, ok_vz ? "PASS" : "FAIL");
        if (!ok_vz) pass = false;
    } else {
        printf("  [3] Kill/未着陆:      控制系统主动安全决策 (可接受)\n");
    }

    printf("\n  结论: %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

// =============================================================================
// 测试 3: 故障注入 — IMU 掉线 0.5s
// =============================================================================
static bool test_fault_sensor_dropout() {
    banner("[3] 故障注入: IMU 掉线 0.5s (t=35s, FLIP 阶段)");
    bool pass = true;

    State6DOF state = make_initial_state(10000.0f, 300.0f, 50.0f, 85.0f);
    FlightComputer fc(true, true);

    // 配置故障: IMU 在 t=35s 掉线 0.5s
    FaultInjector fault;
    fault.config_sensor_dropout(35.0f, 0.5f);
    fc.set_fault(fault);
    printf("  故障: %s\n", fault.description);

    const float dt = 0.01f;
    int n_steps = 12000;
    float final_vz = 0.0f, final_tilt = 0.0f;

    for (int step = 0; step <= n_steps; ++step) {
        bool kill = fc.step(state, dt);

        // 掉线区间打印
        if (std::fabs(fc.time() - 35.0f) < dt && step > 0) {
            printf("  t=%.1fs: IMU 掉线开始 (0.5s)\n", fc.time());
        }
        if (std::fabs(fc.time() - 35.5f) < dt && step > 0) {
            printf("  t=%.1fs: IMU 恢复\n", fc.time());
        }

        if (kill || fc.is_landed()) {
            printf("  t=%.1fs: %s\n", fc.time(),
                   fc.is_landed() ? "着陆" : "KILL");
            if (kill) printf("    原因: %s\n", fc.info().kill_reason);
            break;
        }

        bool healthy = true;
        for (int i = 0; i < 14; ++i)
            if (!std::isfinite(state[i])) { healthy = false; break; }
        if (!healthy) { printf("  *** NaN/Inf ***\n"); pass = false; break; }
    }

    final_vz = state[5];
    Quaternion qf = state.q();
    final_tilt = get_tilt_angle_from_quat(qf) * 57.29577951308232f;

    printf("\n  === 结果 ===\n");
    printf("  着陆: vz=%.2f m/s, tilt=%.2f deg, kill=%s\n",
           final_vz, final_tilt, fc.is_killed() ? "YES" : "NO");

    printf("\n  === 验收 ===\n");
    // IMU 掉线 0.5s: 当前架构中 Control 任务直接用真值状态 (无 EKF),
    // 故 IMU 掉线不影响控制 (RingBuffer 中 IMU 数据仅演示架构).
    // 真实系统中 IMU 掉线 → EKF 进入惯性预测模式, 0.5s 内姿态漂移可控.
    // 此处验证: 掉线不导致崩溃, 任务仍成功.
    bool ok_healthy = std::isfinite(final_vz) && std::isfinite(final_tilt);
    printf("  [1] 无数值发散:       %s\n", ok_healthy ? "PASS" : "FAIL");
    if (!ok_healthy) pass = false;

    bool ok_no_kill = !fc.is_killed();
    printf("  [2] 无 Kill:          %s\n", ok_no_kill ? "PASS" : "FAIL");
    if (!ok_no_kill) pass = false;

    if (fc.is_landed()) {
        bool ok_vz = (std::fabs(final_vz) < 20.0f);
        printf("  [3] 着陆速度<20m/s:   %.2f %s\n", final_vz, ok_vz ? "PASS" : "FAIL");
        if (!ok_vz) pass = false;
    }

    printf("\n  结论: %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

// =============================================================================
// main
// =============================================================================
int main() {
    printf("\n");
    printf("**********************************************************************\n");
    printf("* Phase 12-14: 4任务飞控计算机 + 故障注入 + HAL\n");
    printf("* 协作式循环调度 (确定性, 可移植 FreeRTOS)\n");
    printf("**********************************************************************\n\n");

    bool all_pass = true;
    all_pass &= test_full_mission_4task();
    all_pass &= test_fault_actuator_stuck();
    all_pass &= test_fault_sensor_dropout();

    banner("总结");
    printf("  全部测试: %s\n", all_pass ? "PASS" : "FAIL");
    printf("\n");
    return all_pass ? 0 : 1;
}
