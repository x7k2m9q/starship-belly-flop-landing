// =============================================================================
// test_belly_flop_6dof.cpp - 星舰 6-DOF C++ 验证 (Phase 11)
// =============================================================================
// 验证目标 (聚焦关键物理, 拒绝无意义测试):
//   1. 欧拉角↔四元数转换正确性 (Belly-Flop θ=0°/85°/90° 三点验证)
//   2. omega 反馈通道完整性 (猎鹰9号血泪教训: 严禁置零)
//   3. 全程闭环 BELLY→FLIP→LANDING (物理稳定性 + 无发散)
//
// 编译:
//   cmake -B build && cmake --build build --target starship_6dof_test
// =============================================================================
#include "../belly_flop_6dof/phase_controller_6dof.hpp"
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
// 测试 1: 欧拉角 ↔ 四元数转换 (Belly-Flop 定义验证)
// =============================================================================
// θ=0°  → 垂直 (头朝上), 应等于 Q_VERT = [√2/2, 0, √2/2, 0]
// θ=90° → 水平 (腹部朝下), 应接近 identity = [1, 0, 0, 0]
// θ=85° → BELLY 姿态, tilt≈85°
// =============================================================================
static bool test_quat_euler_conversion() {
    banner("[1] 欧拉角 <-> 四元数转换 (Belly-Flop 定义)");
    bool pass = true;

    // θ=0°: 垂直, 应等于 Q_VERT
    {
        Quaternion q = euler_angle_to_quat(0.0f);
        float tilt = get_tilt_angle_from_quat(q) * 57.29577951308232f;
        printf("  theta=0deg  -> q=[%.6f, %.6f, %.6f, %.6f]  tilt=%.4fdeg",
               q.w, q.x, q.y, q.z, tilt);
        // 验证: tilt≈0, 且 q≈Q_VERT
        bool ok = (tilt < 1.0f) &&
                  (std::fabs(q.w - 0.7071067811865476f) < 1e-5f) &&
                  (std::fabs(q.y - 0.7071067811865476f) < 1e-5f);
        printf("  %s\n", ok ? "OK" : "FAIL");
        if (!ok) pass = false;
    }

    // θ=90°: 水平, 应接近 identity
    {
        Quaternion q = euler_angle_to_quat(90.0f);
        float tilt = get_tilt_angle_from_quat(q) * 57.29577951308232f;
        printf("  theta=90deg -> q=[%.6f, %.6f, %.6f, %.6f]  tilt=%.4fdeg",
               q.w, q.x, q.y, q.z, tilt);
        // 验证: tilt≈90, 且 q≈[1,0,0,0]
        bool ok = (std::fabs(tilt - 90.0f) < 1.0f) &&
                  (std::fabs(q.w - 1.0f) < 1e-5f);
        printf("  %s\n", ok ? "OK" : "FAIL");
        if (!ok) pass = false;
    }

    // θ=85°: BELLY 姿态, tilt≈85°
    {
        Quaternion q = euler_angle_to_quat(85.0f);
        float tilt = get_tilt_angle_from_quat(q) * 57.29577951308232f;
        float pitch = get_pitch_angle_from_quat(q) * 57.29577951308232f;
        printf("  theta=85deg -> q=[%.6f, %.6f, %.6f, %.6f]  tilt=%.4fdeg  pitch=%.4fdeg",
               q.w, q.x, q.y, q.z, tilt, pitch);
        // 验证: tilt≈85, pitch≈85
        bool ok = (std::fabs(tilt - 85.0f) < 1.0f) &&
                  (std::fabs(pitch - 85.0f) < 1.0f);
        printf("  %s\n", ok ? "OK" : "FAIL");
        if (!ok) pass = false;
    }

    printf("\n  结论: %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

// =============================================================================
// 测试 2: omega 反馈通道完整性 (猎鹰9号血泪教训)
// =============================================================================
// 猎鹰9号 C++ 移植 Bug: omega 反馈通道丢失 → PD 退化为 P → 姿态发散
//   (开发日志: tilt 从 0.6° → 65.8° at t=14, 火箭失控)
//
// 验证方法:
//   构造一个含非零 omega 的状态, 计算 state_derivative,
//   检查 domega 中是否包含陀螺耦合项 omega × (I·omega).
//   若 omega 被错误置零, 陀螺耦合项 = 0, domega 仅由 M/I 决定.
// =============================================================================
static bool test_omega_feedback_channel() {
    banner("[2] omega 反馈通道完整性 (猎鹰9号血泪教训)");
    bool pass = true;

    // 构造状态: 85° 俯仰, 非零 omega (p=0.1, q=0.5, r=0.05 rad/s)
    State6DOF s = make_initial_state(10000.0f, 300.0f, 50.0f, 85.0f);
    s.set_omega_b(0.1f, 0.5f, 0.05f);   // ← 非零 omega

    // 零襟翼, 零推力, 零 TVC (隔离陀螺耦合项)
    float delta_zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float tvc_zero[2] = {0.0f, 0.0f};

    State6DOF d = state_derivative(s, 0.0f, delta_zero, tvc_zero);

    // 提取 omega 和 domega
    Vec3f omega = s.omega_b();
    Vec3f domega;
    domega[0] = d[10]; domega[1] = d[11]; domega[2] = d[12];

    // 计算预期陀螺耦合项: omega × (I·omega)
    InertiaTensor I = get_inertia_tensor(s.m_fuel());
    Vec3f I_omega;
    I_omega[0] = I.Ixx * omega[0];
    I_omega[1] = I.Iyy * omega[1];
    I_omega[2] = I.Izz * omega[2];
    Vec3f gyro = omega.cross(I_omega);

    printf("  omega = [%.4f, %.4f, %.4f] rad/s\n", omega[0], omega[1], omega[2]);
    printf("  I     = [Ixx=%.0f, Iyy=%.0f, Izz=%.0f]\n", I.Ixx, I.Iyy, I.Izz);
    printf("  I*omega = [%.4f, %.4f, %.4f]\n", I_omega[0], I_omega[1], I_omega[2]);
    printf("  gyro = omega x (I*omega) = [%.4f, %.4f, %.4f]\n",
           gyro[0], gyro[1], gyro[2]);
    printf("  domega = [%.6f, %.6f, %.6f] rad/s^2\n", domega[0], domega[1], domega[2]);

    // 验证: domega 应包含陀螺耦合项的影响
    // 若 omega 被置零, gyro=0, 则 domega 仅由气动力矩/I 决定 (此处气动力矩很小)
    // 若 omega 正确传入, gyro≠0, domega 应显著非零
    float domega_mag = std::sqrt(domega[0]*domega[0] +
                                  domega[1]*domega[1] +
                                  domega[2]*domega[2]);
    float gyro_mag = std::sqrt(gyro[0]*gyro[0] +
                                gyro[1]*gyro[1] +
                                gyro[2]*gyro[2]);

    printf("  |domega| = %.6f, |gyro| = %.6f\n", domega_mag, gyro_mag);

    // 关键验证: |domega| 应该与 |gyro|/I 同量级 (气动力矩在此状态很小)
    // 若 omega 被置零, gyro=0, domega 会非常小 (仅气动力矩)
    bool omega_channel_ok = (gyro_mag > 1e-6f) && (domega_mag > 1e-8f);
    printf("  omega 通道: %s\n", omega_channel_ok ? "OK (陀螺耦合项已生效)" : "FAIL (omega 被置零!)");
    if (!omega_channel_ok) pass = false;

    // 补充验证: 对比 omega=0 和 omega≠0 的 domega 差异
    State6DOF s_zero_omega = s;
    s_zero_omega.set_omega_b(0.0f, 0.0f, 0.0f);
    State6DOF d_zero = state_derivative(s_zero_omega, 0.0f, delta_zero, tvc_zero);
    Vec3f domega_zero;
    domega_zero[0] = d_zero[10]; domega_zero[1] = d_zero[11]; domega_zero[2] = d_zero[12];
    float domega_zero_mag = std::sqrt(domega_zero[0]*domega_zero[0] +
                                       domega_zero[1]*domega_zero[1] +
                                       domega_zero[2]*domega_zero[2]);
    printf("  对比: |domega(omega=0)| = %.6f vs |domega(omega!=0)| = %.6f\n",
           domega_zero_mag, domega_mag);

    bool diff_ok = (domega_mag > domega_zero_mag * 1.1f);
    printf("  差异验证: %s\n", diff_ok ? "OK (omega 影响显著)" : "WARN (omega 影响不显著)");
    if (!diff_ok) pass = false;

    printf("\n  结论: %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

// =============================================================================
// 测试 3: 全程闭环 BELLY → FLIP → LANDING
// =============================================================================
// 验证:
//   1. 仿真不崩溃 (无 NaN/Inf)
//   2. 阶段切换正常 (BELLY→FLIP→LANDING)
//   3. 着陆时姿态接近垂直 (tilt < 15°)
//   4. 着陆速度合理 (vz < 20 m/s)
//   5. 无 Kill 触发
// =============================================================================
static bool test_full_mission() {
    banner("[3] 全程闭环 BELLY -> FLIP -> LANDING");
    bool pass = true;

    // 初始状态: 10km, 300m/s 下降, 50m/s 水平, 85° 俯仰, 70% 燃料
    State6DOF state = make_initial_state(10000.0f, 300.0f, 50.0f, 85.0f);

    // 控制器 (启用非理想执行器 + 陷波滤波)
    PhaseController6DOF controller(true, true);

    const float dt = 0.01f;
    const float t_end = 120.0f;
    int n_steps = (int)(t_end / dt);

    // 状态跟踪
    bool reached_belly = false;
    bool reached_flip = false;
    bool reached_landing = false;
    bool kill_triggered = false;
    float final_h = 0.0f, final_vz = 0.0f, final_tilt_deg = 0.0f;
    float final_t = 0.0f;
    int belly_to_flip_step = -1;
    int flip_to_landing_step = -1;

    printf("  step     t(s)    h(m)     V(m/s)  theta(deg)  phase     T_cmd(kN)\n");
    printf("  -------- -------- -------- -------- ----------- --------- ---------\n");

    for (int step = 0; step <= n_steps; ++step) {
        float t = step * dt;

        // 控制器更新
        float T_cmd, delta_flaps[4], tvc_gimbal[2];
        ControllerInfo info;
        bool kill = controller.update(state, dt, T_cmd, delta_flaps,
                                       tvc_gimbal, info);

        // 阶段跟踪
        if (info.phase == Phase::BELLY) reached_belly = true;
        if (info.phase == Phase::FLIP && !reached_flip) {
            reached_flip = true;
            belly_to_flip_step = step;
        }
        if (info.phase == Phase::LANDING && !reached_landing) {
            reached_landing = true;
            flip_to_landing_step = step;
        }

        // 每 10秒 或 阶段切换 或 Kill 时打印
        bool phase_change = (info.phase_transition[0] != '\0');
        if (step % 1000 == 0 || phase_change || kill ||
            (step % 100 == 0 && info.h < 1000.0f)) {
            const char* phase_str = "BELLY";
            if (info.phase == Phase::FLIP) phase_str = "FLIP";
            if (info.phase == Phase::LANDING) phase_str = "LANDING";
            printf("  %8d %8.2f %8.1f %8.1f %11.2f %-9s %9.1f\n",
                   step, t, info.h, info.V, info.theta_deg,
                   phase_str, T_cmd / 1000.0f);
            if (phase_change) {
                printf("           *** 阶段切换: %s ***\n", info.phase_transition);
            }
        }

        if (kill) {
            kill_triggered = true;
            printf("  *** KILL 触发: %s ***\n", info.kill_reason);
            final_h = info.h;
            final_vz = state[5];
            final_tilt_deg = info.theta_deg;
            final_t = t;
            break;
        }

        // 动力学积分
        state = rk4_step(state, T_cmd, delta_flaps, dt, tvc_gimbal);

        // 着陆判断
        float h = -state[2];
        if (h <= 0.0f) {
            final_h = 0.0f;
            final_vz = state[5];
            Quaternion q_final = state.q();
            final_tilt_deg = get_tilt_angle_from_quat(q_final) * 57.29577951308232f;
            final_t = t + dt;
            printf("  %8d %8.2f %8.1f %8.1f %11.2f %-9s %9.1f\n",
                   step + 1, final_t, 0.0f,
                   std::sqrt(state[3]*state[3]+state[4]*state[4]+state[5]*state[5]),
                   final_tilt_deg, "LANDED", T_cmd / 1000.0f);
            break;
        }

        // 数值健康检查 (无 NaN/Inf)
        bool healthy = true;
        for (int i = 0; i < 14; ++i) {
            if (!std::isfinite(state[i])) { healthy = false; break; }
        }
        if (!healthy) {
            printf("  *** 数值发散 (NaN/Inf) at step %d, t=%.2f ***\n", step, t);
            pass = false;
            break;
        }

        final_h = h;
        final_vz = state[5];
        Quaternion q_final = state.q();
        final_tilt_deg = get_tilt_angle_from_quat(q_final) * 57.29577951308232f;
        final_t = t;
    }

    // ============ 结果分析 ============
    printf("\n  === 结果分析 ===\n");
    printf("  阶段切换:\n");
    printf("    BELLY:     %s\n", reached_belly ? "已进入" : "未进入");
    printf("    FLIP:      %s", reached_flip ? "已进入" : "未进入");
    if (belly_to_flip_step > 0)
        printf(" (t=%.2fs)", belly_to_flip_step * dt);
    printf("\n");
    printf("    LANDING:   %s", reached_landing ? "已进入" : "未进入");
    if (flip_to_landing_step > 0)
        printf(" (t=%.2fs)", flip_to_landing_step * dt);
    printf("\n");

    printf("  最终状态:\n");
    printf("    t          = %.2f s\n", final_t);
    printf("    h          = %.1f m\n", final_h);
    printf("    vz         = %.2f m/s\n", final_vz);
    printf("    tilt       = %.2f deg\n", final_tilt_deg);
    printf("    kill       = %s\n", kill_triggered ? "YES" : "NO");

    // 验收标准
    printf("\n  === 验收 ===\n");

    // 1. 无 NaN/Inf
    bool no_divergence = std::isfinite(final_h) && std::isfinite(final_vz) &&
                         std::isfinite(final_tilt_deg);
    printf("  [1] 无数值发散:    %s\n", no_divergence ? "PASS" : "FAIL");
    if (!no_divergence) pass = false;

    // 2. 阶段切换正常
    bool phase_ok = reached_belly && reached_flip && reached_landing;
    printf("  [2] 三阶段全到达:  %s\n", phase_ok ? "PASS" : "FAIL");
    if (!phase_ok) pass = false;

    // 3. 无 Kill (或 Kill 仅在着陆后)
    bool no_kill = !kill_triggered;
    printf("  [3] 无异常 Kill:   %s\n", no_kill ? "PASS" : "FAIL");
    if (!no_kill) pass = false;

    // 4. 着陆姿态 (tilt < 15°)
    bool attitude_ok = (final_tilt_deg < 15.0f);
    printf("  [4] 着陆姿态 tilt<15deg: %.2f deg  %s\n",
           final_tilt_deg, attitude_ok ? "PASS" : "FAIL");
    if (!attitude_ok) pass = false;

    // 5. 着陆速度 (|vz| < 20 m/s)
    bool velocity_ok = (std::fabs(final_vz) < 20.0f);
    printf("  [5] 着陆速度 |vz|<20:  %.2f m/s  %s\n",
           final_vz, velocity_ok ? "PASS" : "FAIL");
    if (!velocity_ok) pass = false;

    printf("\n  结论: %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

// =============================================================================
// 主函数
// =============================================================================
int main() {
    printf("\n");
    printf("**********************************************************************\n");
    printf("* 星舰 6-DOF 回收仿真 C++ 验证 (Phase 11)\n");
    printf("* Python -> C++ 移植: aero + dynamics + control + actuator + phase\n");
    printf("**********************************************************************\n\n");

    bool all_pass = true;

    all_pass &= test_quat_euler_conversion();
    all_pass &= test_omega_feedback_channel();
    all_pass &= test_full_mission();

    banner("总结");
    printf("  全部测试: %s\n", all_pass ? "PASS" : "FAIL");
    printf("\n");
    return all_pass ? 0 : 1;
}
