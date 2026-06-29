// =============================================================================
// actuator_nonideal_6dof.hpp - 星舰 6-DOF 非理想执行器 (Phase 11 移植)
// =============================================================================
// 对应 Python: src/belly_flop/actuator_nonideal_6dof.py + src/starship_non_ideal.py
//
// 理论方案 7.0/9.0:
//   - 问题 20: 襟翼速率限制 (30°/s)
//   - 问题 21: Bouc-Wen 模型 (Phase 9.0 修复: 标准公式 y=α·x+(1-α)·z)
//   - 问题 22: 死区补偿 (偏置 + Dither, Dither 幅值 > 死区最大值)
//
// 4 片独立非理想执行器: FL/FR/RL/RR
//   每片独立参数: 死区 / Bouc-Wen / 速率限制
//
// 死区补偿策略 (7.0 铁律):
//   "死区补偿 (偏置+Dither) 必须加在控制分配之后、实际指令输出前"
//   Phase 8.0 缺陷 27 证明: 配平点死区不补偿会导致姿态完全发散
//
// 补偿方法:
//   1. 偏置: 对小指令 (|δ|<dead_zone) 施加偏置, 使其越过死区
//   2. Dither: 高频抖动信号, 幅值 > 死区最大值, 频率高于控制带宽
//      - Dither 频率: 20Hz (高于控制带宽 0.5Hz, 低于结构模态 2.35Hz)
//      - Dither 幅值: 1.0° (> 死区最大值 0.8°)
//      - Dither 使执行器在小指令区持续微振, 线性化死区特性
//
// 缺陷清单:
//   缺陷 29: Bouc-Wen gamma 符号修正 (+0.5, 标准公式 z 有界)
//   缺陷 37: 速率限制在死区补偿之前 (否则大指令被限到 < 死区被吞掉)
//   缺陷 38: TVC 延迟缓冲区未填满时返回 0 (纯延迟的物理特性)
//
// 零动态内存: 所有缓冲区为定长数组, 无 new/malloc/vector
// =============================================================================
#ifndef STARSHIP_BELLY_FLOP_6DOF_ACTUATOR_NONIDEAL_HPP
#define STARSHIP_BELLY_FLOP_6DOF_ACTUATOR_NONIDEAL_HPP

#include <cmath>
#include "aero_6dof.hpp"

namespace starship {
namespace belly_flop_6dof {

// =============================================================================
// 襟翼伺服电机: 死区 + Bouc-Wen 摩擦滞环
// =============================================================================
// 物理背景:
//   星舰襟翼由电机 + 减速器驱动, 存在齿轮间隙与静摩擦.
//   - 死区 (±0.5°): 小指令无法克服静摩擦, 电机不响应.
//     在大攻角配平点 (α≈85°) 附近, 配平襟翼角接近死区边界,
//     控制器不断发小修正指令却无输出 → 累积误差 → 突然大修正,
//     形成高频极限环振荡 (limit cycle).
//   - Bouc-Wen 滞环: 描述摩擦力的记忆效应. 内部状态 z 跟踪
//     滞回回线, 使正向与反向运动产生不同输出, 造成相位滞后.
//
// Bouc-Wen 模型 (Phase 9.0 修正, 缺陷 29):
//   dz/dt = A·(dx/dt) - β·|dx/dt|·|z|^(n-1)·z - γ·(dx/dt)·|z|^n
//   实际输出 = α·x + (1-α)·z  (标准 Bouc-Wen: α=线性分量比)
//   z 稳态幅度 = A/(β+γ) = 1.0/1.0 = 1.0 rad (有界)
// =============================================================================
class FlapServoNonIdeal {
public:
    // Bouc-Wen 参数
    float alpha;    // 线性刚度比 (0=纯滞环, 1=纯线性), 默认 0.3
    float A;        // Bouc-Wen 滞环斜率
    float beta;     // Bouc-Wen 回线参数 (正向)
    float gamma;    // Bouc-Wen 回线参数 (Phase 9.0: 正值)
    int   n;        // Bouc-Wen 平滑度指数 (n=1 经典线性滞环)

    // 内部状态
    float z;        // Bouc-Wen 滞环内部变量
    float x_prev;   // 上一拍指令 (用于数值微分)

    FlapServoNonIdeal(float alpha_ = 0.3f, float A_ = 1.0f,
                      float beta_ = 0.5f, float gamma_ = 0.5f, int n_ = 1)
        : alpha(alpha_), A(A_), beta(beta_), gamma(gamma_), n(n_),
          z(0.0f), x_prev(0.0f) {}

    void reset() { z = 0.0f; x_prev = 0.0f; }

    // 更新襟翼伺服输出 (无内部死区, 死区在外部补偿处理)
    // delta_cmd: 指令偏转角 (rad)
    // dt: 时间步长 (s)
    // 返回: 实际偏转角 (rad), 已限幅到 ±DELTA_MAX
    float update(float delta_cmd, float dt) {
        if (dt < 1e-10f) dt = 1e-10f;  // 防除零

        // Bouc-Wen 滞环模型
        // 输入速度 (数值微分)
        float dx_dt = (delta_cmd - x_prev) / dt;

        // Bouc-Wen 微分方程 (n=1 简化):
        //   dz/dt = A·(dx/dt) - β·|dx/dt|·z - γ·(dx/dt)·|z|
        float abs_dx = std::fabs(dx_dt);
        float abs_z  = std::fabs(z);
        // n=1: |z|^n = |z|, |z|^(n-1)·z = z
        float dz_dt = A * dx_dt
                      - beta * abs_dx * z
                      - gamma * dx_dt * abs_z;

        // Euler 积分
        z += dz_dt * dt;

        // 实际输出 = α·指令 + (1-α)·滞环变量 (Phase 9.0 标准公式)
        float delta_actual = alpha * delta_cmd + (1.0f - alpha) * z;

        // 限幅 ±DELTA_MAX
        if (delta_actual >  DELTA_MAX) delta_actual =  DELTA_MAX;
        if (delta_actual < -DELTA_MAX) delta_actual = -DELTA_MAX;

        // 更新状态
        x_prev = delta_cmd;

        return delta_actual;
    }
};

// =============================================================================
// 4 片独立非理想襟翼执行器 (FL/FR/RL/RR)
// =============================================================================
// 每片独立:
//   - 死区 (0.3°~0.8°, 制造公差)
//   - Bouc-Wen 滞环 (α=0.3, A=1.0, β=0.5, γ=0.5)
//   - 速率限制 (30°/s)
//
// 死区补偿:
//   - 偏置 + Dither, 加在控制分配之后、实际指令输出前
//   - Dither: 20Hz 正弦, 幅值 1.0° (> 死区最大值 0.8°)
//
// 流程 (缺陷 37 修正: 速率限制 → 死区补偿 → Bouc-Wen):
//   1. 速率限制: 斜坡跟踪目标, |δ_target[k+1]-δ_target[k]| ≤ rate·dt
//   2. 死区补偿: 偏置 + Dither (确保信号越过死区)
//   3. Bouc-Wen 滞环: 标准公式 y=α·x+(1-α)·z (无内部死区)
//   4. 限幅 ±DELTA_MAX
//
// 缺陷 37: 速率限制在死区补偿之前, 否则大指令被限制到 < 死区被吞掉
//   原顺序: 补偿→速率限制→Bouc-Wen(含死区) → 5°指令被限到 0.3°<死区 0.5°→全 0
//   修正顺序: 速率限制→补偿→Bouc-Wen(无死区) → 5°指令斜坡跟踪, 补偿确保越过死区
// =============================================================================
class FlapActuatorSuite6DOF {
public:
    static constexpr int N_FLAPS = 4;       // 4 片襟翼
    static constexpr float DITHER_FREQ = 20.0f;   // Hz
    static constexpr float DITHER_AMP  = 0.017453292519943295f;  // 1° in rad

    // 4 片独立参数
    float dead_zones[N_FLAPS];       // 死区半宽 (rad)
    float rate_limits[N_FLAPS];      // 速率限制 (rad/s)
    bool  use_compensation;          // 是否启用死区补偿

    // 4 片独立 Bouc-Wen 伺服 (死区=0, 死区在外部补偿处理)
    FlapServoNonIdeal servos[N_FLAPS];

    // 4 片速率限制跟踪目标 (斜坡跟踪)
    float delta_target[N_FLAPS];

    // Dither 相位 (每片独立, 避免同步)
    float dither_phases[N_FLAPS];

    // 当前时间
    float t;

    FlapActuatorSuite6DOF()
        : use_compensation(true), t(0.0f) {
        const float DEG2RAD = 0.017453292519943295f;
        const float DZ_DEFAULT = 0.5f * DEG2RAD;      // 0.5°
        const float RATE_DEFAULT = 30.0f * DEG2RAD;   // 30°/s
        for (int i = 0; i < N_FLAPS; ++i) {
            dead_zones[i] = DZ_DEFAULT;
            rate_limits[i] = RATE_DEFAULT;
            delta_target[i] = 0.0f;
            // Dither 相位: 0, π/2, π, 3π/2 (每片独立)
            dither_phases[i] = i * 3.14159265358979f * 0.5f;
        }
    }

    void reset() {
        for (int i = 0; i < N_FLAPS; ++i) {
            servos[i].reset();
            delta_target[i] = 0.0f;
        }
        t = 0.0f;
    }

    // =========================================================================
    // 更新 4 片襟翼执行器
    // =========================================================================
    // delta_cmds: 4 片指令偏转 [d_FL, d_FR, d_RL, d_RR] (rad)
    // dt: 时间步长 (s)
    // delta_actuals: 4 片实际偏转 (rad), 已限幅 ±DELTA_MAX
    // =========================================================================
    void update(const float delta_cmds[4], float dt,
                float delta_actuals[4]) {
        t += dt;
        const float PI = 3.14159265358979f;

        for (int i = 0; i < N_FLAPS; ++i) {
            float cmd = delta_cmds[i];

            // ---- 1. 速率限制 (斜坡跟踪) ----
            // delta_target 以最大速率逼近 cmd
            float max_delta = rate_limits[i] * dt;
            float delta_error = cmd - delta_target[i];
            float delta_step = delta_error;
            if (delta_step >  max_delta) delta_step =  max_delta;
            if (delta_step < -max_delta) delta_step = -max_delta;
            delta_target[i] += delta_step;
            float delta_rated = delta_target[i];

            // ---- 2. 死区补偿 (偏置 + Dither) ----
            float cmd_compensated;
            if (use_compensation) {
                // Dither: 高频正弦, 幅值 > 死区最大值
                float dither = DITHER_AMP * std::sin(
                    2.0f * PI * DITHER_FREQ * t + dither_phases[i]);
                // 偏置: 对小指令施加偏置, 使其越过死区
                if (std::fabs(delta_rated) < dead_zones[i] * 2.0f) {
                    float bias = 0.0f;
                    if (std::fabs(delta_rated) > 1e-10f) {
                        bias = (delta_rated > 0.0f ? 1.0f : -1.0f) * dead_zones[i];
                    }
                    cmd_compensated = delta_rated + bias + dither;
                } else {
                    cmd_compensated = delta_rated + dither;
                }
            } else {
                // 无补偿: 直接施加死区
                if (std::fabs(delta_rated) < dead_zones[i]) {
                    cmd_compensated = 0.0f;
                } else {
                    cmd_compensated = delta_rated;
                }
            }

            // ---- 3. Bouc-Wen 滞环 (无内部死区, dead_zone=0) ----
            delta_actuals[i] = servos[i].update(cmd_compensated, dt);
        }
    }
};

// =============================================================================
// TVC 推力矢量控制非理想模型 (延迟 + 速率限制 + 限幅)
// =============================================================================
// 包含:
//   - 纯延迟 (80ms, 定长环形缓冲区实现)
//   - 速率限制 (20°/s)
//   - 限幅 (±10°)
//
// 缺陷 38: TVC 延迟缓冲区未填满时返回 0, 模拟纯延迟的物理特性.
//   启动阶段 (t < delay) 返回 0, 而非当前指令.
// =============================================================================
class TVCActuator6DOF {
public:
    static constexpr int BUFFER_SIZE = 16;  // 覆盖 delay/dt + 冗余

    float delay;           // 纯延迟 (s), 默认 0.08
    float rate_limit;      // 速率限制 (rad/s), 默认 20°/s
    float gimbal_limit;    // 最大偏转 (rad), 默认 10°

    // 双轴延迟缓冲区 (定长环形缓冲区, 零动态内存)
    float buf_y[BUFFER_SIZE];   // gimbal_y 历史值
    float buf_z[BUFFER_SIZE];   // gimbal_z 历史值
    int   buf_head;             // 写入位置
    int   buf_count;            // 已填充样本数

    // 上一拍输出 (用于速率限制)
    float gimbal_prev[2];

    // 当前时间
    float t;

    // 延迟样本数 (根据 delay 和 dt 计算)
    int delay_samples;

    TVCActuator6DOF(float delay_ = 0.08f,
                    float rate_limit_ = 0.34906585039886590f,    // 20° in rad
                    float gimbal_limit_ = 0.17453292519943295f,  // 10° in rad
                    float dt = 0.01f)
        : delay(delay_), rate_limit(rate_limit_), gimbal_limit(gimbal_limit_),
          buf_head(0), buf_count(0), t(0.0f) {
        gimbal_prev[0] = 0.0f;
        gimbal_prev[1] = 0.0f;
        for (int i = 0; i < BUFFER_SIZE; ++i) {
            buf_y[i] = 0.0f;
            buf_z[i] = 0.0f;
        }
        // 延迟样本数 = round(delay / dt)
        delay_samples = (int)(delay / dt + 0.5f);
        if (delay_samples < 1) delay_samples = 1;
        if (delay_samples >= BUFFER_SIZE) delay_samples = BUFFER_SIZE - 1;
    }

    void reset() {
        buf_head = 0;
        buf_count = 0;
        t = 0.0f;
        gimbal_prev[0] = 0.0f;
        gimbal_prev[1] = 0.0f;
        for (int i = 0; i < BUFFER_SIZE; ++i) {
            buf_y[i] = 0.0f;
            buf_z[i] = 0.0f;
        }
    }

    // =========================================================================
    // 更新 TVC 偏转
    // =========================================================================
    // gimbal_cmds: [gimbal_y, gimbal_z] 指令 (rad)
    // dt: 时间步长 (s)
    // gimbal_actuals: [gimbal_y, gimbal_z] 实际偏转 (rad)
    //
    // 缺陷 38: 启动阶段 (t < delay) 返回 0, 模拟纯延迟的物理特性.
    // =========================================================================
    void update(const float gimbal_cmds[2], float dt,
                float gimbal_actuals[2]) {
        t += dt;
        gimbal_actuals[0] = 0.0f;
        gimbal_actuals[1] = 0.0f;

        // 缺陷 38: 启动阶段 t < delay 时返回 0 (纯延迟的物理特性)
        if (t < delay) {
            return;
        }

        // 写入当前指令到环形缓冲区
        buf_y[buf_head] = gimbal_cmds[0];
        buf_z[buf_head] = gimbal_cmds[1];
        buf_head = (buf_head + 1) % BUFFER_SIZE;
        if (buf_count < BUFFER_SIZE) buf_count++;

        // 读取 delay_samples 前的样本 (延迟后的指令)
        // 读位置 = (写位置 - delay_samples) mod BUFFER_SIZE
        int read_pos = (buf_head - delay_samples + BUFFER_SIZE) % BUFFER_SIZE;
        float delayed_y = buf_y[read_pos];
        float delayed_z = buf_z[read_pos];

        // 速率限制 (双轴独立)
        float max_delta = rate_limit * dt;
        float limited_y = delayed_y;
        if (limited_y > gimbal_prev[0] + max_delta) limited_y = gimbal_prev[0] + max_delta;
        if (limited_y < gimbal_prev[0] - max_delta) limited_y = gimbal_prev[0] - max_delta;
        float limited_z = delayed_z;
        if (limited_z > gimbal_prev[1] + max_delta) limited_z = gimbal_prev[1] + max_delta;
        if (limited_z < gimbal_prev[1] - max_delta) limited_z = gimbal_prev[1] - max_delta;

        // 限幅
        if (limited_y >  gimbal_limit) limited_y =  gimbal_limit;
        if (limited_y < -gimbal_limit) limited_y = -gimbal_limit;
        if (limited_z >  gimbal_limit) limited_z =  gimbal_limit;
        if (limited_z < -gimbal_limit) limited_z = -gimbal_limit;

        gimbal_actuals[0] = limited_y;
        gimbal_actuals[1] = limited_z;
        gimbal_prev[0] = limited_y;
        gimbal_prev[1] = limited_z;
    }
};

}  // namespace belly_flop_6dof
}  // namespace starship

#endif  // STARSHIP_BELLY_FLOP_6DOF_ACTUATOR_NONIDEAL_HPP
