// =============================================================================
// attitude_control_6dof.hpp - 星舰 6-DOF 姿态控制器 (Phase 11 移植)
// =============================================================================
// 对应 Python: src/belly_flop/attitude_control_6dof.py
//
// 理论方案 7.0 问题 17-18:
//   - 问题 17: PD 增益在 6DOF 下需重新整定 (Kp=2·I·wn², Kd=2·ζ·wn·I)
//   - 问题 18: 四元数控制律双覆盖错误 (sign(qw) 处理)
//
// 控制律: 四元数误差 PD (复用前项目架构)
//   e_q = q_actual^{-1} ⊗ q_des   (body 系误差, 修复 10)
//   e_vec = e_q[1:4]               (矢量部分, ≈ θ_err/2)
//   sign(qw) 处理双覆盖: w<0 整体取反
//   M_cmd = Kp·e_vec + Kd·e_omega + I·omega_des_dot
//
// 增益物理整定 (自适应惯量):
//   Kp = 2·I·wn²  (补偿 e_vec=θ/2 因子, 使闭环为标准二阶)
//   Kd = 2·ζ·wn·I
//   wn = 2π·0.5 ≈ 3.14 rad/s (~0.5Hz)
//   ζ = 0.9
//
// 陷波滤波器 (缺陷 35: 陷波器相位匹配):
//   中心频率: 2.35Hz, 14.7Hz (结构模态, 继承自 Python flex_dynamics)
//   串联在 pitch/yaw 误差通道, 防止激励弯曲共振
//   滚转通道 (Xb) 无弯曲耦合, 不过滤
//
// 前项目 C++ 移植血泪教训 (开发日志.md):
//   "omega 反馈通道丢失, PD 退化为 P, 姿态发散 (0.6°→65.8° at t=14)"
//   → omega_actual 为必填参数 (无默认值), 强制调用方传入 state.omega_b()
//   → e_omega = omega_des - omega_actual, 严禁 omega_actual 置零
//
// 输出:
//   M_cmd → allocate_flaps_normalized → 4 片襟翼偏转
//   TVC gimbal → 推力矢量控制 (俯仰/偏航)
// =============================================================================
#ifndef STARSHIP_BELLY_FLOP_6DOF_ATTITUDE_CONTROL_HPP
#define STARSHIP_BELLY_FLOP_6DOF_ATTITUDE_CONTROL_HPP

#include <cmath>
#include "aero_6dof.hpp"
#include "dynamics_6dof.hpp"
#include "control_allocation_6dof.hpp"
#include "../core/fixed_matrix.hpp"
#include "../core/quaternion.hpp"

namespace starship {
namespace belly_flop_6dof {

// =============================================================================
// 二阶陷波器 (IIR) — 与 Python flex_dynamics.NotchFilter 一致
// =============================================================================
// H(z) = (z² - 2cos(ωn)z + 1) / (z² - 2r·cos(ωn)z + r²)
// ωn = 弯曲频率 (归一化), r = 深度因子 (0.9~0.99, 越大坑越窄)
// 直接 II 型实现, 内部 2 阶缓冲区 (零动态内存)
// =============================================================================
class NotchFilter {
public:
    NotchFilter() : b0_(1.0f), b1_(0.0f), b2_(1.0f),
                    a1_(0.0f), a2_(0.0f),
                    x1_(0.0f), x2_(0.0f), y1_(0.0f), y2_(0.0f) {}

    // 构造: 指定中心频率 (Hz), 深度 (0~1), 采样率 (Hz)
    NotchFilter(float freq_hz, float depth, float sample_rate) {
        const float PI = 3.14159265358979f;
        float wn = 2.0f * PI * freq_hz / sample_rate;
        float r = 1.0f - depth;  // depth 越大, r 越小, 坑越深
        // 分子系数: z² - 2cos(ωn)z + 1
        b0_ = 1.0f;
        b1_ = -2.0f * std::cos(wn);
        b2_ = 1.0f;
        // 分母系数: z² - 2r·cos(ωn)z + r² (a0=1 归一化)
        a1_ = -2.0f * r * std::cos(wn);
        a2_ = r * r;
        // 缓冲区清零
        x1_ = x2_ = y1_ = y2_ = 0.0f;
    }

    // 滤波单个样本, 返回滤波后的值
    float filter(float x) {
        // 直接 II 型: y = (b0·x + b1·x1 + b2·x2 - a1·y1 - a2·y2) / a0
        // a0 = 1 (归一化)
        float y = b0_ * x + b1_ * x1_ + b2_ * x2_ - a1_ * y1_ - a2_ * y2_;
        // 更新缓冲
        x2_ = x1_; x1_ = x;
        y2_ = y1_; y1_ = y;
        return y;
    }

    void reset() { x1_ = x2_ = y1_ = y2_ = 0.0f; }

private:
    float b0_, b1_, b2_;   // 分子系数
    float a1_, a2_;        // 分母系数 (a0=1)
    float x1_, x2_;        // 输入缓冲
    float y1_, y2_;        // 输出缓冲
};

// =============================================================================
// 陷波器组: 在 ω1 和 ω2 处各挖一个坑, 串联在姿态误差通道
// =============================================================================
// 与 Python NotchFilterBank 一致:
//   freq1 = 2.35Hz (一阶弯曲模态)
//   freq2 = 14.7Hz (二阶弯曲模态)
//   depth = 0.1, sample_rate = 100Hz
// 注: 频率值继承自 Python flex_dynamics (基于悬臂梁公式).
//     严格移植要求与 Python 数值一致, 不擅自改为星舰专属结构频率.
// =============================================================================
class NotchFilterBank {
public:
    NotchFilterBank() : notch1_(2.35f, 0.1f, 100.0f),
                        notch2_(14.7f, 0.1f, 100.0f) {}
    explicit NotchFilterBank(float sample_rate)
        : notch1_(2.35f, 0.1f, sample_rate),
          notch2_(14.7f, 0.1f, sample_rate) {}

    // 串联滤波: 先 notch1 再 notch2
    float filter(float x) {
        return notch2_.filter(notch1_.filter(x));
    }

    void reset() { notch1_.reset(); notch2_.reset(); }

private:
    NotchFilter notch1_;   // 一阶弯曲模态 (2.35Hz)
    NotchFilter notch2_;   // 二阶弯曲模态 (14.7Hz)
};

// =============================================================================
// 6-DOF 姿态控制器 (四元数 PD + 陷波滤波)
// =============================================================================
class AttitudeController6DOF {
public:
    // 构造参数
    float wn;             // 自然频率 (rad/s), 默认 2π·0.5 ≈ 3.14
    float zeta;           // 阻尼比, 默认 0.9
    float sample_rate;    // 采样率 (Hz), 默认 100
    bool  use_notch;      // 是否使用陷波滤波器

    // 状态记录 (用于诊断)
    float last_M_cmd[3];
    float last_delta_flaps[4];
    float last_tvc_gimbal[2];

    // 陷波器组 (pitch/yaw 各一组, 含角速度通道)
    NotchFilterBank notch_pitch;
    NotchFilterBank notch_yaw;
    NotchFilterBank notch_pitch_omega;
    NotchFilterBank notch_yaw_omega;

    AttitudeController6DOF(float wn_ = 2.0f * 3.14159265358979f * 0.5f,
                           float zeta_ = 0.9f,
                           float sample_rate_ = 100.0f,
                           bool use_notch_ = true)
        : wn(wn_), zeta(zeta_), sample_rate(sample_rate_), use_notch(use_notch_),
          notch_pitch(sample_rate_), notch_yaw(sample_rate_),
          notch_pitch_omega(sample_rate_), notch_yaw_omega(sample_rate_) {
        reset();
    }

    // 重置控制器状态
    void reset() {
        for (int i = 0; i < 3; ++i) last_M_cmd[i] = 0.0f;
        for (int i = 0; i < 4; ++i) last_delta_flaps[i] = 0.0f;
        for (int i = 0; i < 2; ++i) last_tvc_gimbal[i] = 0.0f;
        if (use_notch) {
            notch_pitch.reset();
            notch_yaw.reset();
            notch_pitch_omega.reset();
            notch_yaw_omega.reset();
        }
    }

    // =========================================================================
    // 计算期望力矩和襟翼/TVC 指令
    // =========================================================================
    // 参数:
    //   q_des:        期望四元数
    //   omega_des:    期望角速度 (body 系) [p, q, r]
    //   q_actual:     实际四元数
    //   omega_actual: 实际角速度 (body 系) [p, q, r]   ← 必填, 禁置零 (前项目教训)
    //   I:            转动惯量 (对角阵)
    //   m_fuel:       当前燃料 (kg, 用于惯量自适应 — 已通过 I 传入)
    //   Q_dyn:        动压 (Pa)
    //   omega_des_dot: 期望角加速度 (前馈), 可为 nullptr
    // 输出:
    //   M_cmd:        期望力矩 (body 系) [Mx, My, Mz]
    //   delta_flaps:  4 片襟翼偏转 [d_FL, d_FR, d_RL, d_RR] (rad)
    //   tvc_gimbal:   TVC 偏转 [gimbal_y, gimbal_z] (rad)
    // =========================================================================
    void compute_torque(const Quaternion& q_des,
                        const Vec3f& omega_des,
                        const Quaternion& q_actual,
                        const Vec3f& omega_actual,   // ← 必填, 禁置零
                        const InertiaTensor& I,
                        float Q_dyn,
                        const Vec3f* omega_des_dot,  // 可为 nullptr
                        float M_cmd[3],
                        float delta_flaps[4],
                        float tvc_gimbal[2]) {
        // ---- 四元数误差 (body 系, 修复 10) ----
        // e_q = q_actual^{-1} ⊗ q_des
        // e_vec[0]=滚转, e_vec[1]=俯仰, e_vec[2]=偏航
        Quaternion e_q = q_actual.inverse() * q_des;
        // 缺陷 18: sign(qw) 处理双覆盖 — w<0 整体取反, 保证最短路径
        if (e_q.w < 0.0f) {
            e_q.w = -e_q.w; e_q.x = -e_q.x; e_q.y = -e_q.y; e_q.z = -e_q.z;
        }
        float e_vec[3] = {e_q.x, e_q.y, e_q.z};

        // 角速度误差 (前项目教训: omega_actual 必须从 state 传入)
        float e_omega[3] = {
            omega_des[0] - omega_actual[0],
            omega_des[1] - omega_actual[1],
            omega_des[2] - omega_actual[2]
        };

        // ---- 陷波滤波 (pitch/yaw 通道, 缺陷 35) ----
        // 滚转通道 (Xb) 无弯曲耦合, 不过滤
        if (use_notch) {
            e_vec[1] = notch_pitch.filter(e_vec[1]);
            e_vec[2] = notch_yaw.filter(e_vec[2]);
            e_omega[1] = notch_pitch_omega.filter(e_omega[1]);
            e_omega[2] = notch_yaw_omega.filter(e_omega[2]);
        }

        // ---- 增益 (自适应惯量) ----
        // Kp = 2·I·wn²  (补偿 e_vec=θ/2 因子, 使闭环为标准二阶)
        // Kd = 2·ζ·wn·I
        float Ixx = I.Ixx, Iyy = I.Iyy, Izz = I.Izz;
        float Kp_x = 2.0f * Ixx * wn * wn;
        float Kp_y = 2.0f * Iyy * wn * wn;
        float Kp_z = 2.0f * Izz * wn * wn;
        float Kd_x = 2.0f * zeta * wn * Ixx;
        float Kd_y = 2.0f * zeta * wn * Iyy;
        float Kd_z = 2.0f * zeta * wn * Izz;

        // ---- 力矩指令 ----
        // M_cmd = Kp·e_vec + Kd·e_omega + I·omega_des_dot (I 对角, 逐元素)
        M_cmd[0] = Kp_x * e_vec[0] + Kd_x * e_omega[0];
        M_cmd[1] = Kp_y * e_vec[1] + Kd_y * e_omega[1];
        M_cmd[2] = Kp_z * e_vec[2] + Kd_z * e_omega[2];
        if (omega_des_dot != nullptr) {
            M_cmd[0] += Ixx * (*omega_des_dot)[0];
            M_cmd[1] += Iyy * (*omega_des_dot)[1];
            M_cmd[2] += Izz * (*omega_des_dot)[2];
        }

        // ---- 控制分配 ----
        // 襟翼分配 (归一化, 防饱和保持方向)
        allocate_flaps_normalized(M_cmd, Q_dyn, delta_flaps);

        // ---- TVC 分配 (俯仰/偏航, 滚转不可控) ----
        // TVC 力矩 = T·sin(gimbal)·L_tvc
        // 简化: TVC 辅助襟翼, 在低动压时主导
        // TVC 增益: Q<5000 时 TVC 活跃
        float tvc_gain = 1.0f - Q_dyn / 5000.0f;
        if (tvc_gain < 0.0f) tvc_gain = 0.0f;
        const float GIMBAL_LIMIT = 0.17453292519943295f;  // 10° in rad
        tvc_gimbal[0] = 0.0f;
        tvc_gimbal[1] = 0.0f;
        if (tvc_gain > 0.01f) {
            // TVC 俯仰: gimbal_y
            float gy = M_cmd[1] * tvc_gain / 1.0e6f;
            if (gy >  GIMBAL_LIMIT) gy =  GIMBAL_LIMIT;
            if (gy < -GIMBAL_LIMIT) gy = -GIMBAL_LIMIT;
            tvc_gimbal[0] = gy;
            // TVC 偏航: gimbal_z
            float gz = M_cmd[2] * tvc_gain / 1.0e6f;
            if (gz >  GIMBAL_LIMIT) gz =  GIMBAL_LIMIT;
            if (gz < -GIMBAL_LIMIT) gz = -GIMBAL_LIMIT;
            tvc_gimbal[1] = gz;
        }

        // 记录状态
        for (int i = 0; i < 3; ++i) last_M_cmd[i] = M_cmd[i];
        for (int i = 0; i < 4; ++i) last_delta_flaps[i] = delta_flaps[i];
        for (int i = 0; i < 2; ++i) last_tvc_gimbal[i] = tvc_gimbal[i];
    }
};

}  // namespace belly_flop_6dof
}  // namespace starship

#endif  // STARSHIP_BELLY_FLOP_6DOF_ATTITUDE_CONTROL_HPP
