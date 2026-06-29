// =============================================================================
// belly_flop_aero.hpp - Belly-Flop 气动模型 (C++翻译, Step 7E)
// =============================================================================
// 理论方案 9.0-Final Step 7E
// 缺陷24: C++ sigmoid数值差异 → 用tanh()替代sigmoid
//
// 数学等价性: sigmoid(x) = 1/(1+exp(-x)) = 0.5 + 0.5*tanh(x/2)
//   - sigmoid 在 x<-30 时 exp(-x) 溢出 → inf
//   - tanh 在所有 x 值都数值稳定 → ±1
//   - 两者在 [-30, 30] 范围内最大差异 < 1e-15
//
// 对应 Python: src/belly_flop/aero_model.py
// =============================================================================
#ifndef STARSHIP_BELLY_FLOP_AERO_HPP
#define STARSHIP_BELLY_FLOP_AERO_HPP

#include <cmath>
#include <cstdint>

namespace starship {
namespace belly_flop {

// =============================================================================
// 物理常数 (星舰构型, 与 Python aero_model.py 一致)
// =============================================================================
constexpr float DIAMETER    = 9.0f;           // m, 直径
constexpr float L_REF       = 50.0f;          // m, 参考长度
constexpr float S_REF       = 3.14159265358979f * (DIAMETER / 2.0f) * (DIAMETER / 2.0f);  // ≈63.617 m²
constexpr float M_DRY       = 100000.0f;      // kg, 干重
constexpr float M_FUEL_INIT = 50000.0f;       // kg, 初始燃料
constexpr float T_MAX       = 4600e3f;        // N, 最大推力
constexpr float T_IDLE      = 460e3f;         // N, 怠速推力
constexpr float ISP         = 380.0f;         // s, 比冲
constexpr float G0_ISP      = 9.80665f;       // Isp参考重力

constexpr float C_DELTA_FWD = 0.8f;           // 前翼效率
constexpr float C_DELTA_AFT = 0.6f;           // 后翼效率
constexpr float DELTA_MAX   = 0.523598775598f; // rad, ±30°

constexpr float R_EARTH     = 6371000.0f;     // m
constexpr float G0_SL       = 9.80665f;       // m/s²

// =============================================================================
// 1976标准大气 (简化, 与 Python atmosphere.py 一致)
// =============================================================================
struct AtmosphereResult {
    float rho;      // 密度 [kg/m³]
    float a_sound;  // 声速 [m/s]
    float p;        // 压力 [Pa]
    float T;        // 温度 [K]
};

inline AtmosphereResult atmosphere(float h) {
    AtmosphereResult r;
    // 简化: 分段线性拟合1976标准大气
    if (h < 0.0f) h = 0.0f;
    if (h > 86000.0f) h = 86000.0f;

    // 对流层 (0-11km)
    if (h < 11000.0f) {
        r.T = 288.15f - 0.0065f * h;
    } else if (h < 20000.0f) {
        r.T = 216.65f;  // 同温层
    } else if (h < 32000.0f) {
        r.T = 216.65f + 0.001f * (h - 20000.0f);
    } else {
        r.T = 228.65f + 0.0028f * (h - 32000.0f);
    }

    // 压力 (简化指数衰减)
    float p0 = 101325.0f;
    float H_scale = 8500.0f;  // 标高
    r.p = p0 * std::exp(-h / H_scale);

    // 密度
    r.rho = r.p / (287.05287f * r.T);

    // 声速
    r.a_sound = std::sqrt(1.4f * 287.05287f * r.T);

    return r;
}

inline float get_mass(float m_fuel) {
    return M_DRY + m_fuel;
}

inline float get_Iyy(float m_fuel) {
    // Iyy = Iyy_dry + (m_fuel/m_fuel_init) * Iyy_fuel_init
    constexpr float IYY_DRY = (1.0f / 12.0f) * M_DRY * L_REF * L_REF;
    constexpr float L_FUEL = L_REF / 3.0f;
    constexpr float IYY_FUEL_INIT = (1.0f / 12.0f) * M_FUEL_INIT * L_FUEL * L_FUEL;
    return IYY_DRY + (m_fuel / M_FUEL_INIT) * IYY_FUEL_INIT;
}

inline float gravity(float h) {
    return G0_SL * (R_EARTH / (R_EARTH + h)) * (R_EARTH / (R_EARTH + h));
}

// =============================================================================
// 缺陷24: tanh替代sigmoid (数值稳定)
// =============================================================================
// sigmoid(x) = 1/(1+exp(-x)) = 0.5 + 0.5*tanh(x/2)
//
// 优势:
//   1. tanh 在 x→±∞ 时 → ±1, 无溢出
//   2. sigmoid 在 x<-30 时 exp(-x)→inf, 1/(1+inf)=0 (但中间步骤溢出)
//   3. C++ std::tanh 是标准库函数, 数值稳定
inline float sigmoid_tanh(float x) {
    return 0.5f + 0.5f * std::tanh(0.5f * x);
}

// Mach sigmoid 权重 (与 Python mach_sigmoid_weights 一致, 但用 tanh)
inline void mach_sigmoid_weights(float M, float& w_trans, float& w_super) {
    // w_trans = sigmoid(20*(M-1.0))  →  tanh版本
    w_trans = sigmoid_tanh(20.0f * (M - 1.0f));
    // w_super = sigmoid(20*(M-1.5))  →  tanh版本
    w_super = sigmoid_tanh(20.0f * (M - 1.5f));
}

// =============================================================================
// 气动系数 (与 Python aero_coefficients 一致)
// =============================================================================
struct AeroCoeffs {
    float CD0, CDa, CLa, Cma;
};

inline AeroCoeffs aero_coefficients(float M) {
    float w_trans, w_super;
    mach_sigmoid_weights(M, w_trans, w_super);

    AeroCoeffs c;
    // 三层加权: 亚*(1-w_trans) + 跨*w_trans*(1-w_super) + 超*w_super
    c.CD0 = 0.3f * (1.0f - w_trans) + 0.8f * w_trans * (1.0f - w_super) + 0.5f * w_super;
    c.CDa = 1.5f * (1.0f - w_trans) + 2.5f * w_trans * (1.0f - w_super) + 2.0f * w_super;
    c.CLa = 1.2f * (1.0f - w_trans) + 0.8f * w_trans * (1.0f - w_super) + 0.6f * w_super;
    c.Cma = 0.15f * (1.0f - w_trans) + 0.25f * w_trans * (1.0f - w_super) + 0.20f * w_super;
    return c;
}

// =============================================================================
// 攻角计算
// =============================================================================
struct AlphaGamma {
    float alpha;  // rad, [-π, π]
    float gamma;  // rad
};

inline AlphaGamma angle_of_attack(float theta, float vx, float vz) {
    AlphaGamma ag;
    ag.gamma = std::atan2(vx, vz);
    ag.alpha = theta - ag.gamma;
    // 归一化到 [-π, π]
    ag.alpha = std::fmod(ag.alpha + 3.14159265358979f, 2.0f * 3.14159265358979f) - 3.14159265358979f;
    return ag;
}

// =============================================================================
// 气动力/力矩计算
// =============================================================================
struct AeroResult {
    float D, L;           // 阻力, 升力 [N]
    float Fx_aero, Fz_aero;  // 气动力 [N]
    float M_aero, M_flap, M_total;  // 力矩 [N·m]
    float Q;              // 动压 [Pa]
    float alpha, gamma;   // [rad]
    float M_mach;         // 马赫数
    float rho, a_sound;
};

inline AeroResult aero_forces_and_moments(float vx, float vz, float theta, float h,
                                           float delta_fwd, float delta_aft) {
    AeroResult r{};

    AtmosphereResult atm = atmosphere(h);
    r.rho = atm.rho;
    r.a_sound = atm.a_sound;

    float V = std::sqrt(vx * vx + vz * vz);
    if (V < 1e-6f) return r;

    float M = V / atm.a_sound;
    r.M_mach = M;

    AlphaGamma ag = angle_of_attack(theta, vx, vz);
    r.alpha = ag.alpha;
    r.gamma = ag.gamma;

    AeroCoeffs ac = aero_coefficients(M);
    float CD = ac.CD0 + ac.CDa * std::sin(r.alpha) * std::sin(r.alpha);
    float CL = ac.CLa * std::sin(2.0f * r.alpha) * 0.5f;
    float Cm = -ac.Cma * std::sin(r.alpha - 1.48352986419518f);  // 85° in rad

    float Q = 0.5f * atm.rho * V * V * S_REF;
    r.Q = Q;

    r.D = Q * CD;
    r.L = Q * CL;

    // 气动力坐标变换
    r.Fx_aero = -r.D * std::sin(r.gamma) + r.L * std::cos(r.gamma);
    r.Fz_aero = -r.D * std::cos(r.gamma) - r.L * std::sin(r.gamma);

    // 力矩
    r.M_aero = Q * S_REF * L_REF * Cm;
    r.M_flap = Q * S_REF * L_REF * (C_DELTA_FWD * delta_fwd + C_DELTA_AFT * delta_aft);
    r.M_total = r.M_aero + r.M_flap;

    return r;
}

// =============================================================================
// 配平襟翼 (简化: 解析公式, 非查表)
// =============================================================================
inline void trim_flaps(float alpha, float mach, float& delta_fwd, float& delta_aft) {
    AeroCoeffs ac = aero_coefficients(mach);
    float Cm = -ac.Cma * std::sin(alpha - 1.48352986419518f);  // 85° in rad

    // 等力矩分配: δ_fwd = -Cm/(2·Cδf), δ_aft = -Cm/(2·Cδa)
    delta_fwd = -Cm / (2.0f * C_DELTA_FWD);
    delta_aft = -Cm / (2.0f * C_DELTA_AFT);

    // 钳位
    if (delta_fwd > DELTA_MAX) delta_fwd = DELTA_MAX;
    if (delta_fwd < -DELTA_MAX) delta_fwd = -DELTA_MAX;
    if (delta_aft > DELTA_MAX) delta_aft = DELTA_MAX;
    if (delta_aft < -DELTA_MAX) delta_aft = -DELTA_MAX;
}

}  // namespace belly_flop
}  // namespace starship

#endif  // STARSHIP_BELLY_FLOP_AERO_HPP
