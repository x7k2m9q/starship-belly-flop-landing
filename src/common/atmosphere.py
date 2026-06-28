"""
1976 美国标准大气模型 (解析分段实现).
输入: 几何高度 h (m, NED系下 h = -z_n, 即离地高度).
输出: 密度 rho, 温度 T, 压力 p, 音速 a.
分段: 对流层(0-11km), 同温层(11-20km), ...
回收仿真主要在 0-3km, 但实现完整 0-86km 分段以备高速.
"""
import numpy as np

# 海平面常数 (1976 Std Atm)
R = 287.05287      # 气体常数 J/(kg·K)
G0 = 9.80665       # 重力 m/s^2
GAMMA = 1.4        # 空气比热比
T0 = 288.15        # 海平面温度 K
P0 = 101325.0      # 海平面压力 Pa
RHO0 = 1.225       # 海平面密度 kg/m^3

# 分层参数: (高度上界 m, 温度梯度 K/m, 该层底温度 K, 该层底压力 Pa)
# 标准大气7层 (0-84km)
_LAYERS = [
    (11000.0, -0.0065, 288.15, 101325.0),
    (20000.0, 0.0,     216.65, 22632.06),
    (32000.0, 0.001,   216.65, 5474.889),
    (47000.0, 0.0028,  228.65, 868.0187),
    (51000.0, 0.0,     270.65, 110.9063),
    (71000.0, -0.0028, 270.65, 66.93887),
    (84852.0, -0.002,  214.65, 3.956420),
]


def _atmos(h_geom):
    """几何高度 -> (T, p, rho, a). h 限制在 [0, 84852]."""
    h = max(0.0, min(h_geom, 84852.0))
    # 找到所在层
    Tb, Pb, Hb, Lb = 288.15, 101325.0, 0.0, -0.0065
    for (Htop, L, T_bot, P_bot) in _LAYERS:
        if h <= Htop:
            Hb = Htop - (Htop - h)  # 层底高度, 重新算
            break
    # 重新精确确定层底
    Hb = 0.0
    Tb = 288.15
    Pb = 101325.0
    Lb = -0.0065
    for (Htop, L, T_bot, P_bot) in _LAYERS:
        if h <= Htop:
            break
        Hb = Htop
        Tb = T_bot
        Pb = P_bot
        Lb = L
    dh = h - Hb
    if abs(Lb) > 1e-12:
        T = Tb + Lb * dh
        p = Pb * (Tb / T) ** (G0 / (R * Lb))
    else:
        T = Tb
        p = Pb * np.exp(-G0 * dh / (R * T))
    rho = p / (R * T)
    a = np.sqrt(GAMMA * R * T)
    return T, p, rho, a


# 预计算查表加速 (0-3000m 主要工作区, 1m 步长)
_H_TABLE = np.arange(0.0, 3001.0, 1.0)
_T_TABLE = np.empty_like(_H_TABLE)
_P_TABLE = np.empty_like(_H_TABLE)
_RHO_TABLE = np.empty_like(_H_TABLE)
_A_TABLE = np.empty_like(_H_TABLE)
for i, hh in enumerate(_H_TABLE):
    _t, _p, _r, _a = _atmos(hh)
    _T_TABLE[i] = _t
    _P_TABLE[i] = _p
    _RHO_TABLE[i] = _r
    _A_TABLE[i] = _a


def atmosphere(h):
    """高度 h(m) -> (rho, a, p, T). 用线性插值(0-3000m), 超出用解析."""
    if h <= 0.0:
        return RHO0, np.sqrt(GAMMA * R * T0), P0, T0
    if h < 3000.0:
        idx = int(h)
        frac = h - idx
        rho = _RHO_TABLE[idx] * (1 - frac) + _RHO_TABLE[idx + 1] * frac
        a = _A_TABLE[idx] * (1 - frac) + _A_TABLE[idx + 1] * frac
        p = _P_TABLE[idx] * (1 - frac) + _P_TABLE[idx + 1] * frac
        T = _T_TABLE[idx] * (1 - frac) + _T_TABLE[idx + 1] * frac
        return rho, a, p, T
    T, p, rho, a = _atmos(h)
    return rho, a, p, T


def density(h):
    return atmosphere(h)[0]


def sound_speed(h):
    return atmosphere(h)[1]
