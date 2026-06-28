"""
Phase 9.0: 星舰工程级死区补偿控制器
=====================================
理论方案 6.0 (Phase 9.0) — 跨越"暗礁27/28/29"

暗礁回顾:
  暗礁27 (Phase 8.0): alpha=85°时Cm=0, trim flap=0°,
    死区(±0.5°)零化所有PD阻尼指令, theta在10秒内从85°降到43°.
  暗礁28 (Phase 9.0初): Bouc-Wen(A=1.0, gamma=-0.5)完全阻塞恒定信号
    → 偏置绕过执行器(bias_after_actuator) → 但PD仍被阻塞
  暗礁29 (Phase 9.0终): Bouc-Wen模型参数bug
    beta+gamma=0 → z无界 → z追踪x → output≈0 (所有低频信号被阻塞)
    修正: gamma=+0.5 (beta+gamma=1.0>0, z有界), alpha=0.3 (位置保持)

三层递进补偿方案 (暗礁29修复后, 偏置可穿过执行器):
  9a-1: 自适应偏置Trim — 前后翼反向偏置(+1.25°/-1.25°), 打破死区
         偏置穿过执行器(alpha=0.3允许位置保持), 不再需要绕过
  9a-2: 高频Dither信号 — 8Hz/1.0°正弦波, 线性化滞环
  9a-3: 幅值补偿 — ×1.1增益, 补偿滞环幅值损失

关键物理分析 (偏置方向):
  单独偏置前翼或后翼都会产生正反馈(δ→Cm→θ→δ同向放大).
  正确策略: 前后翼反向偏置, 力矩相互抵消, 只增加阻力.
  δ_fwd_bias = +1.25°, δ_aft_bias = -1.25°

阶段切换平滑过渡:
  BELLY→FLIP: 偏置在2秒内线性减小至0 (避免力矩跳变)
  FLIP/LANDING: 偏置=0 (不影响翻转和着陆)
"""
import numpy as np
from .integrated_controller import IntegratedBellyFlopController
from .flip_controller import THETA_BELLY, THETA_LAND


class Phase9BellyFlopController(IntegratedBellyFlopController):
    """
    Phase 9.0 增强控制器: 在7E基础上叠加死区补偿.

    补偿层次 (可独立开关):
      1. bias_enable:     前后翼反向偏置 (+1.25°/-1.25°), 穿过执行器
      2. dither_enable:   高频Dither (8Hz, 1.0°), 线性化滞环
      3. gain_comp_enable: 幅值补偿 (×1.1)

    接口与7E完全一致: update(state, dt) -> (T, theta_cmd, d_fwd, d_aft, phase, kill, info)

    暗礁29修复后: 偏置穿过执行器(不再绕过), bias_fwd_out=0
    """

    def __init__(self, bias_enable=True, dither_enable=False,
                 gain_comp_enable=False,
                 delta_fwd_bias_deg=1.25, delta_aft_bias_deg=-1.25,
                 dither_freq=8.0, dither_amp_deg=1.0,
                 gain_comp=1.1):
        super().__init__()

        # ============ 补偿开关 ============
        self.bias_enable = bias_enable
        self.dither_enable = dither_enable
        self.gain_comp_enable = gain_comp_enable

        # ============ 偏置参数 ============
        self.delta_fwd_bias = np.deg2rad(delta_fwd_bias_deg)  # 前翼偏置 [rad]
        self.delta_aft_bias = np.deg2rad(delta_aft_bias_deg)  # 后翼偏置 [rad] (反向)

        # 偏置平滑过渡 (BELLY→FLIP时2秒内减至0)
        self.bias_ramp_t = 0.0
        self.bias_ramp_duration = 2.0  # 秒
        self._bias_active = 1.0  # 0~1, 当前偏置激活比例

        # ============ Dither参数 ============
        self.dither_freq = dither_freq  # Hz
        self.dither_amp = np.deg2rad(dither_amp_deg)  # [rad]
        self._dither_phase = 0.0

        # ============ 幅值补偿 ============
        self.gain_comp = gain_comp  # 简单增益补偿因子

    def update(self, state, dt):
        """
        Phase 9.0 控制器更新.

        返回: (T, theta_cmd, d_extra_fwd, d_extra_aft, phase, kill, info)

        暗礁29修复后架构:
          d_fwd/d_aft = PD + bias + dither (全部穿过执行器)
          执行器(alpha=0.3)允许位置保持 → 偏置有效
          死区由偏置打破(bias > dead_zone) → PD有效
        """
        # 调用父类7E控制器获取基础控制量
        T, theta_cmd, d_fwd_pd, d_aft_pd, phase, kill, info = super().update(state, dt)

        if kill:
            return T, theta_cmd, d_fwd_pd, d_aft_pd, phase, kill, info

        # ============ 偏置平滑过渡 ============
        if phase == 'BELLY':
            self._bias_active = 1.0
        elif phase == 'FLIP':
            self.bias_ramp_t += dt
            self._bias_active = max(0.0, 1.0 - self.bias_ramp_t / self.bias_ramp_duration)
        else:
            self._bias_active = 0.0

        # ============ 1. 偏置 (穿过执行器, 暗礁29修复后有效) ============
        bias_fwd = 0.0
        bias_aft = 0.0
        if self.bias_enable:
            bias_fwd = self.delta_fwd_bias * self._bias_active
            bias_aft = self.delta_aft_bias * self._bias_active

        # ============ 2. PD + 偏置 ============
        d_fwd_final = d_fwd_pd + bias_fwd
        d_aft_final = d_aft_pd + bias_aft

        # ============ 3. Dither (仅BELLY, 前后翼同向不产生力矩) ============
        if self.dither_enable and phase == 'BELLY':
            self._dither_phase += 2.0 * np.pi * self.dither_freq * dt
            dither_signal = self.dither_amp * np.sin(self._dither_phase)
            d_fwd_final += dither_signal
            d_aft_final += dither_signal

        # ============ 4. 幅值补偿 ============
        if self.gain_comp_enable:
            d_fwd_final *= self.gain_comp
            d_aft_final *= self.gain_comp

        # 记录补偿信息
        info['bias_active'] = self._bias_active
        info['d_fwd_pd'] = d_fwd_pd
        info['d_aft_pd'] = d_aft_pd
        info['d_fwd_final'] = d_fwd_final
        info['d_aft_final'] = d_aft_final
        info['bias_fwd_out'] = 0.0  # 暗礁29修复: 偏置已穿过执行器, 不再绕过
        info['bias_aft_out'] = 0.0

        return T, theta_cmd, d_fwd_final, d_aft_final, phase, kill, info

    def reset(self):
        """重置控制器状态."""
        super().__init__()
        self.bias_ramp_t = 0.0
        self._bias_active = 1.0
        self._dither_phase = 0.0
