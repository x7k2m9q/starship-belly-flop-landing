"""
Phase 9.0 Step 9b: 100次MC鲁棒性验证
=====================================
9a结果: 20次MC×7配置全部100% Safe Landing
9b目标: 100次MC验证鲁棒性, 验收标准 Safe Landing > 50%

测试配置 (基于9a最优):
  1. 无偏置baseline (验证缺陷29+30修复的独立效果)
  2. 无偏置+Dither (9a最优BELLY err)
  3. F-1.25/A+1.25+Dither (偏置+Dither综合)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from belly_flop_9a2_9a3 import run_mc, print_result


if __name__ == '__main__':
    N_MC = 100
    print("=" * 70)
    print("Phase 9.0 Step 9b: 100次MC鲁棒性验证")
    print("=" * 70)

    configs = [
        ('baseline',         0.0, 0.0, False, False),
        ('baseline+Dither',  0.0, 0.0, True,  False),
        ('F-1.25/A+1.25+Dither', -1.25, 1.25, True, True),
    ]

    results = []
    for label, bf, ba, di, gc in configs:
        r = run_mc(N_MC, bias_fwd_deg=bf, bias_aft_deg=ba,
                   dither_enable=di, gain_comp_enable=gc)
        print_result(label, N_MC, *r)
        results.append((label, r[0]))

    print("\n" + "=" * 70)
    print("9b验收标准: Safe Landing > 50%")
    for label, rate in results:
        status = 'PASS' if rate > 0.5 else 'FAIL'
        print(f"  {label}: {rate*100:.0f}% [{status}]")
    print("=" * 70)
