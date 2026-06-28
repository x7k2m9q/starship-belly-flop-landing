"""调试: 检查tilt和e_vec分量的关系."""
import sys; sys.path.insert(0, '.')
import numpy as np
import src.quaternion_utils as qu

# 模拟tilt=9.74°的火箭姿态
# 火箭绕某个轴倾斜9.74°
tilt_rad = np.radians(9.74)
# 绕Y轴倾斜
q_y = qu.quat_multiply(qu.Q_VERT, np.array([np.cos(tilt_rad/2), 0, np.sin(tilt_rad/2), 0]))
# 绕Z轴倾斜
q_z = qu.quat_multiply(qu.Q_VERT, np.array([np.cos(tilt_rad/2), 0, 0, np.sin(tilt_rad/2)]))
# 绕X轴倾斜(滚转)
q_x = qu.quat_multiply(qu.Q_VERT, np.array([np.cos(tilt_rad/2), np.sin(tilt_rad/2), 0, 0]))

for name, q in [("绕Y", q_y), ("绕Z", q_z), ("绕X", q_x)]:
    tilt = np.degrees(qu.tilt_angle_from_vertical(q))
    e_q = qu.quat_error(qu.Q_VERT, q)
    print("%s: tilt=%.2f°  e_q=[%.4f, %.4f, %.4f, %.4f]  e_vec=[%.4f, %.4f, %.4f]" %
          (name, tilt, e_q[0], e_q[1], e_q[2], e_q[3], e_q[1], e_q[2], e_q[3]))