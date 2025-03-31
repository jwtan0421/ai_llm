import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==============================
# 1. 中文和负号显示设置
# ==============================
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 假设系统已安装 SimHei 字体
matplotlib.rcParams['axes.unicode_minus'] = False

# ========== 物理参数 ==========
g = 9.8
m = 75
R = 10
v = 5
omega = v / R
theta = np.arctan(v ** 2 / (g * R))  # 倾斜角（车身相对于垂直方向）
dt = 0.02
T = 6
frames = int(T / dt)

# ========== 设置图形 ==========
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("自行车转弯 3D 可视化")
ax.set_xlim(-12, 12)
ax.set_ylim(-12, 12)
ax.set_zlim(0, 8)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=25, azim=135)  # 初始视角（可调）

# ========== 初始化对象 ==========
bike_dot, = ax.plot([], [], [], 'ro', markersize=6)
bike_body_line, = ax.plot([], [], [], 'k-', lw=2)

# 地面辅助圆（轨迹）
circle_pts = np.linspace(0, 2 * np.pi, 100)
circle_x = R * np.cos(circle_pts)
circle_y = R * np.sin(circle_pts)
circle_z = np.zeros_like(circle_x)
ax.plot(circle_x, circle_y, circle_z, 'gray', linestyle='--', alpha=0.3)

# ========== 初始化函数 ==========
def init():
    bike_dot.set_data([], [])
    bike_dot.set_3d_properties([])
    bike_body_line.set_data([], [])
    bike_body_line.set_3d_properties([])
    return bike_dot, bike_body_line

# ========== 动画更新函数 ==========
def update(frame):
    t = frame * dt
    x = R * np.cos(omega * t)
    y = R * np.sin(omega * t)
    z = 0  # 在地面上

    # 车身倾斜方向：绕 z 轴切向方向倾斜 theta
    # 求单位切向方向
    dx = -np.sin(omega * t)
    dy = np.cos(omega * t)
    dz = 0

    # 车身长度 & 倾斜计算（绕切线方向旋转）
    length = 2.5
    tilt_dx = dx * np.cos(theta)
    tilt_dy = dy * np.cos(theta)
    tilt_dz = np.sin(theta)  # 向上倾斜 z 分量

    # 中心点为车身中点，绘制两端
    x1 = x - (length / 2) * tilt_dx
    y1 = y - (length / 2) * tilt_dy
    z1 = z - (length / 2) * tilt_dz
    x2 = x + (length / 2) * tilt_dx
    y2 = y + (length / 2) * tilt_dy
    z2 = z + (length / 2) * tilt_dz

    # 更新
    bike_dot.set_data([x], [y])
    bike_dot.set_3d_properties([z])
    bike_body_line.set_data([x1, x2], [y1, y2])
    bike_body_line.set_3d_properties([z1, z2])

    return bike_dot, bike_body_line

# ========== 生成动画 ==========
ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, interval=20)

# 保存为 mp4（可选）
# ani.save("bike_3d.mp4", fps=30, dpi=150, extra_args=['-vcodec', 'libx264'])

plt.show()
