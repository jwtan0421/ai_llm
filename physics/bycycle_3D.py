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
theta = np.arctan(v ** 2 / (g * R))
dt = 0.02
T = 6
frames = int(T / dt)

# ========== 图形 ==========
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D 自行车转弯动画（含受力箭头）", fontsize=14)
ax.set_xlim(-12, 12)
ax.set_ylim(-12, 12)
ax.set_zlim(0, 8)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.view_init(elev=25, azim=135)

# ========== 初始图元 ==========
bike_dot, = ax.plot([], [], [], 'ro', markersize=6)
bike_body_line, = ax.plot([], [], [], 'k-', lw=2)

arrow_G = arrow_N = arrow_F = arrow_Cf = None

# 辅助路径
theta_arr = np.linspace(0, 2 * np.pi, 200)
circle_x = R * np.cos(theta_arr)
circle_y = R * np.sin(theta_arr)
ax.plot(circle_x, circle_y, np.zeros_like(circle_x), 'gray', linestyle='--', alpha=0.3)

# ========== 添加箭头图例（伪箭头） ==========
dummy_quivers = []
legend_info = [
    ('blue', '重力 G'),
    ('green', '法向力 N'),
    ('red', '摩擦力 F'),
    ('orange', '离心力 Cf')
]
for color, label in legend_info:
    q = ax.quiver(0, 0, 0, 0, 0, 0, color=color)
    dummy_quivers.append(q)
ax.legend(dummy_quivers, [label for _, label in legend_info], loc='upper left')

# ========== 初始化函数 ==========
def init():
    bike_dot.set_data([], [])
    bike_dot.set_3d_properties([])
    bike_body_line.set_data([], [])
    bike_body_line.set_3d_properties([])
    return bike_dot, bike_body_line

# ========== 动画更新函数 ==========
def update(frame):
    global arrow_G, arrow_N, arrow_F, arrow_Cf

    t = frame * dt
    x = R * np.cos(omega * t)
    y = R * np.sin(omega * t)
    z = 0

    # 单位切向方向（速度方向）
    tx = -np.sin(omega * t)
    ty = np.cos(omega * t)

    # 倾斜方向分量
    length = 2.5
    tilt_dx = tx * np.cos(theta)
    tilt_dy = ty * np.cos(theta)
    tilt_dz = np.sin(theta)

    x1 = x - (length / 2) * tilt_dx
    y1 = y - (length / 2) * tilt_dy
    z1 = z - (length / 2) * tilt_dz
    x2 = x + (length / 2) * tilt_dx
    y2 = y + (length / 2) * tilt_dy
    z2 = z + (length / 2) * tilt_dz

    # 更新车身位置
    bike_dot.set_data([x], [y])
    bike_dot.set_3d_properties([z])
    bike_body_line.set_data([x1, x2], [y1, y2])
    bike_body_line.set_3d_properties([z1, z2])

    # 力计算
    G = m * g
    F_c = m * v ** 2 / R
    N = m * g / np.cos(theta)

    r_vec = np.array([-x, -y])
    dist = np.linalg.norm(r_vec)
    center_dir = r_vec / dist if dist > 0 else np.array([0.0, 0.0])
    Fx, Fy = center_dir

    # 清除旧箭头
    for arrow in [arrow_G, arrow_N, arrow_F, arrow_Cf]:
        if arrow: arrow.remove()

    scale = 0.02  # 🔧 缩小箭头长度

    # 四个力箭头
    arrow_G = ax.quiver(x, y, z, 0, 0, -G * scale, color='blue')
    arrow_N = ax.quiver(x, y, z, tilt_dx * N * scale, tilt_dy * N * scale, tilt_dz * N * scale, color='green')
    arrow_F = ax.quiver(x, y, z, Fx * F_c * scale, Fy * F_c * scale, 0, color='red')
    arrow_Cf = ax.quiver(x, y, z, -Fx * F_c * scale, -Fy * F_c * scale, 0, color='orange')

    return [bike_dot, bike_body_line, arrow_G, arrow_N, arrow_F, arrow_Cf]

# ========== 创建动画 ==========
ani = FuncAnimation(fig, update, frames=frames, init_func=init, interval=20, blit=False)

# ani.save("bike_3d_with_forces_legend.mp4", fps=30, dpi=150, extra_args=['-vcodec', 'libx264'])

plt.tight_layout()
plt.show()
