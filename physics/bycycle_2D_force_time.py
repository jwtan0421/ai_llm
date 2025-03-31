import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 物理参数
g = 9.8
m = 75
R = 10
v = 5
omega = v / R
dt = 0.02
theta = np.arctan(v ** 2 / (g * R))  # 倾斜角

# 时间设置
T = 6  # 秒
frames = int(T / dt)

# 存储用于力图的历史数据
time_history = []
F_history = {'G': [], 'N': [], 'F': [], 'Cf': []}

# ===============================
# 图像设置（多子图）
# ===============================
fig, (ax_main, ax_force) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("自行车转弯受力与车身倾斜动画", fontsize=16)

# 左侧主视图
ax_main.set_xlim(-12, 12)
ax_main.set_ylim(-12, 12)
ax_main.set_aspect('equal')
ax_main.set_title("受力示意 + 倾斜角")
bike_point, = ax_main.plot([], [], 'ro', ms=8)
bike_body, = ax_main.plot([], [], 'k-', lw=2)  # 黑线表示车身

# 初始箭头设置为 None，之后动态重建
arrow_gravity = arrow_normal = arrow_friction = arrow_centrifugal = None

# 加图例用空箭头
legend_labels = [('blue', '重力 G'), ('green', '法向力 N'),
                 ('red', '摩擦力 F'), ('orange', '离心力(示意)')]
for color, label in legend_labels:
    ax_main.quiver([], [], [], [], color=color, label=label)
ax_main.legend(loc='upper right')
time_text = ax_main.text(0.02, 0.95, '', transform=ax_main.transAxes)

# 右侧力随时间变化图
ax_force.set_xlim(0, T)
ax_force.set_ylim(0, m * g * 1.5)
ax_force.set_title("各力随时间变化")
ax_force.set_xlabel("时间 (s)")
ax_force.set_ylabel("力 (N)")
force_lines = {
    'G': ax_force.plot([], [], 'b-', label='重力 G')[0],
    'N': ax_force.plot([], [], 'g-', label='法向力 N')[0],
    'F': ax_force.plot([], [], 'r-', label='摩擦力 F')[0],
    'Cf': ax_force.plot([], [], 'orange', label='离心力 Cf')[0]
}
ax_force.legend()


def init():
    bike_point.set_data([], [])
    bike_body.set_data([], [])
    time_text.set_text('')
    for line in force_lines.values():
        line.set_data([], [])
    return [bike_point, bike_body, time_text] + list(force_lines.values())


def update(frame):
    global arrow_gravity, arrow_normal, arrow_friction, arrow_centrifugal

    t = frame * dt
    x = R * np.cos(omega * t)
    y = R * np.sin(omega * t)
    bike_point.set_data([x], [y])

    G = m * g
    F_c = m * v ** 2 / R
    N = m * g / np.cos(theta)

    # 向心力方向
    r_vec = np.array([-x, -y])
    dist = np.linalg.norm(r_vec)
    pointing_to_center = r_vec / dist if dist > 0 else np.array([0.0, 0.0])

    Fx_c = F_c * pointing_to_center[0]
    Fy_c = F_c * pointing_to_center[1]
    Fx_cf = -Fx_c
    Fy_cf = -Fy_c
    Nx = F_c
    Ny = m * g

    # ========== 删除旧箭头 ==========
    for arrow in [arrow_gravity, arrow_normal, arrow_friction, arrow_centrifugal]:
        if arrow: arrow.remove()

    # ========== 画新箭头 ==========
    arrow_gravity = ax_main.quiver(x, y, 0, -G * 0.05, color='blue')
    arrow_normal = ax_main.quiver(x, y, Nx * 0.05, Ny * 0.05, color='green')
    arrow_friction = ax_main.quiver(x, y, Fx_c * 0.05, Fy_c * 0.05, color='red')
    arrow_centrifugal = ax_main.quiver(x, y, Fx_cf * 0.05, Fy_cf * 0.05, color='orange')

    # ========== 倾斜角（车身线） ==========
    # 使用倾角 theta 绘制一条斜线模拟车身
    length = 2.5  # 车身长度
    dx = length * np.sin(theta)
    dy = length * np.cos(theta)
    bike_body.set_data([x - dx / 2, x + dx / 2], [y - dy / 2, y + dy / 2])

    # ========== 更新力历史 ==========
    time_history.append(t)
    F_history['G'].append(G)
    F_history['N'].append(N)
    F_history['F'].append(F_c)
    F_history['Cf'].append(F_c)  # 离心力与向心力等大

    for key in F_history:
        force_lines[key].set_data(time_history, F_history[key])

    time_text.set_text(f'time = {t:.2f}s')

    return [bike_point, bike_body, time_text] + list(force_lines.values()) + [
        arrow_gravity, arrow_normal, arrow_friction, arrow_centrifugal
    ]


# 动画
ani = animation.FuncAnimation(
    fig, update, frames=frames,
    init_func=init, blit=False, interval=20, repeat=False
)

plt.tight_layout()
plt.show()
