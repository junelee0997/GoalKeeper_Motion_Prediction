import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
joint_number = [6, 5, 4, 0, 1, 2, 3, 2, 1, 0, 7, 8, 9, 10, 9, 8, 11, 12, 13, 12, 11, 8, 14, 15, 16]
ncurrent = np.array([])
current = np.array([])
def pair_line(fram):
    fram = np.array(fram)
    joint = fram.reshape(-1, 3)
    x = []
    y = []
    z = []
    for k in joint_number:
        nx, ny, nz = joint[k]
        x.append(nx)
        y.append(ny)
        z.append(nz)
    return x, y, z

def draw_animation(frame, end_scene):
    global current
    global ncurrent
    frame = np.array(frame)
    end_scene = np.array(end_scene)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    line, = ax.plot([], [], [], lw=3)
    ncurrent = end_scene.copy().astype(np.float64)
    current = end_scene.copy().astype(np.float64)
    def make_frame(i):
        global current
        global ncurrent
        if i == 0:
            current = ncurrent.copy()
        current += frame[i]
        x, y, z = pair_line(current)
        line.set_data(x, y)
        line.set_3d_properties(z)
        return line,

    ax.set_xlim3d([-4.0, 4.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-4.0, 4.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-10.0, 10.0])
    ax.set_zlabel('Z')
    anim = FuncAnimation(fig, make_frame, frames=frame.shape[0], interval=100)
    plt.show()
