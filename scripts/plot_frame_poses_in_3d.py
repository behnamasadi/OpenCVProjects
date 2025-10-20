import numpy as np
import matplotlib.pyplot as plt


def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.

    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
      :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - \
        np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - \
        np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return np.array([qw, qx, qy, qz])

def quaternion_to_rot_matrix(q):
    """
    Convert a quaternion into a rotation matrix.
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def plot_frame(ax, x, y, z, q, length=1.0, frameName=""):
    """
    Plot a 3D frame with axes colored by red, green, and blue based on coordinates and quaternion.
    """
    R = quaternion_to_rot_matrix(q)
    origin = np.array([[x, y, z]]).T

    axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * length
    transformed_axes = R @ axes + origin

    ax.quiver(origin[0, 0], origin[1, 0], origin[2, 0],
              transformed_axes[0, 0]-x, transformed_axes[1,
                                                         0]-y, transformed_axes[2, 0]-z,
              color='r', label='')
    ax.quiver(origin[0, 0], origin[1, 0], origin[2, 0],
              transformed_axes[0, 1]-x, transformed_axes[1,
                                                         1]-y, transformed_axes[2, 1]-z,
              color='g', label='')
    ax.quiver(origin[0, 0], origin[1, 0], origin[2, 0],
              transformed_axes[0, 2]-x, transformed_axes[1,
                                                         2]-y, transformed_axes[2, 2]-z,
              color='b', label='')

    # Set aspect ratio
    max_range = np.array([transformed_axes[:, 0].max()-transformed_axes[:, 0].min(),
                          transformed_axes[:, 1].max(
    )-transformed_axes[:, 1].min(),
        transformed_axes[:, 2].max()-transformed_axes[:, 2].min()]).max() / 2.0

    mid_x = (transformed_axes[:, 0].max() + transformed_axes[:, 0].min()) * 0.5
    mid_y = (transformed_axes[:, 1].max() + transformed_axes[:, 1].min()) * 0.5
    mid_z = (transformed_axes[:, 2].max() + transformed_axes[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.text(x, y, z, frameName, fontsize=12,
            color='black', ha='center', va='bottom')


if __name__ == "__main__":

    # Example usage:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # world
    length = 1.0
    x, y, z = 0, 0, 0
    q = [1, 0, 0, 0]

    plot_frame(ax, x, y, z, q, length, "world")

    # First frame
    x1, y1, z1 = 1, -1, 2
    q1 = [1, 0, 0, 0]  # Identity quaternion, no rotation

    plot_frame(ax, x1, y1, z1, q1, length, "Frame 1")

    # Optionally, add a second frame or other plot elements
    x2, y2, z2 = 3, 1, -2
    q2 = [0, 1, 0, 0]

    plot_frame(ax, x2, y2, z2, q2, length, "Frame 2")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Frame Plot')
    ax.legend()

    plt.show()
