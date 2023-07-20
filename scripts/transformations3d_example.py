import numpy as np
import matplotlib.pyplot as plt
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager

# https://dfki-ric.github.io/pytransform3d/_auto_examples/plots/plot_transform_manager.html#sphx-glr-auto-examples-plots-plot-transform-manager-py
# https://dfki-ric.github.io/pytransform3d/_auto_examples/index.html
t = np.array([[0],
              [0],
              [1]])

R = np.array([[-1, 0, 0],
              [0, 1, 0],
              [0, 0, -1]])

print(R.shape)
print(t.shape)

cam0_in_world = pt.transform_from(R, t.ravel())
cam1_in_cam0 = pt.transform_from(np.eye(3), t.ravel())

tm = TransformManager()

tm.add_transform("world", "cam0", cam0_in_world)
tm.add_transform("cam0", "cam1", cam1_in_cam0)

cam1_in_world = tm.get_transform("cam1", "world")
ax = tm.plot_frames_in("world", s=0.1)


ax.set_xlim((-0.25, 0.75))
ax.set_ylim((-0.5, 0.5))
ax.set_zlim((0.0, 1.0))
plt.show()


T_cam_new_in_cam_previous = np.zeros([4, 4])
T_cam_new_in_cam_previous[:3, :3] = R
T_cam_new_in_cam_previous[:3, 3] = t.ravel()
T_cam_new_in_cam_previous[3, 3] = 1
print(T_cam_new_in_cam_previous[:3, :3])
print(T_cam_new_in_cam_previous)


# Monocular Vision and OpenCV
# https://www.youtube.com/watch?v=wwpKvGfNwIc&t
# https://www.youtube.com/watch?v=N451VeA8XRA&t

# Visual Odometry with RGBD Images in Open3D
# https://www.youtube.com/watch?v=_6JHjY6MwrU

# Visual Odometry with a Stereo Camera
# https://www.youtube.com/watch?v=WV3ZiPqd2G4
