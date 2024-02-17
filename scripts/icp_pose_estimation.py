import open3d as o3d
import numpy as np
import copy

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

# Generate random point cloud
np.random.seed(0)
source_points = np.random.rand(100, 3)
source = o3d.geometry.PointCloud()
source.points = o3d.utility.Vector3dVector(source_points)

# Apply an arbitrary transformation to generate target point cloud
transformation_matrix = np.array([[0.86, -0.5, 0, 1.5],
                                  [0.5 , 0.86, 0, -0.1],
                                  [0   , 0   , 1, 2.2],
                                  [0   , 0   , 0, 1]])
target_points = np.dot(np.hstack((source_points, np.ones((100, 1)))), transformation_matrix.T)[:, :3]
target = o3d.geometry.PointCloud()
target.points = o3d.utility.Vector3dVector(target_points)


# estimation_method (open3d.pipelines.registration.TransformationEstimation, optional,
# default=TransformationEstimationPointToPoint without scaling.):
# Estimation method. One of
# (``TransformationEstimationPointToPoint``,
# ``TransformationEstimationPointToPlane``,
# ``TransformationEstimationForGeneralizedICP``,
# ``TransformationEstimationForColoredICP``)

# Apply ICP
threshold = 0.002  # Distance threshold
reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint())

print("Transformation is:")
print(reg_p2p.transformation)

draw_registration_result(source, target, reg_p2p.transformation)
