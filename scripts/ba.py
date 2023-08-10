import numpy as np

# Toy data for demonstration purposes
# Assuming only one camera with intrinsic parameters (focal length, principal point)
focal_length = 500.0
principal_point = (320.0, 240.0)
camera_params = np.array(
    [focal_length, principal_point[0], principal_point[1]])

# Assuming 3D points (X, Y, Z)
num_points = 5
points_3d = np.random.rand(num_points, 3)

# Assuming 2D image observations (u, v) for each 3D point
observations = np.random.rand(num_points, 2)


def project_points(camera_params, points_3d):
    # Implement a simple pinhole camera model to project 3D points to 2D
    # Return the projected 2D points (u, v)
    focal_length, cx, cy = camera_params
    projections = np.zeros((num_points, 2))
    for i in range(num_points):
        x, y, z = points_3d[i]
        u = focal_length * x / z + cx
        v = focal_length * y / z + cy
        projections[i] = u, v
    return projections


def residual_function(params, observations, points_3d):
    # Compute the difference between observed 2D points and projected 2D points
    # using the camera parameters and 3D points.
    camera_params = params[:3]
    projected_points = project_points(camera_params, points_3d)
    residuals = observations - projected_points
    return residuals.flatten()


def bundle_adjust(camera_params, observations, points_3d):
    # Combine the camera parameters and 3D points into a single parameter vector
    params = np.concatenate([camera_params, points_3d.flatten()])

    # Use Gauss-Newton optimization to minimize the residual function
    from scipy.optimize import least_squares
    result = least_squares(residual_function, params,
                           args=(observations, points_3d))

    # Retrieve the optimized parameters
    optimized_params = result.x
    optimized_camera_params = optimized_params[:3]
    optimized_points_3d = optimized_params[3:].reshape(-1, 3)

    return optimized_camera_params, optimized_points_3d


# Perform bundle adjustment
optimized_camera_params, optimized_points_3d = bundle_adjust(
    camera_params, observations, points_3d)
print("Optimized Camera Parameters:", optimized_camera_params)
print("Optimized 3D Points:", optimized_points_3d)
print("ground truth:\n")
print("camera params:\n", camera_params)
print("points 3d\n", points_3d)

f_optimized, cx_optimized, cy_optimized = optimized_camera_params

cx, cy = principal_point

print("ratio:\n", focal_length /
      f_optimized, cx/cx_optimized, cy/cy_optimized)


projections = project_points(camera_params, points_3d)

optimized_projections = project_points(
    optimized_camera_params, optimized_points_3d)


print(optimized_projections-projections)
