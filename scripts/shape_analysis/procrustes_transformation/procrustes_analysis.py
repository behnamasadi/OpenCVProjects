import numpy as np

def procrustes(src, dst):
    """
    Aligns two sets of points using the Procrustes Analysis algorithm
    
    Parameters:
    src: numpy array (N, d)
        The first set of points, where N is the number of points and d is the dimensionality of the points
    dst: numpy array (N, d)
        The second set of points, where N is the number of points and d is the dimensionality of the points
        
    Returns:
    aligned_src: numpy array (N, d)
        The aligned first set of points
    scaling: float
        The scaling factor used to align the points
    rotation: numpy array (d, d)
        The rotation matrix used to align the points
    translation: numpy array (d,)
        The translation vector used to align the points
    """
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_centered = src - src_mean
    dst_center = dst - dst_mean

    covariance = np.dot(src_center.T, dst_center) / src.shape[0]
    U, S, Vt = np.linalg.svd(covariance)

    rotation = np.dot(U, Vt)
    if np.linalg.det(rotation) < 0:
        Vt[-1,:] *= -1
        rotation = np.dot(U, Vt)

    scaling = np.trace(np.dot(S, Vt)) / np.sum(np.var(src, axis=0))
    translation = dst_mean - scaling * np.dot(src_mean, rotation)

    aligned_src = scaling * np.dot(src, rotation) + translation

    return aligned_src, scaling, rotation, translation

# Example usage
src = np.array([[1, 2], [2, 3], [3, 4]])
dst = np.array([[2, 3], [3, 4], [4, 5]])

aligned_src, scaling, rotation, translation = procrustes(src, dst)

print("Aligned source points:")
print(aligned_src)
print("Scaling factor:")
print(scaling)
print("Rotation matrix:")
print(rotation)
print("Translation vector:")
print(translation)

