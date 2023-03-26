import numpy as np

def umeyama(src, dst, with_scaling=False):
    """
    Aligns two sets of points using the Umeyama algorithm
    
    Parameters:
    src: numpy array (N, d)
        The first set of points, where N is the number of points and d is the dimensionality of the points
    dst: numpy array (N, d)
        The second set of points, where N is the number of points and d is the dimensionality of the points
    with_scaling: bool
        If True, the Umeyama algorithm will also estimate a scaling factor to align the points, otherwise no scaling is performed
        
    Returns:
    rotation: numpy array (d, d)
        The rotation matrix to align the points
    translation: numpy array (d,)
        The translation vector to align the points
    scaling: float
        The scaling factor to align the points (only returned if with_scaling is True)
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

    translation = dst_mean - np.dot(src_mean, rotation)

    if with_scaling:
        scaling = np.trace(np.dot(S, Vt)) / np.sum(np.var(src, axis=0))
        return rotation, translation, scaling
    else:
        return rotation, translation

# Example usage
src = np.array([[1, 2], [2, 3], [3, 4]])
dst = np.array([[2, 3], [3, 4], [4, 5]])

rotation, translation = umeyama(src, dst)
aligned_src = np.dot(src, rotation) + translation

print("Aligned source points:")
print(aligned_src)
