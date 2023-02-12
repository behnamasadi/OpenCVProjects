#Procrustes
#Umeyama
#Multiple View Geometry in Computer Vision (Second Edition)->78
#Enhancing Projective Spatial Augmented Reality in Industry A Model Based Approach for Registration and Calibration-> 58
#http://nghiaho.com/?page_id=671


import numpy as np

def kabsch(P, Q):
    """
    Compute the optimal rigid transformation between two sets of points in 2D or 3D space using the Kabsch algorithm.
    
    Parameters:
    - P: A (N, D) array representing the first set of points, where N is the number of points and D is the number of dimensions (2 or 3).
    - Q: A (N, D) array representing the second set of points, where N is the number of points and D is the number of dimensions (2 or 3).
    
    Returns:
    - R: A (D, D) rotation matrix representing the optimal rotation.
    - t: A (D,) translation vector representing the optimal translation.
    """
    # Center the points
    mean_P = np.mean(P, axis=0)
    mean_Q = np.mean(Q, axis=0)
    P_centered = P - mean_P
    Q_centered = Q - mean_Q
    
    # Compute the covariance matrix
    cov = P_centered.T @ Q_centered
    
    # Compute the singular value decomposition of the covariance matrix
    U, S, Vt = np.linalg.svd(cov)
    
    # Compute the optimal rotation
    D = np.eye(P.shape[1])
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[-1, -1] = -1
    R = U @ D @ Vt
    
    # Compute the optimal translation
    t = mean_Q - R @ mean_P
    
    return R, t
    
    
# Example usage

import numpy as np

# Generate two sets of points
np.random.seed(0)
P = np.random.rand(10, 2)
Q = np.random.rand(10, 2)

# Apply the Kabsch algorithm
R, t = kabsch(P, Q)

# Align the points
Q_aligned = (Q - t) @ R

# Compute the root mean square distance between the aligned points and the original points
rmse = np.sqrt(np.mean((P - Q_aligned)**2))
print("RMSE:", rmse)    
