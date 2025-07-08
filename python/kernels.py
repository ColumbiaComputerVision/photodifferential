import sys
import numpy as np
from scipy.spatial.transform import Rotation

def gen_kernel(X: np.ndarray, 
               kernel_type: str, 
               R: np.ndarray=None) -> np.ndarray:
    """
    Generate a kernel of the given type.

    Args:
        X (np.ndarray): _description_
        kernel_type (str): _description_
        R (np.ndarray, optional): _description_. Defaults to None.

    Returns:
        np.ndarray: _description_
    """
    assert X.shape[0] == 3

    # Rotate the coordinates by R' so that the kernel is rotated by R
    if R is not None:
        X = R.T @ X
    
    if kernel_type == "half_cosine":
        n_dot_L = X[2]
        k = n_dot_L * (n_dot_L > 0)
    
    elif "tilt_" in kernel_type or kernel_type == "gradient_estimator":
        if "tilt_" in kernel_type:
            sensor_tilt_theta = np.deg2rad(float(kernel_type.split("_")[1]))
        else:
            print("Invalid kernel type: %s" % kernel_type)
            sys.exit(1)
        
        R1 = Rotation.from_rotvec(
            np.asarray([0, 1, 0]) * sensor_tilt_theta).as_matrix()
        R2 = R1.T
        R3 = Rotation.from_rotvec(
            np.asarray([1, 0, 0]) * sensor_tilt_theta).as_matrix()
        R4 = R3.T

        k = np.stack((
            gen_kernel(X, "half_cosine", R1),
            gen_kernel(X, "half_cosine", R2),
            gen_kernel(X, "half_cosine", R3),
            gen_kernel(X, "half_cosine", R4),
        ))

    else:
        print("Invalid kernel type")
        sys.exit(1)
    
    return k