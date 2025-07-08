import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
from matplotlib.figure import Figure
import scipy.signal

def get_data_path():
    return (Path(__file__).parent.parent / "data").resolve()

def get_results_path():
    return get_data_path() / "results"

def get_figures_path():
    return get_data_path() / "figures"

def get_config_path():
    return get_data_path() / "config"

def get_processed_data_path():
    return get_data_path() / "processed"

def get_optimization_path():
    return get_data_path() / "optimization"

def get_grad_ascent_rots_paths():
    return get_data_path() / "gradient_ascent_rotations"

def whos(x: np.ndarray):
    return x.min(), x.mean(), x.max()

def u(x):
    return np.asarray([[np.cos(x), -np.sin(x), 0], 
                       [np.sin(x), np.cos(x), 0], 
                       [0, 0, 1]])


def a(x):
    return np.asarray([[np.cos(x), 0, np.sin(x)],
                       [0, 1, 0],
                       [-np.sin(x), 0, np.cos(x)]])

def cart2sph(xyz: np.ndarray):
    """
    Convert cartesian coordinates on the unit sphere to spherical coordinates

    xyz: Nx3
    """
    theta = np.arccos(xyz[:,2])
    phi = np.sign(xyz[:,1]) * np.arccos(xyz[:,0] / np.sqrt(xyz[:,0]**2 + xyz[:,1]**2))
    
    y0_idx = xyz[:,1] == 0
    xneg_idx = xyz[:,0] < 0
    phi[y0_idx & xneg_idx] = np.pi
    phi[y0_idx & ~xneg_idx] = 0

    phi = np.mod(phi, 2 * np.pi)

    return theta, phi

def sph2cart(theta: np.ndarray, phi: np.ndarray):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.stack((x, y, z), axis=1) # Nx3

def sph2rot(theta, phi, psi=0):
    """
    Create a rotation matrix that rotates the north pole to the coordinates on
    the sphere given by theta and phi.
    """
    u = lambda x : np.asarray([[np.cos(x), -np.sin(x), 0],
                               [np.sin(x), np.cos(x), 0],
                               [0, 0, 1]])
    a = lambda x : np.asarray([[np.cos(x), 0, np.sin(x)],
                               [0, 1, 0],
                               [-np.sin(x), 0, np.cos(x)]])
    R = u(phi) @ a(theta) @ u(psi)
    return R

def create_Rmats(
    X_sphere: np.ndarray,
    theta_icosphere: np.ndarray,
    phi_icosphere: np.ndarray):

    N = X_sphere.shape[0]
    Rmats = np.zeros((N, 3, 3))
    for i in range(N):
        th = theta_icosphere[i]
        phi = phi_icosphere[i]
        R = sph2rot(th, phi)
        Rmats[i,:,:] = R

    return Rmats


def downsample_box(img: np.ndarray, D: int):
    """
    img: HxWxC or HxW
    """
    if D == 1:
        return img

    orig_ndim = img.ndim
    assert orig_ndim == 2 or orig_ndim == 3

    orig_dtype = img.dtype
    img = img.astype(np.float64) # Perform downsampling with 64-bit float

    if orig_ndim == 2:
        img = img[:,:,None]

    box = np.ones((D, D)) / D**2

    img_b = None
    for c in range(img.shape[2]):
        img_bc = scipy.signal.convolve2d(img[:,:,c], box, mode="valid")
        if img_b is None:
            img_b = np.zeros((*img_bc.shape, img.shape[2]))
        img_b[:,:,c] = img_bc
    
    img_d = img_b[::D,::D,:]

    if orig_ndim == 2:
        img_d = img_d.squeeze()
    
    if orig_dtype == np.float32:
        img_d = img_d.astype(np.float32)
    
    return img_d


def tonemap_img(img_hdr: np.ndarray, enhance_local_contrast: bool=False):
    # Input: BGR image
    tonemapper = cv2.createTonemapReinhard(2.0, 0.9, 0.5, 0.5)
    img_hdr_tmp = np.clip(img_hdr / img_hdr.max(), 0, 1)
    img_tm = tonemapper.process(img_hdr_tmp)
    img_tm[np.isnan(img_tm)] = 0

    if enhance_local_contrast:
        # Incrase local contrast
        img_tm = (img_tm * 255).astype(np.uint8)
        img_lab = cv2.cvtColor(img_tm, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8,8))
        img_lab[..., 0] = clahe.apply(img_lab[..., 0])

        img_enhanced = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

        return img_enhanced.astype(np.float64) / 255
    
    else:
        return img_tm