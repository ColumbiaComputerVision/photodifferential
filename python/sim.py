import sys
import os
import numpy as np
import scipy
import scipy.interpolate
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from scipy.spatial.transform import Rotation
from pathlib import Path
from stl import mesh
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pickle
import multiprocessing as mp
from typing import Dict, List, Tuple
from collections import defaultdict
import itertools
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

import kernels
import utils

NUM_PROCESSES = os.cpu_count() // 2

class GradientAscentRotations:
    def __init__(self, rot_step_size: float, icosphere_path: Path):
        save_path = utils.get_grad_ascent_rots_paths() / \
            str(rot_step_size) / ("%s.npz" % icosphere_path.stem)

        if save_path.exists():
            D = np.load(save_path)
            self.R_lr = D["R_lr"]
            self.R_ud = D["R_ud"]
        else:
            # load the icosphere points
            icosphere = mesh.Mesh.from_file(str(icosphere_path))
            X_sphere = icosphere.get_unit_normals() # Nx3
            theta_icosphere, phi_icosphere = utils.cart2sph(X_sphere)

            # Compute the Rmats that rotate the north pole to each point
            Rmats = utils.create_Rmats(X_sphere, theta_icosphere, phi_icosphere)

            # Create the rotation matrices
            self._precompute_gradient_ascent_rotations(
                X_sphere, Rmats, rot_step_size)
            save_path.parent.mkdir(exist_ok=True, parents=True)
            np.savez(save_path, R_lr=self.R_lr, R_ud=self.R_ud)

    def _precompute_gradient_ascent_rotations(
        self, X_sphere: np.ndarray, Rmats: np.ndarray, rot_step_size: float):

        N = X_sphere.shape[0]
        self.R_lr = np.zeros((N, 3, 3))
        self.R_ud = np.zeros((N, 3, 3))

        W = np.asarray([[1, 0, 0], [0, 1, 0]]).T

        for i in range(N):
            R = Rmats[i]
            self.R_lr[i] = Rotation.from_rotvec(
                (R @ W[:,1]) * rot_step_size
            ).as_matrix()
            self.R_ud[i] = Rotation.from_rotvec(
                (R @ W[:,0]) * rot_step_size
            ).as_matrix()


class Scene:
    PROCESSED_SAVE_KEYS = [
        "X_sphere",
        "L_sphere",
        "L_sphere_color_tm",
        "domega",
        "theta_icosphere",
        "phi_icosphere",
        "Rmats",
        "icosphere_vertices",
        "irradiance_maps",
        "local_max",
        "face_neighbors",
        "lower_hemisphere_clipped",
        "icosphere_path",
        "img_path"
    ]

    def __init__(self, img_path: Path=None, icosphere_path: Path=None,
                 color_channel: int = None) -> None:
        self.irradiance_maps = {} # Pre-computed irradiance maps
        self.lower_hemisphere_clipped = False
        self.optimization_runs = defaultdict(list)

        if img_path is None and icosphere_path is None:
            # No arguments given
            return

        if icosphere_path is not None:
            # Load the icosphere
            self.icosphere_path = icosphere_path
            icosphere = mesh.Mesh.from_file(str(icosphere_path))
            self.icosphere_vertices = icosphere.vectors # Nx3x3
            self.X_sphere = icosphere.get_unit_normals() # Nx3
            self.theta_icosphere, self.phi_icosphere = utils.cart2sph(self.X_sphere)
            # Approximate the solid angle as the triangle area
            self.domega = self._compute_face_area(icosphere.vectors)

            # Compute the Rmats that rotate the north pole to each point
            self.Rmats = utils.create_Rmats(
                self.X_sphere, self.theta_icosphere, self.phi_icosphere)

        # Load the image
        if img_path is not None:
            self.img_path = img_path
            img = self._load_img(img_path)
            theta_grid = np.linspace(0, np.pi * (1 - 1/img.shape[0]), img.shape[0])
            phi_grid = np.linspace(0, 2*np.pi * (1 - 1/img.shape[1]), img.shape[1])

            # Resample the image on the icosphere
            self.color_channel = color_channel
            if color_channel is None:
                img_bw = img.sum(2) # Convert to black and white
            else:
                img_bw = img[:,:,color_channel]
            self.L_sphere = self._resample_img_on_sphere(
                theta_grid, phi_grid, img_bw)
            self.L_sphere_color_tm = self._resample_img_on_sphere(
                theta_grid, phi_grid, utils.tonemap_img(img))

        self.img_D = {} # Downsampled images


    def _load_img(self, img_path):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        img = img[:,:,::-1] # convert BGR to RGB
        return img

    def cut_lower_hemisphere(self):
        # Cut the lower hemisphere
        self.L_sphere[self.theta_icosphere > np.pi/2] = 0
        self.L_sphere_color_tm[self.theta_icosphere > np.pi/2] = 0
        self.lower_hemisphere_clipped = True


    def _resample_img_on_sphere(self, theta_grid, phi_grid, img):
        F = scipy.interpolate.RegularGridInterpolator(
            (theta_grid, phi_grid), img, method="linear", fill_value=np.nan,
            bounds_error=False)
        L_sphere = F((self.theta_icosphere, self.phi_icosphere))
        L_sphere[np.isnan(L_sphere)] = 0
        return L_sphere


    def _compute_face_area(self, face_vertices):
        num_faces = face_vertices.shape[0]
        face_area = np.zeros(num_faces)
        for i in range(num_faces):
            v = face_vertices[i]
            s1 = np.linalg.norm(v[1] - v[0])
            s2 = np.linalg.norm(v[2] - v[0])
            s3 = np.linalg.norm(v[2] - v[1])
            s = 0.5 * (s1 + s2 + s3)
            face_area[i] = np.sqrt(s * (s - s1) * (s - s2) * (s - s3))

        return face_area

    def find_neighbors(self, xi, num_levels=0):
        neighbors = self.face_neighbors[xi]
        last_neighbors = neighbors
        all_neighbors = [neighbors]

        for i in range(num_levels):
            new_neighbors = np.zeros(len(last_neighbors) * 3, dtype=int)
            for i in range(len(last_neighbors)):
                new_neighbors[i*3:(i+1)*3] = self.face_neighbors[last_neighbors[i]]
            all_neighbors.append(new_neighbors)
            last_neighbors = new_neighbors

        all_neighbors = np.concatenate(all_neighbors)
        return all_neighbors

    def find_nearest_pt(self, x, last_xi):
        num_levels = 2 # adjust num levels based on the rotation step size
        all_neighbors = self.find_neighbors(last_xi, num_levels)
        Xneighbors = self.X_sphere[all_neighbors]
        xi_nearest = all_neighbors[np.argmax(Xneighbors @ x)]

        #assert xi_nearest == np.argmax(self.X_sphere @ x)

        return xi_nearest

    def run_gradient_ascent(
        self,
        p: np.ndarray,
        iter_max: int,
        x_init: np.ndarray,
        noise_level: float,
        grad_rots: GradientAscentRotations,
        stop_threshold=None) -> Tuple[np.ndarray, np.ndarray]:

        p_max = p.max(axis=0) # Maximum measurement for all 4 sensors

        x = x_init / np.linalg.norm(x_init)
        x_hist = np.full((iter_max, 3), fill_value=np.nan)

        # index of nearest point on the sphere
        xi_rounded = np.argmax(self.X_sphere @ x)

        for i in range(iter_max):
            # only search neighboring faces on the icosphere
            xi_rounded = self.find_nearest_pt(x, xi_rounded)
            x_hist[i] = x

            # Compute photodifferential
            p_curr = p[xi_rounded,:]
            p_curr = add_noise(p_curr, noise_level, p_max)

            photodiff = np.asarray(
                [p_curr[0] - p_curr[1], p_curr[2] - p_curr[3]])
            
            if stop_threshold is not None and \
                (np.abs(photodiff[0]) / (p_curr[:2].mean() + 1e-9) < stop_threshold) and \
                (np.abs(photodiff[1]) / (p_curr[2:4].mean() + 1e-9) < stop_threshold):
                    break

            R_lr = grad_rots.R_lr[xi_rounded]
            if np.sign(photodiff[0]) < 0:
                R_lr = R_lr.T
            R_ud = grad_rots.R_ud[xi_rounded]
            if np.sign(photodiff[1]) < 0:
                R_ud = R_ud.T

            x = R_ud @ R_lr @ x

        return x, x_hist


    def gradient_ascent_multi_init(self,
                                   p: np.ndarray,
                                   iter_max: int,
                                   x_init: np.ndarray,
                                   noise_level: float,
                                   grad_rots: GradientAscentRotations):
        num_trials = x_init.shape[0]
        x_final = np.zeros(x_init.shape)
        x_hist_all = np.zeros((num_trials, iter_max, 3))

        for i in range(num_trials):
            x_curr, x_hist = self.run_gradient_ascent(
                p, iter_max, x_init[i], noise_level, grad_rots)
            x_final[i] = x_curr
            x_hist_all[i] = x_hist

        return x_final, x_hist_all


    def convolve_sphere(self, kernel_type: str):
        N = self.X_sphere.shape[0]
        p = None # Allocated on the fly

        L_sphere_domega = self.L_sphere * self.domega

        for i in range(N):
            R = self.Rmats[i]
            k = kernels.gen_kernel(self.X_sphere.T, kernel_type, R=R)

            if p is None:
                n_kernels = k.shape[0] if k.ndim > 1 else 1
                p = np.zeros((N, n_kernels))

            p[i] = k @ L_sphere_domega

        return p

    def convolve_latlon(self, kernel_type: str, convolve_latlon_D=None):
        N = self.X_sphere.shape[0]
        p = None # Allocated on the fly

        if convolve_latlon_D not in self.img_D.keys():
            img = self._load_img(self.img_path)
            if self.color_channel is None:
                img = img.sum(2) # Convert to black and white
            else:
                img = img[:,:,self.color_channel]

            if convolve_latlon_D is not None:
                img = utils.downsample_box(img, convolve_latlon_D)

            self.img_D[convolve_latlon_D] = img
        else:
            img = self.img_D[convolve_latlon_D].copy()

        # Coordinates of the lat-lon image
        th, phi = np.meshgrid(
            np.linspace(0, np.pi, img.shape[0], endpoint=False),
            np.linspace(0, 2*np.pi, img.shape[1], endpoint=False),
            indexing="ij"
        )
        th = th.ravel()
        phi = phi.ravel()

        domega = np.sin(th) * (np.pi / img.shape[0]) * (2 * np.pi / img.shape[1])

        img_domega = img.ravel() * domega

        X_latlon = utils.sph2cart(th, phi)

        for i in range(N):
            R = self.Rmats[i]
            k = kernels.gen_kernel(X_latlon.T, kernel_type, R=R)

            if p is None:
                n_kernels = k.shape[0] if k.ndim > 1 else 1
                p = np.zeros((N, n_kernels))

            p[i] = k @ img_domega

        return p
    


    def sample_init_points(self, num_pts):
        """
        Sample initial points in the upper hemisphere.
        """
        available_idx = np.where(self.X_sphere[:,2] >= 0)[0]

        probs = self.domega[available_idx] / self.domega[available_idx].sum()
        idx = np.random.choice(available_idx, size=num_pts, replace=False,
                               p=probs)
        return self.X_sphere[idx]


    def compute_xi_optimal(self):
        assert self.irradiance_map_exists("half_cosine")
        total_irr = self.irradiance_maps["half_cosine"]
        optimal_i = np.argmax(total_irr)
        return optimal_i


    def compute_x_optimal(self):
        optimal_i = self.compute_xi_optimal()
        x_optimal = self.X_sphere[optimal_i]
        return x_optimal


    def optimization_run_exists(self,
                                kernel_type: str,
                                noise_level: float=0):
        runs = self.optimization_runs[kernel_type]
        for r in runs:
            if r["noise_level"] == noise_level:
                return True

        return False

    def clear_optimization_runs(self):
        self.optimization_runs = defaultdict(list)

    def log_optimization_run(self,
                             x_init: np.ndarray,
                             x_final: np.ndarray,
                             kernel_type: str,
                             noise_level: float=0):
        assert not self.optimization_run_exists(kernel_type, noise_level)
        run = {
            "x_init": x_init,
            "x_final": x_final,
            "noise_level": noise_level,
        }
        self.optimization_runs[kernel_type].append(run)

    def get_noise_levels_for_run(self, kernel_type: str):
        noise_levels = []
        for r in self.optimization_runs[kernel_type]:
            noise_levels.append(r["noise_level"])
        noise_levels = np.asarray(noise_levels)
        noise_levels.sort()
        return noise_levels

    def get_optimization_run(self,
                             kernel_type: str,
                             noise_level: float=0):
        assert self.optimization_run_exists(kernel_type, noise_level)

        runs = self.optimization_runs[kernel_type]
        for r in runs:
            if r["noise_level"] == noise_level:
                return r

    def irradiance_map_exists(self, kernel_type: str):
        return kernel_type in self.irradiance_maps.keys()

    def store_irradiance_map(self, map: np.ndarray, kernel_type: str):
        self.irradiance_maps[kernel_type] = map

    def get_irradiance_map(self, kernel_type: str):
        assert self.irradiance_map_exists(kernel_type)
        return self.irradiance_maps[kernel_type]


    def _load_face_neighbors(self):
        if "face_neighbors" in self.__dict__.keys():
            return

        # Find the local max in the total irradiance function
        mesh = trimesh.load(str(self.icosphere_path))

        #assert np.all(np.abs(mesh.face_normals - self.X_sphere) < 1e-6)

        # Convert face_adjacency matrix to Nx3 matrix of neighboring faces
        N_faces = mesh.faces.shape[0]
        self.face_neighbors = np.zeros((N_faces, 3), dtype=np.int64)
        for face_id in range(N_faces):
            idx_left = np.where(mesh.face_adjacency[:,0] == face_id)[0]
            idx_right = np.where(mesh.face_adjacency[:,1] == face_id)[0]
            self.face_neighbors[face_id,:len(idx_left)] = mesh.face_adjacency[idx_left,1]
            self.face_neighbors[face_id,len(idx_left):] = mesh.face_adjacency[idx_right,0]

    def _compute_critical_pts(self, local_max: bool=True):
        if "face_neighbors" not in self.__dict__.keys():
            self._load_face_neighbors()

        p = self.get_irradiance_map("half_cosine")
        if local_max:
            global_min = p.min()
        else:
            global_max = p.max()

        N_faces = self.face_neighbors.shape[0]

        critical_pts = np.zeros((N_faces,), dtype=np.bool_)
        for face_id in range(N_faces):
            neighbor_face_ids = self.find_neighbors(face_id, 1)

            neighbor_irradiance = p[neighbor_face_ids]
            current_irradiance = p[face_id]
            if local_max:
                critical_pts[face_id] = np.all(
                    current_irradiance >= neighbor_irradiance) and \
                    (current_irradiance > global_min)
            else:
                critical_pts[face_id] = np.all(
                    current_irradiance <= neighbor_irradiance) and \
                    (current_irradiance < global_max)

        return critical_pts

    def compute_local_max(self):
        self.local_max = self._compute_critical_pts(local_max=True)

    def compute_local_min(self):
        self.local_min = self._compute_critical_pts(local_max=False)

    def energy_harvested(self,
                         kernel_type: str,
                         noise_level: float,
                         normalize_by_E_optimal: bool):
        E_final = self.energy_harvested_unnormalized(kernel_type, noise_level)

        if normalize_by_E_optimal:
            E_optimal = self.get_irradiance_map("half_cosine").max()
            return E_final / E_optimal
        else:
            return E_final

    def energy_harvested_unnormalized(self, kernel_type: str, noise_level: float):
        x_final = self.get_optimization_run(
            kernel_type, noise_level)["x_final"]
        xi_final = np.argmax(x_final @ self.X_sphere.T, axis=1).squeeze()

        E_final = self.get_irradiance_map("half_cosine")[xi_final].squeeze()
        return E_final


    def save_colored_mesh(
        self, face_colors: np.ndarray, icosphere_path: Path, save_path: Path):
        assert face_colors.dtype == np.float32 or face_colors.dtype == np.float64
        assert face_colors.max() <= 1

        if face_colors.ndim == 1:
            face_colors = np.tile(face_colors[:,None], (1, 3))
        face_colors = (face_colors * 255).astype(np.uint8)

        sphere_mesh = trimesh.load(str(icosphere_path))

        # Convert to PLY data
        vertex = [
            (v[0], v[1], v[2]) for v in sphere_mesh.vertices
        ]
        ply_vertices = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        ply_faces = [
            (list(sphere_mesh.faces[i]), *face_colors[i]) for i in range(face_colors.shape[0])]
        ply_faces = np.array(
            ply_faces, dtype=[('vertex_indices', 'i4', (3,)), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

        el_v = PlyElement.describe(ply_vertices, "vertex")
        el_f = PlyElement.describe(ply_faces, "face")
        PlyData([el_v, el_f]).write(str(save_path.resolve()))


    def plot_sphere(self, face_colors=None):
        if face_colors is None:
            face_colors = self.L_sphere_color_tm
        if face_colors.ndim == 1 or face_colors.shape[1] == 1:
            face_colors = np.tile(face_colors.ravel()[:,None], (1, 3))

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        polys = mplot3d.art3d.Poly3DCollection(
            self.icosphere_vertices, facecolors=face_colors)
        polys.set_edgecolors(face_colors)
        ax.add_collection3d(polys)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_zticks([-1, 0, 1])
        ax.set_aspect("equal", "box")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        fig.tight_layout()
        plt.show()

    def plot_local_maxima(self):
        E = self.get_irradiance_map("half_cosine")
        local_max_idx = np.where(self.local_max)[0]
        global_max_idx = np.argmax(E)

        face_colors = E / E.max()
        if face_colors.ndim == 1 or face_colors.shape[1] == 1:
            face_colors = np.tile(face_colors.ravel()[:,None], (1, 3))

        face_colors[local_max_idx,:] = [0,1,0] # Highlight local max in green
        face_colors[global_max_idx,:] = [1,0,0] # Highlight global max in red

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        polys = mplot3d.art3d.Poly3DCollection(
            self.icosphere_vertices, facecolors=face_colors)
        polys.set_edgecolors(face_colors)
        ax.add_collection3d(polys)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_zticks([-1, 0, 1])
        ax.set_aspect("equal", "box")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        fig.tight_layout()
        plt.show()


    def plot_trajectory(self, x_hist):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        polys = mplot3d.art3d.Poly3DCollection(
            self.icosphere_vertices, facecolors=self.L_sphere_color_tm, alpha=0.8)

        ax.add_collection3d(polys)
        ax.plot(*x_hist.T, 'r.-', ms=4)

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_zticks([-1, 0, 1])
        ax.set_aspect("equal", "box")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()

    def plot_optim_run(self,
                       kernel_type: str,
                       noise_level: float=0):
        ## TODO Add a lat-lon plot?
        run = self.get_optimization_run(
            kernel_type, noise_level)
        x_final = run["x_final"]
        x_opt = self.compute_x_optimal()
        x_local_max = self.X_sphere[self.local_max]

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.plot(*x_final.T, 'r.', alpha=0.8)
        ax.plot(*x_opt, 'g+', alpha=0.8)
        ax.plot(*x_local_max.T, 'bx', alpha=0.8)

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_zticks([-1, 0, 1])
        ax.set_aspect("equal", "box")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend(["Final", "Optimal", "Local Max"])

        fig.tight_layout()
        plt.show()


    def annotated_latlon_img(self, D=8):
        img = self._load_img(self.img_path)
        img = utils.downsample_box(img, D)
        img_tm = utils.tonemap_img(img)
        img_tm[np.isnan(img_tm)] = 0
        img_tm = (img_tm * 255).astype(np.uint8)

        E = self.get_irradiance_map("half_cosine").squeeze()
        global_max_idx = np.argmax(E)
        local_max_idx = np.where(self.local_max & (E != E.max()))

        th_local_max = self.theta_icosphere[local_max_idx]
        phi_local_max = self.phi_icosphere[local_max_idx]

        th_globa_max = self.theta_icosphere[global_max_idx]
        phi_global_max = self.phi_icosphere[global_max_idx]

        radius = 32 // D

        for phi, th in zip(phi_local_max, th_local_max):
            cv2.circle(
                img_tm,
                (int(np.round(phi * img.shape[1] / (2 * np.pi))),
                    int(np.round(th * img.shape[0] / np.pi))),
                radius,
                (0, 255, 0),
                -1
            )

        cv2.circle(
            img_tm,
            (int(np.round(phi_global_max * img.shape[1] / (2 * np.pi))),
                int(np.round(th_globa_max * img.shape[0] / np.pi))),
            radius,
            (255, 0, 0),
            -1
        )

        return img_tm


    def show_latlon_img(self, plot_local_max=False, ms=8, figsize=(10,5), D=8):
        img = self._load_img(self.img_path)
        img = utils.downsample_box(img, D)
        img_tm = utils.tonemap_img(img)

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img_tm)
        ax.axis("off")

        if plot_local_max:
            E = self.get_irradiance_map("half_cosine").squeeze()
            global_max_idx = np.argmax(E)
            local_max_idx = np.where(self.local_max & (E != E.max()))

            th_local_max = self.theta_icosphere[local_max_idx]
            phi_local_max = self.phi_icosphere[local_max_idx]

            th_globa_max = self.theta_icosphere[global_max_idx]
            phi_global_max = self.phi_icosphere[global_max_idx]

            ax.plot(phi_local_max * img.shape[1] / (2 * np.pi), th_local_max * img.shape[0] / np.pi, 'g.', ms=ms)
            ax.plot(phi_global_max * img.shape[1] / (2 * np.pi), th_globa_max * img.shape[0] / np.pi, 'r.', ms=ms)

        fig.tight_layout()
        plt.show()

    def save_processed_data_to_file(self, save_path: Path):
        save_path.parent.mkdir(exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self.processed_data_to_dict(), f)

    def processed_data_to_dict(self):
        D = {}
        for k in Scene.PROCESSED_SAVE_KEYS:
            if k in self.__dict__.keys():
                D[k] = self.__dict__[k]

        return D

    def processed_data_from_dict(D: Dict):
        s = Scene()

        for k in D.keys():
            if k in Scene.PROCESSED_SAVE_KEYS:
                s.__dict__[k] = D[k]

        return s

    def save_optimization_runs_to_file(self, save_path: Path):
        save_path.parent.mkdir(exist_ok=True, parents=True)
        with open(save_path, "wb") as f:
            pickle.dump(self.optimization_runs, f)

    def load_optimization_runs_from_file(self, save_path: Path):
        if save_path.exists():
            with open(save_path, "rb") as f:
                self.optimization_runs = pickle.load(f)


class SceneCollection:
    def __init__(self,
                 img_dataset_path: str | Path,
                 cut_lower_hemisphere: bool,
                 multiprocessing: bool,
                 exp_name: str,
                 icosphere_filename: str) -> None:

        if type(img_dataset_path) == str:
            img_dataset_path = Path(img_dataset_path)
        self.img_dataset_path = img_dataset_path

        # create img names from img_dataset_path
        self.img_names = [p.name for p in img_dataset_path.glob("*.exr")]

        self.cut_lower_hemisphere = cut_lower_hemisphere
        self.multiprocessing = multiprocessing
        self.exp_name = exp_name
        self.icosphere_filename = icosphere_filename

    def _preprocess_scene_worker(self,
                                 img_names_batch: List[str],
                                 all_kernel_types: List[str],
                                 convolve_latlon_D: int):
        data_path = utils.get_data_path()
        icosphere_path = data_path / self.icosphere_filename

        for img_name in img_names_batch:
            img_path = self.img_dataset_path / img_name
            img_path = img_path.resolve()
            save_path = utils.get_processed_data_path() / self.exp_name / \
                ("%s.pt" % img_name.split(".")[0])
            save_path.parent.mkdir(exist_ok=True, parents=True)

            # Load previous scene if applicable
            if save_path.exists():
                s = load_processed_scene(self.exp_name, img_name)
            else:
                s = Scene(img_path, icosphere_path)

            if self.cut_lower_hemisphere:
                s.cut_lower_hemisphere()

            # run convolution
            for kernel_type in all_kernel_types:
                # Do not recompute irradiance map if already exists
                if s.irradiance_map_exists(kernel_type):
                    continue

                #p = s.convolve(kernel_type)
                p = s.convolve_latlon(kernel_type, convolve_latlon_D)
                s.store_irradiance_map(p, kernel_type)

            if "local_max" not in s.__dict__.keys() and s.irradiance_map_exists("half_cosine"):
                s.compute_local_max()

            # save the scene
            s.save_processed_data_to_file(save_path)

    def preprocess_scenes(self,
                          all_kernel_types: List[str],
                          convolve_latlon_D: int):
        if self.multiprocessing:
            idx = np.arange(len(self.img_names))
            idx_splits = np.array_split(idx, NUM_PROCESSES)

            processes = []
            for idx_split in idx_splits:
                img_names_batch = utils.split_list(self.img_names, idx_split)
                p = mp.Process(
                    target=self._preprocess_scene_worker,
                    args=(img_names_batch, all_kernel_types, convolve_latlon_D))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
        else:
            self._preprocess_scene_worker(
                self.img_names, all_kernel_types, convolve_latlon_D)


    def _generic_process_scene_worker(self, img_names_batch: List[str], process_fn):
        data_path = utils.get_data_path()
        icosphere_path = data_path / self.icosphere_filename

        for img_name in img_names_batch:
            img_path = self.img_dataset_path / img_name
            img_path = img_path.resolve()
            save_path = utils.get_processed_data_path() / self.exp_name / \
                ("%s.pt" % img_name.split(".")[0])
            save_path.parent.mkdir(exist_ok=True)

            # Load previous scene if applicable
            if save_path.exists():
                s = load_processed_scene(self.exp_name, img_name)
            else:
                s = Scene(img_path, icosphere_path)

            # Process
            process_fn(s)

            # save the scene
            s.save_processed_data_to_file(save_path)


    def generic_process_scenes(self, process_fn):
        if self.multiprocessing:
            num_processes = 128
            idx = np.arange(len(self.img_names))
            idx_splits = np.array_split(idx, num_processes)

            processes = []
            for idx_split in idx_splits:
                img_names_batch = utils.split_list(self.img_names, idx_split)
                p = mp.Process(
                    target=self._generic_process_scene_worker,
                    args=(img_names_batch, process_fn))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
        else:
            self._generic_process_scene_worker(self.img_names, process_fn)


    def energy_harvested_for_kernel(self,
                                    scenes: List[Scene],
                                    kernel_type: str,
                                    noise_level: float,
                                    normalize_per_scene=True):
        E = np.zeros(len(scenes))
        for i, s in enumerate(scenes):
            E[i] = s.energy_harvested(kernel_type, noise_level, normalize_per_scene).mean()

        return E

    def min_energy_harvested_for_kernel(self,
                                    scenes: List[Scene],
                                    kernel_type: str,
                                    noise_level: float,
                                    normalize_per_scene=True):
        E = np.zeros(len(scenes))
        for i, s in enumerate(scenes):
            E[i] = s.energy_harvested(kernel_type, noise_level, normalize_per_scene).min()
        return E




class Optimizer:
    def __init__(self,
                 rot_step_size_deg: float,
                 iter_max: int,
                 num_trials: int,
                 kernel_types: List[str],
                 noise_level: np.ndarray,
                 multiprocessing: bool,
                 exp_name: str,
                 override_prev_runs: bool):
        self.rot_step_size = np.deg2rad(rot_step_size_deg)
        self.iter_max = iter_max
        self.num_trials = num_trials
        self.kernel_types = kernel_types
        self.noise_level = noise_level
        for i in range(len(noise_level)):
            self.noise_level[i] = float(eval(self.noise_level[i]))
        self.multiprocessing = multiprocessing
        self.exp_name = exp_name
        self.override_prev_runs = override_prev_runs

    def _optimization_worker(self, img_name: str):
        optim_save_path = utils.get_optimization_path() / self.exp_name / \
            ("%s.pt" % img_name.split(".")[0])

        # load the scene
        s = load_processed_scene(self.exp_name, img_name) # type:Scene
        s.load_optimization_runs_from_file(optim_save_path)

        # load gradient ascent rotations
        grad_rots = GradientAscentRotations(self.rot_step_size, s.icosphere_path)

        if "face_neighbors" not in s.__dict__.keys():
            s._load_face_neighbors()

        x_init = s.sample_init_points(self.num_trials)

        for noise_level, kernel_type in \
            itertools.product(self.noise_level, self.kernel_types):

            # Keep previous optimization run. Note that x_init is different
            if not self.override_prev_runs and \
                s.optimization_run_exists(kernel_type, noise_level):
                continue

            x_init_curr = x_init
            kernel_type_curr = kernel_type

            p = s.get_irradiance_map(kernel_type_curr)

            x_final, _ = s.gradient_ascent_multi_init(
                p,
                self.iter_max,
                x_init_curr,
                noise_level,
                grad_rots)

            s.log_optimization_run(
                x_init,
                x_final,
                kernel_type,
                noise_level)

        # save the optimization_runs
        s.save_optimization_runs_to_file(optim_save_path)


    def _batch_optimization(self, img_names: List[str]):
        for img_name in img_names:
            self._optimization_worker(img_name)

    def run_optimization(self, scenes: SceneCollection):
        if self.multiprocessing:
            idx = np.arange(len(scenes.img_names))
            idx_splits = np.array_split(idx, NUM_PROCESSES)

            processes = []
            for idx_split in idx_splits:
                img_names_batch = utils.split_list(scenes.img_names, idx_split)
                p = mp.Process(
                    target=self._batch_optimization, args=(img_names_batch,))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
        else:
            self._batch_optimization(scenes.img_names)


def add_noise(meas: np.ndarray, noise_level: float, peak_meas: float) -> np.ndarray:
    """
    Add noise to the measurement given by the peak noise-to-signal ratio.
    """
    if noise_level == np.inf:
        return np.zeros(meas.shape)
    elif noise_level == 0:
        return meas
    else:
        noise = np.random.randn(*meas.shape) * noise_level * peak_meas
        return meas + noise


def load_processed_scene(exp_name, img_name):
    try:
        with open(utils.get_processed_data_path() / exp_name / \
            ("%s.pt" % img_name.split(".")[0]), "rb") as f:
            D = pickle.load(f)
    except EOFError:
        print("Error loading:", img_name)
        sys.exit(1)

    s = Scene.processed_data_from_dict(D)
    return s
