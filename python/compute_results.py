import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
import argparse

import main
import sim
import utils


def load_scene_by_name(scene_collection, name):
    s = sim.load_processed_scene(scene_collection.exp_name, name)
    
    optim_save_path = utils.get_optimization_path() / scene_collection.exp_name / \
        ("%s.pt" % name.split(".")[0])
    s.load_optimization_runs_from_file(optim_save_path)
    
    return s


def parse_args():
    parser = argparse.ArgumentParser("Simulation")

    parser.add_argument("--config", type=str, help="Configuration file name")

    return parser.parse_args()


def run_compute_results():
    args = parse_args()
    config_file = args.config

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    scene_collection = main.instantiate_from_config(
        main.add_global_params_to_config(
            config["scene_collection"], 
            config["global_params"])
    )

    # Load all scenes
    scenes = []
    for name in tqdm(scene_collection.img_names):        
        s = load_scene_by_name(scene_collection, name)
        scenes.append(s)

    multimodal_idx = []
    unimodal_idx = []
    for i, s in enumerate(scenes):
        if s.local_max.sum() > 1:
            multimodal_idx.append(i)
        else:
            unimodal_idx.append(i)
    multimodal_idx = np.asarray(multimodal_idx, dtype=np.int64)
    unimodal_idx = np.asarray(unimodal_idx, dtype=np.int64)

    print("Total         : %d" % len(scenes))
    print("Num unimodal  : %d" % len(unimodal_idx))
    print("Num multimodal: %d" % len(multimodal_idx))

    
    kernel_types = config["optimization"]["params"]["kernel_types"]
    tilt_angles = np.asarray([eval(k.split("_")[1]) for k in kernel_types])
    noise_levels = config["optimization"]["params"]["noise_level"]
    noise_levels = [float(eval(n)) for n in noise_levels]

    # Only plot results for the zero-noise simulation
    ni = 0
    assert noise_levels[ni] == 0

    normalize_by_scene = True

    # Compute mean energy harvested
    E = np.zeros((len(kernel_types), len(scenes)))
    for ki in tqdm(range(len(kernel_types))):
        E[ki] = scene_collection.energy_harvested_for_kernel(scenes, kernel_types[ki], noise_levels[ni], normalize_by_scene)
    
    # Average case (multimodal scenes)
    fig, ax = plt.subplots()
    ax.plot(tilt_angles, E[:,multimodal_idx].mean(1) * 100, ".-")
    ax.set_xlabel(r"Tilt Angle $\Delta\theta$", fontsize=8)
    ax.set_ylabel(r"Harvested Energy (\%)", fontsize=8)
    ax.set_xlim([0, 90])
    ax.set_ylim([94, 100])
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    fig.tight_layout()
    fig.show()

    # Average case (unimodal scenes)
    fig, ax = plt.subplots()
    ax.plot(tilt_angles, E[:,unimodal_idx].mean(1) * 100, ".-")
    ax.set_xlabel(r"Tilt Angle $\Delta\theta$", fontsize=8)
    ax.set_ylabel(r"Harvested Energy (\%)", fontsize=8)
    ax.set_xlim([0, 90])
    ax.set_ylim([99.75, 100])
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    fig.tight_layout()
    fig.show()

    plt.show()


if __name__ == "__main__":
    run_compute_results()