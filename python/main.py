import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import yaml
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import importlib
from typing import Dict
import copy

from sim import SceneCollection, Optimizer
import utils

def add_global_params_to_config(config, global_params):
    new_config = copy.deepcopy(config)
    new_config["params"] = {**global_params, **new_config["params"]}
    return new_config

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def cli_args():
    parser = argparse.ArgumentParser("Simulation")

    parser.add_argument("--config", type=str, help="Configuration file name")

    return parser.parse_args()


def run_preprocessing(scene_collection: SceneCollection,
                      preprocessing_config: Dict):
    scene_collection.preprocess_scenes(
        preprocessing_config["kernel_types"],
        preprocessing_config["convolve_latlon_D"])


def run_optimization(scene_collection: SceneCollection, optim_config: Dict):
    optimizer = instantiate_from_config(optim_config) # type:Optimizer
    optimizer.run_optimization(scene_collection)


def run_simulation():
    args = cli_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    scene_collection = instantiate_from_config(
        add_global_params_to_config(
            config["scene_collection"],
            config["global_params"])
    ) # type:SceneCollection

    if "preprocessing" in config.keys():
        run_preprocessing(scene_collection, config["preprocessing"])

    if "optimization" in config.keys():
        run_optimization(
            scene_collection,
            add_global_params_to_config(
                config["optimization"], config["global_params"]
            )
        )

if __name__ == "__main__":
    run_simulation()
