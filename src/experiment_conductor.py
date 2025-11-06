import math
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
from typing import Any

import tomllib
import toml
from loguru import logger

NUM_WORKERS = 2

CONFIG_BASE_FILE = Path("config/earth.toml")
OUTPUT_DIR = Path("experiment_output")
BUILD_DIR_BASE = Path("build_earth") # base build dir from which initial files are copied
DATA_DIR = Path("data_earth")

MAKEFILE_TARGET = "test"

VARIABLES = {
    #     "blur_angle_kernel_size_perc": [0.1, 0.2, 0.3, 0.4, 0.5],
    #     # "blur_color_kernel_size_perc": [0.1, 0.2, 0.3, 0.4, 0.5],
    #     # "blur_distance_kernel_size_perc": [0.1, 0.2, 0.3, 0.4, 0.5],
    #     "flowlines_line_distance_end_factor": [0.25, 0.5, 0.75, 1.0],
    #     # "flowlines_line_distance": [(0.8, 5), (0.8, 10), (0.8, 15)],
    #     # "flowlines_line_max_length": [(3, 9), (3, 12), (3, 16), (3, 20)],
    #     "flowlines_line_max_length": [(3, 3), (6, 6), (12, 12), (16, 16), (20, 20), (30, 30), (40, 40)],
    #     # "flowlines_line_max_length": [(1, 16), (2, 16), (3, 16), (4, 16), (5, 16), (6, 16), (7, 16), (8, 16), (10, 16), (12, 16), (14, 16), (16, 16)],
    #     "flowlines_max_angle_discontinuity": [math.pi / 16, math.pi / 8, math.pi / 4, math.pi / 2]
    # "hatch|flowlines_line_distance": [(0.8, 5.), (0.8, 10.), (0.8, 15.), (0.8, 20.)],
    # "process_blender|mixture": [[0.010, 0.06], [0.015, 0.06], [0.020, 0.06], [0.025, 0.06], [0.030, 0.06], [0.035, 0.06], [0.040, 0.06], [0.045, 0.06], [0.050, 0.06]],
    # "process_blender|mixture": [
    #     [0.03, 0.04],
    #     [0.03, 0.045],
    #     [0.03, 0.050],
    #     [0.03, 0.055],
    #     [0.03, 0.060],
    #     [0.03, 0.065],
    #     [0.03, 0.070],
    #     [0.03, 0.080],
    #     [0.03, 0.090],
    #     [0.03, 0.100],
    # ],
    "adjust_camera|camera_focal_length": [30, 50, 70, 150]
}


def worker_get_build_dir() -> Path:
    return Path(f"build_{multiprocessing.current_process().name}")


def process(num_experiment: int, config_override: dict[str, Any]) -> None:
    # logger.info(f"Executing experiment {num_experiment:> 5}")

    filename = f"experiment_{num_experiment}"
    config_file = OUTPUT_DIR / (filename + ".toml")
    image_file = OUTPUT_DIR / (filename + ".png")

    # restructure "foo|bar: 3" to [foo] bar: 3, i.e. split off 'bar' into a sub-dict
    config_override_restructured = {}
    for key, value in config_override.items():
        if "|" in key:
            parent, child = key.split("|")[0:2]

            if parent not in config_override_restructured:
                config_override_restructured[parent] = {child: value}
            else:
                config_override_restructured[parent][child] = value
        else:
            config_override_restructured[key] = value

    with open(config_file, "w") as f:
        toml.dump(config_override_restructured, f)

    build_dir = worker_get_build_dir()

    try:
        subprocess.run(
            [
                "make",
                MAKEFILE_TARGET,
                f"CONFIG={config_file}",
                f"DIR_BUILD={build_dir}",
                f"OUTPUT_PNG={image_file}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        # shutil.copy(build_dir.parent / Path(str(build_dir.stem) + "_debug") / "mixture.png", image_file)

    except Exception as e:
        logger.error(f"Subprocess failed: {e}")
        raise e


def rec_looping(variables: dict[str, list[Any]], config_override: dict[str, Any] = {}) -> list[dict[str, Any]]:
    if len(variables.keys()) > 0:
        configs = []
        key = list(variables.keys())[0]
        for value in variables[key]:
            config_override_copy = config_override.copy()
            config_override_copy[key] = value

            variables_copy = variables.copy()
            variables_copy.pop(key)

            configs += rec_looping(variables_copy, config_override_copy)
        return configs
    else:
        return [config_override]


def worker_init() -> None:
    build_dir = worker_get_build_dir()
    logger.info(f"worker init {build_dir}")
    os.makedirs(build_dir, exist_ok=True)

    if BUILD_DIR_BASE is not None:
        subprocess.run(
            ["rsync", "-av", str(BUILD_DIR_BASE) + "/", str(build_dir) + "/"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

    logger.info(f"worker init {build_dir} completed")


def main() -> None:
    shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_experiments = math.prod([len(values) for values in VARIABLES.values()])

    with open(CONFIG_BASE_FILE, "rb") as f:
        base_config = tomllib.load(f)

    overrides = [{**base_config, **override} for override in rec_looping(VARIABLES)]

    completed_experiments = 0
    with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=worker_init) as executor:
        futures = [executor.submit(process, i, config_override) for i, config_override in enumerate(overrides)]

        for future in as_completed(futures):
            completed_experiments += 1
            logger.info(f"processed {completed_experiments}/{total_experiments} | {(completed_experiments / total_experiments):5.2%}")


if __name__ == "__main__":
    main()
