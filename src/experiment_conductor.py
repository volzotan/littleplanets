import datetime
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

PLANET = "mars"
PLANET = "moon"

FILE_CONFIG_BASE = Path(f"config/{PLANET}.toml")
FILE_POI = Path(f"config/{PLANET}_poi.json")
DIR_OUTPUT = Path("experiment_output")
DIR_BUILD_BASE = Path(f"build_{PLANET}")  # base build dir from which initial files are copied
DIR_DATA = Path(f"data_{PLANET}")

# MAKEFILE_TARGET = "run"
MAKEFILE_TARGET = "run_no_overlays"

VARIABLES = {
    # "blur_angle_kernel_size_perc": [0.1, 0.2, 0.3, 0.4, 0.5],
    # "blur_color_kernel_size_perc": [0.1, 0.2, 0.3, 0.4, 0.5],
    # "blur_distance_kernel_size_perc": [0.1, 0.2, 0.3, 0.4, 0.5],
    # "hatch|flowlines_line_distance_end_factor": [0.25, 0.5, 0.75, 1.0],
    # "hatch|flowlines_line_distance": [(0.6, 3.0),(0.6, 5.0),(0.6, 7.0),(0.6, 10.0), (0.6, 12.0), (0.6, 15.0), (0.6, 20.0)],
    # "hatch|flowlines_line_max_length": [(3, 9), (3, 12), (3, 16), (3, 20)],
    # "flowlines_line_max_length": [(3, 3), (6, 6), (12, 12), (16, 16), (20, 20), (30, 30), (40, 40)],
    # "flowlines_line_max_length": [(1, 16), (2, 16), (3, 16), (4, 16), (5, 16), (6, 16), (7, 16), (8, 16), (10, 16), (12, 16), (14, 16), (16, 16)],
    # "flowlines_max_angle_discontinuity": [math.pi / 16, math.pi / 8, math.pi / 4, math.pi / 2]
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
    # "adjust_camera|camera_focal_length": [10, 15, 20, 30, 50, 90, 150, 300],
    # "mesh|scale": [0.01, 0.02, 0.04, 0.06, 0.08],
    # "downloader|clouds_datetime": [
    #     "2025-07-01 12:00",
    #     "2025-07-02 12:00",
    #     "2025-07-03 12:00",
    #     "2025-07-04 12:00",
    #     "2025-07-05 12:00",
    #     "2025-07-06 12:00",
    #     "2025-07-07 12:00",
    # ],
    # "mesh|scale": [0.05, 0.07, 0.09, 0.11],
    # "mesh|blur": [1, 10, 50, 100, 150, 200],
    # "adjust_scene|camera_focal_length": [15, 20, 30, 50, 90, 150],
    # "modify_dem|blur": [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
    # "process_blender|mixture": [[0.04, 0.06], [0.04, 0.08], [0.04, 0.1], [0.04, 0.15], [0.04, 0.20], [0.04, 0.25], [0.04, 0.30], [0.04, 0.35], [0.04, 0.40]]
    # "process_blender|mixture": [[0.04, 0.2], [0.05, 0.20], [0.06, 0.20], [0.07, 0.20], [0.08, 0.20], [0.09, 0.20], [0.10, 0.20]]
    # "hatch|flowlines_line_max_length": [(3, 3), (6, 6), (12, 12), (16, 16), (20, 20), (30, 30), (40, 40), (50, 50), (60, 60), (70, 70), (80, 80)],
    # "hatch|flowlines_line_max_length": [(5, 25), (10, 25), (15, 25), (20, 25), (25, 25)],
    # "hatch|flowlines_line_distance_end_factor": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    # "hatch|flowlines_line_max_length": [[3, 25], [4, 25], [5, 25], [7.5, 25], [10, 25], [12.5, 25], [15, 25], [20, 25], [25, 25]],
    # "hatch|flowlines_line_distance": [[0.6, 4.],[0.6, 5.],[0.6, 6.],[0.6, 7.],[0.6, 8.],[0.6, 9.],[0.6, 10.],[0.6, 11.],[0.6, 12.],[0.6, 13.],[0.6, 14.]],
    # "adjust_scene|light_pos_y": [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15],
    # "adjust_scene|light_pos_x": [1., 2., 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    # "hatch|flowlines_max_angle_discontinuity": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.57],
    # "mesh|blur": [1, 10, 50, 100, 150, 200, 250, 300, 350, 400, 500],
    # "hatch|flowlines_line_distance": [[0.6, 4.],[0.6, 6.],[0.6, 8.],[0.6, 10.],[0.6, 12.],[0.6, 14.]],
    "modify_surfacecolor|contrast_grid_size": [4, 8, 12],
    "modify_surfacecolor|contrast_increase": [0., 0.5, 1., 2., 3., 4., 5., 6.],
    # "process_blender|contrast_increase": [0., 0.5, 1., 2., 3., 4., 5., 6.],
}


def worker_get_build_dir() -> Path:
    return Path(f"build_{multiprocessing.current_process().name}")


def process(num_experiment: int, config_override: dict[str, Any]) -> None:
    # logger.info(f"Executing experiment {num_experiment:> 5}")

    try:
        filename = f"experiment_{num_experiment}"
        config_file = DIR_OUTPUT / (filename + ".toml")
        image_file = DIR_OUTPUT / (filename + "_0" + ".png")
        render_output_file = DIR_OUTPUT / (filename + "_1" + ".tif")
        mapping_output_file = DIR_OUTPUT / (filename + "_2" + ".png")
        freestyle_output_file = DIR_OUTPUT / (filename + "_3" + ".png")

        timer_start = datetime.datetime.now()

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

        # delete the cloud netCDF file so a redownload is required
        # TODO: issue: the data dir is shared across workers
        clouds_file = DIR_DATA / "cds_clouds.nc"
        if clouds_file.exists():
            clouds_file.unlink()

        cmd = [
            "make",  # "-j4",
            "setup",
            MAKEFILE_TARGET,
            f"CONFIG_FILE={config_file}",
            f"DIR_DATA={DIR_DATA}",
            f"DIR_BUILD={build_dir}",
            f"POI_FILE={FILE_POI}",
            f"OUTPUT_PNG={image_file}",
        ]

        logger.debug(f"make {cmd}")

        subprocess.run(
            cmd,
            # stdout=subprocess.DEVNULL,
            # stderr=subprocess.DEVNULL,
            check=True,
        )

        shutil.copy(build_dir / "image.tif", render_output_file)
        shutil.copy(build_dir / "mapping_distance.png", mapping_output_file)
        # shutil.copy(build_dir / "freestyle.png", freestyle_output_file)

        # shutil.copy(build_dir.parent / Path(str(build_dir.stem) + "_debug") / "mixture.png", image_file)

        logger.info("processing time {}: {:5.2f}s".format(filename, (datetime.datetime.now() - timer_start).total_seconds()))

    except Exception as e:
        logger.error(f"process failed: {e}")
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

    if DIR_BUILD_BASE is not None and DIR_BUILD_BASE.exists():
        subprocess.run(
            ["rsync", "-av", "--exclude", "*.toml", "--exclude", "*.blend", str(DIR_BUILD_BASE) + "/", str(build_dir) + "/"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

    logger.info(f"worker init {build_dir} completed")


def main() -> None:
    try:
        shutil.rmtree(DIR_OUTPUT)
    except FileNotFoundError:
        pass
    os.makedirs(DIR_OUTPUT)

    total_experiments = math.prod([len(values) for values in VARIABLES.values()])

    with open(FILE_CONFIG_BASE, "rb") as f:
        base_config = tomllib.load(f)

    overrides = [{**base_config, **override} for override in rec_looping(VARIABLES)]

    completed_experiments = 0
    with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=worker_init) as executor:
        futures = [executor.submit(process, i, config_override) for i, config_override in enumerate(overrides)]

        for future in as_completed(futures):
            completed_experiments += 1
            logger.success(f"processed {completed_experiments}/{total_experiments} | {(completed_experiments / total_experiments):5.2%}")


if __name__ == "__main__":
    main()
