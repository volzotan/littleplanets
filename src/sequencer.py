import datetime
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
from typing import Any
from itertools import product

import tomllib
import toml
from loguru import logger

NUM_WORKERS = 2

PLANET = "moon"

FILE_CONFIG_BASE = Path(f"config/{PLANET}.toml")
FILE_POI = Path(f"config/{PLANET}_poi.json")
DIR_OUTPUT = Path("sequencer_output")
DIR_BUILD_BASE = Path(f"build_{PLANET}")  # base build dir from which initial files are copied
DIR_DATA = Path(f"data_{PLANET}")

MAKEFILE_TARGET = "run_no_axis"

def _circle_pos(radius: float, num_positions: int) -> list[tuple[float, float]]:
    return [(radius * math.cos(2 * math.pi * i / num_positions), radius * math.sin(2 * math.pi * i / num_positions)) for i in range(num_positions)]


def worker_get_build_dir() -> Path:
    return Path(f"build_{multiprocessing.current_process().name}")


def process(num_experiment: int, config_override: dict[str, Any]) -> None:
    # logger.info(f"Executing experiment {num_experiment:> 5}")

    try:
        filename = f"sequence_{num_experiment}"
        config_file = DIR_OUTPUT / (filename + ".toml")
        image_file = DIR_OUTPUT / (filename + "_0" + ".png")
        render_output_file = DIR_OUTPUT / (filename + "_1" + ".tif")
        mapping_output_file = DIR_OUTPUT / (filename + "_2" + ".png")
        misc_output_file = DIR_OUTPUT / (filename + "_3" + ".png")

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

        # shutil.copy(build_dir / "freestyle.png", misc_output_file)
        # shutil.copy(build_dir / "freestyle.png", misc_output_file)

        # shutil.copy(build_dir.parent / Path(str(build_dir.stem) + "_debug") / "mixture.png", image_file)

        logger.info("processing time {}: {:5.2f}s".format(filename, (datetime.datetime.now() - timer_start).total_seconds()))

    except Exception as e:
        logger.error(f"process failed: {e}")
        raise e


def worker_init() -> None:
    build_dir = worker_get_build_dir()
    logger.info(f"worker init {build_dir}")
    os.makedirs(build_dir, exist_ok=True)

    if DIR_BUILD_BASE is not None and DIR_BUILD_BASE.exists():
        subprocess.run(
            ["rsync", "-av", "--verbose", "--exclude", "*.blend", str(DIR_BUILD_BASE) + "/", str(build_dir) + "/"],
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

    with open(FILE_CONFIG_BASE, "rb") as f:
        base_config = tomllib.load(f)




    variable_sets = []

    for x, z in _circle_pos(4, 25*5):
        variable_sets.append({
            "light_pos_x": x,
            "light_pos_y": 0.0,
            "light_pos_z": z,
            "combine|add_frame": False
        })





    overrides = [{**base_config, **override} for override in variable_sets]

    total_experiments = len(variable_sets)
    completed_experiments = 0
    with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=worker_init) as executor:
        futures = [executor.submit(process, i, config_override) for i, config_override in enumerate(overrides)]

        for future in as_completed(futures):
            completed_experiments += 1
            logger.success(f"processed {completed_experiments}/{total_experiments} | {(completed_experiments / total_experiments):5.2%}")


if __name__ == "__main__":
    main()
