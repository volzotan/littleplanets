import math
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import tomllib
import toml
from loguru import logger

NUM_WORKERS = 4

CONFIG_BASE_FILE = Path("config_hatch.toml")
CONFIG_OVERRIDE_FILE = Path("config_hatch_override.toml")

PY_FILE = "src/hatch.py"
INKSCAPE_BIN = "/Applications/Inkscape.app/Contents/MacOS/inkscape"
TEMP_DIR = Path("experiment_temp")
OUTPUT_DIR = Path("experiment_output")

ARGUMENTS = [
    "build/mapping_color.npy",
    "build/mapping_angle_5.png",
    "build/mapping_distance.png",
    "build/mapping_line_length.png",
    "build/mapping_flat.png",
]

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
    "flowlines_line_distance": [(0.8, 5), (0.8, 10), (0.8, 15), (0.8, 20)],
}


def process(num_experiment: int, config_override: dict[str, Any]) -> None:
    # logger.info(f"Executing experiment {num_experiment:> 5}")

    filename = f"experiment_{num_experiment}"
    config_file = OUTPUT_DIR / (filename + ".toml")
    image_file = OUTPUT_DIR / (filename + ".png")

    with open(config_file, "w") as f:
        toml.dump(config_override, f)

    svg_path = TEMP_DIR / f"{num_experiment}.svg"

    try:
        subprocess.run(
            [
                "uv",
                "run",
                PY_FILE,
                *ARGUMENTS,
                "--config",
                config_file,
                # "--contours",
                # "build/contours.npz",
                "--output",
                svg_path,
                "--palette-color",
                "240",
                "126",
                "50",
                "--palette-color",
                "65",
                "102",
                "174",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        subprocess.run(
            [INKSCAPE_BIN, svg_path, f"--export-filename={image_file}", "--export-width=2000", "--export-background=#000000"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        os.remove(svg_path)

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


def main() -> None:
    shutil.rmtree(TEMP_DIR)
    shutil.rmtree(OUTPUT_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_experiments = math.prod([len(values) for values in VARIABLES.values()])
    overrides = rec_looping(VARIABLES)

    with open(CONFIG_BASE_FILE, "rb") as f:
        base_config = tomllib.load(f)

    overrides = [{**base_config, **override} for override in overrides]

    # [process(i, config_override) for i, config_override in enumerate(overrides)]

    completed_experiments = 0
    with ProcessPoolExecutor(NUM_WORKERS) as executor:
        futures = [executor.submit(process, i, config_override) for i, config_override in enumerate(overrides)]

        for future in as_completed(futures):
            completed_experiments += 1
            logger.info(f"processed {completed_experiments}/{total_experiments} | {(completed_experiments / total_experiments):5.2%}")


if __name__ == "__main__":
    main()
