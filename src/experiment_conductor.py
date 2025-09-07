import math
import os
import subprocess
from pathlib import Path
from typing import Any

import toml

CONFIG_OVERRIDE_FILE = Path("config_hatch_override.toml")

PY_FILE = "src/hatch.py"
INKSCAPE_BIN = "/Applications/Inkscape.app/Contents/MacOS/inkscape"
OUTPUT_DIR = Path("experiment_output")

ARGUMENTS = [
    "build/mapping_color.png",
    "build/mapping_angle_5.png",
    "build/mapping_distance.png",
    "build/mapping_line_length.png",
    "build/mapping_flat.png",
]

VARIABLES = {
    "blur_angle_kernel_size_perc": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0],
    "flowlines_line_distance_end_factor": [0.0, 0.25, 0.5, 0.75, 1.0],
}

num_experiment = 0


def make(config_overwrite) -> None:
    global num_experiment

    num_experiment += 1
    print(f"Executing experiment {num_experiment:> 5}/{total_experiments}")

    filename = f"experiment_{num_experiment}"
    config_file = OUTPUT_DIR / (filename + ".toml")
    image_file = OUTPUT_DIR / (filename + ".png")

    with open(config_file, "w") as f:
        toml.dump(config_overwrite, f)

    subprocess.run(
        [
            "uv",
            "run",
            PY_FILE,
            *ARGUMENTS,
            "--config",
            config_file,
            "--contours",
            "build/contours.npz",
            "--output",
            "build/littleplanets.svg",
        ],
        check=True,
    )

    subprocess.run(
        [INKSCAPE_BIN, "build/littleplanets.svg", f"--export-filename={image_file}", "--export-width=2000", "--export-background=#000000"], check=True
    )


def rec_looping(vars: dict[str, list[Any]], config_overwrite: dict[str, Any] = {}) -> None:
    if len(vars.keys()) > 0:
        key = list(vars.keys())[0]
        for value in vars[key]:
            config_overwrite[key] = value
            vars_copy = vars.copy()
            vars_copy.pop(key)
            rec_looping(vars_copy, config_overwrite)
    else:
        make(config_overwrite)


if __name__ == "__main__":
    global total_experiments

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_experiments = math.prod([len(values) for values in VARIABLES.values()])
    rec_looping(VARIABLES)
