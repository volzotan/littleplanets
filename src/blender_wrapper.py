import argparse
import subprocess
from pathlib import Path

import toml

"""
Python wrapper for executing python scripts within Blender with arguments from a TOML configuration file.
Required in order to detected changes in the config file with make to re-execute if necessary.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("blender_binary", type=Path, help="Path to the Blender binary")
    parser.add_argument("blender_file", type=Path, help="Path to the Blender file [BLEND]")
    parser.add_argument("python_script", type=Path, help="Path to the Python script executed within Blender")
    parser.add_argument("--config", type=Path, help="Configuration file for values passed on to the script [TOML]")
    parser.add_argument("--params", nargs="+", type=str, help="Additional strings passed as parameters")
    args = parser.parse_args()

    config_values = []
    with open(args.config) as f:
        data = toml.load(f)
        for key, value in data.items():
            key = key.replace("_", "-")
            config_values += [f"--{key}", str(value)]

    params_values = []
    if args.params is not None:
        for val in args.params:
            params_values += val.split(" ")

    command = [args.blender_binary, args.blender_file, "--background", "--python", args.python_script, "--"] + config_values + params_values
    # print(" ".join([str(e) for e in command]))

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
