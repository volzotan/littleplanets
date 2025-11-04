import argparse
from pathlib import Path

import toml
from loguru import logger


def is_different(a: dict, b: dict) -> bool:
    return a != b


def load_toml(filename: Path) -> dict:
    with open(filename, "r") as f:
        return toml.load(f)


def write_toml(filename: Path, data: dict) -> None:
    with open(filename, "w") as f:
        toml.dump(data, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="Main TOML configuration file")
    parser.add_argument("--output", type=Path, default=Path("."), help="Output directory")
    args = parser.parse_args()

    data = load_toml(args.config)

    global_config = {}
    sub_configs = {}

    # collect all global values (global = not in a TOML table block [foo])
    for config_name, config_data in data.items():
        if type(config_data) is dict:
            sub_configs[config_name] = config_data
        else:
            global_config[config_name] = config_data

    # for each table block write the specific and global variables to a toml file of the same name
    for config_name, config_data in sub_configs.items():
        filename = args.output / f"{config_name}.toml"
        combined_config = {**global_config, **config_data}

        if filename.exists() and load_toml(filename) == combined_config:
            logger.info(f"Skip {filename}")
        else:
            write_toml(filename, combined_config)
            logger.info(f"Write {filename}")


if __name__ == "__main__":
    main()
