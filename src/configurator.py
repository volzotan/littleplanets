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
    parser.add_argument("config", type=Path, default="config.toml", help="Main TOML configuration file")
    parser.add_argument("--output", type=Path, default=Path("."), help="Output directory")
    args = parser.parse_args()

    data = load_toml(args.config)
    for config_name, config_data in data.items():
        filename = args.output / f"{config_name}.toml"

        if filename.exists() and load_toml(filename) == config_data:
            logger.info(f"Skip {filename}")
        else:
            write_toml(filename, config_data)
            logger.info(f"Write {filename}")


if __name__ == "__main__":
    main()
