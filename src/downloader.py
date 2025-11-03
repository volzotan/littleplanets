"""
Python wrapper for wget to load files specified in a TOML configuration file.
"""

import argparse
import subprocess
from pathlib import Path

import toml
from pydantic import BaseModel


class DownloaderConfig(BaseModel):
    dem_uri: Path
    surface_color_uri: Path


def _run(cmd: list) -> None:
    subprocess.run([str(e) for e in cmd], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--config", type=Path, help="Configuration file for values passed on to the script [TOML]")
    args = parser.parse_args()

    config = None
    if args.config is not None:
        with open(args.config, "r") as f:
            data = toml.load(f)
            config = DownloaderConfig(**data)

    filename_dem = args.output_dir / "dem.tif"
    if not filename_dem.exists():
        _run(["wget", config.dem_uri, "-O", filename_dem])

    filename_surface_color = args.output_dir / "surface_color.tif"
    if not filename_surface_color.exists():
        _run(["wget", config.surface_color_uri, "-O", filename_surface_color])


if __name__ == "__main__":
    main()
