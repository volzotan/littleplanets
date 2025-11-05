"""
Python wrapper for wget to load files specified in a TOML configuration file.
"""

import argparse
import subprocess
from pathlib import Path

import toml
from pydantic import BaseModel

from loguru import logger

class DownloaderConfig(BaseModel):
    dem_url: str
    surface_color_url: str


def _run(cmd: list) -> None:
    subprocess.run([str(e) for e in cmd], check=True)


def download(url: str, filename: Path) -> None:
    if len(str(url)) == 0:
        logger.info("Skipping download: URI is empty")
        return

    if filename.exists():
        logger.info(f"Skipping download: file {filename} already exists")
        return

    if "." not in url:
        logger.error(f"Skipping download: cannot determine file type of URL {url}")
    else:
        if url[url.rfind("."):].lower() == ".tif":
            _run(["wget", url, "-O", filename])
        else:
            original_filename = filename.parent / url[url.rfind("/")+1:]
            logger.warning(f"Non-TIFF download file {original_filename}, attempting conversion")
            _run(["wget", url, "--directory-prefix", filename.parent])
            subprocess.run(["magick", str(original_filename), str(filename)], check=True)

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

    download(config.dem_url, args.output_dir / "dem.tif")
    download(config.surface_color_url, args.output_dir / "surface_color.tif")


if __name__ == "__main__":
    main()
