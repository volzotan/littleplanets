"""
Python wrapper for wget to load files specified in a TOML configuration file.
"""

import argparse
import datetime
import subprocess
from pathlib import Path
import os

import toml
from pydantic import BaseModel
import numpy as np
import tifffile
import cdsapi

from loguru import logger

DATETIME_FORMAT = "%Y-%m-%d-%H-%M"


class DownloaderConfig(BaseModel):
    dem_url: str
    surface_color_url: str
    clouds_download: bool = False
    clouds_datetime: datetime.datetime = datetime.datetime.fromisoformat("2025-07-01 00:00")


def _run(cmd: list) -> None:
    subprocess.run([str(e) for e in cmd], check=True)


def download(url: str, filename: Path) -> None:
    if filename.exists():
        logger.info(f"Skipping download: file {filename} already exists")
        return

    if len(str(url)) == 0:
        logger.info(f"Skipping download: URI is empty, writing zero value file to {filename}")
        tifffile.imwrite(filename, np.ones([1, 1, 1], dtype=float))
        return

    if "." not in url:
        logger.error(f"Skipping download: cannot determine file type of URL {url}")
    else:
        if url[url.rfind(".") :].lower() == ".tif":
            _run(["wget", url, "-O", filename])
        else:
            original_filename = filename.parent / url[url.rfind("/") + 1 :]
            logger.warning(f"Non-TIFF download file {original_filename}, attempting conversion")
            _run(["wget", url, "--directory-prefix", filename.parent])
            subprocess.run(["magick", str(original_filename), str(filename)], check=True)


def retrieve_from_cdsapi(timestamp: datetime.datetime, filename: Path) -> None:
    if filename.exists():
        logger.info(f"Skipping download: file {filename} already exists")
        return

    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind", "total_cloud_cover", "land_sea_mask"],
        "year": [str(timestamp.year)],
        "month": [str(timestamp.month)],
        "day": [str(timestamp.day)],
        "time": [timestamp.strftime("%H:00")],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    try:
        client = cdsapi.Client()
        client.retrieve(dataset, request).download(target=filename)
    except Exception as e:
        logger.error(f"Retrieval from the Copernicus Climate Data Store API failed: {e}")


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

    if config.clouds_download:
        path_generic_file = args.output_dir / "cds_clouds.nc"
        path_unique_file = args.output_dir / f"cds_clouds_{config.clouds_datetime.strftime(DATETIME_FORMAT)}.nc"
        retrieve_from_cdsapi(config.clouds_datetime, path_unique_file)

        if path_generic_file.exists() and not path_generic_file.is_symlink():
            logger.debug(f"removing generic CDS cloud data file: {path_generic_file} (expected a symlink, not a file)")
            os.remove(path_generic_file)

        if path_generic_file.is_symlink():
            if os.readlink(path_generic_file) != path_unique_file:
                # the symlink has pointed to another unique file beforehand
                # make only checks the mtime of the unique file and will not detect that data has changed
                # due to a different symlink.
                os.utime(path_unique_file, (os.stat(path_unique_file).st_atime, datetime.datetime.now().timestamp()))
            path_generic_file.unlink()

        os.symlink(path_unique_file.name, path_generic_file)


if __name__ == "__main__":
    main()
