"""
Download files specified in a TOML configuration file.
"""

import argparse
import datetime
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse
import os

import requests
import toml
from pydantic import BaseModel
import numpy as np
import tifffile
import cdsapi

from loguru import logger

import modify_tiff

DATETIME_FORMAT = "%Y-%m-%d-%H-%M"
TILES_SCALING_FACTOR: float = 0.25


class DownloaderConfig(BaseModel):
    dem_url: str
    surface_color_url: str
    clouds_download: bool = False
    clouds_datetime: datetime.datetime = datetime.datetime.fromisoformat("2025-07-01 00:00")


def _server_filename(response: requests.Response) -> str:
    """Derive the server-side filename from a response, preferring Content-Disposition."""
    disposition = response.headers.get("Content-Disposition", "")
    match = re.search(r'filename\*=(?:UTF-8\'\')?"?([^";]+)"?', disposition, flags=re.IGNORECASE)
    if not match:
        match = re.search(r'filename="?([^";]+)"?', disposition, flags=re.IGNORECASE)
    if match:
        return unquote(match.group(1)).strip()
    return Path(unquote(urlparse(response.url).path)).name


def download(url: str, filename: Path) -> None:
    if filename.exists():
        logger.info(f"Skipping download: file {filename} already exists")
        return

    if len(str(url)) == 0:
        logger.info(f"Skipping download: URI is empty, writing zero value file to {filename}")
        tifffile.imwrite(filename, np.ones([1, 1, 1], dtype=float))
        return

    filename.parent.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Starting download: {url} -> {filename}")
    with requests.get(url, stream=True, allow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        original_filename = filename.parent / _server_filename(response)

        logger.debug(f"Temporary filename {original_filename}")

        total = int(response.headers.get("Content-Length", 0))
        downloaded, next_pct = 0, 10

        if not original_filename.exists():
            with open(original_filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    f.write(chunk)

                    # logging progress
                    downloaded += len(chunk)
                    while total and downloaded * 100 >= total * next_pct:
                        logger.info(f"{original_filename.name}: {next_pct}%")
                        next_pct += 10
        else:
            logger.debug(f"Skipping download: intermediate file {original_filename} already exists")

    suffix = original_filename.suffix.lower()
    if suffix in (".tif", ".tiff"):
        if original_filename != filename:
            shutil.move(str(original_filename), str(filename))
    elif suffix == ".zip":
        with zipfile.ZipFile(original_filename, "r") as z:
            tif_members = [m for m in z.namelist() if m.lower().endswith((".tif", ".tiff"))]
            if len(tif_members) == 0:
                logger.error(f"No TIFF found in archive {original_filename.name}")
                return
            if len(tif_members) == 1:
                logger.info(f"Extracting TIFF from downloaded archive {original_filename.name}")
                with z.open(tif_members[0]) as src, open(filename, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            else:
                # This is an archive with multiple tiffs that need to be merged to a single image.
                # Assume this is a GEBCO bathymetry elevation dataset

                logger.info(f"Extracting {len(tif_members)} TIFFs from archive {original_filename.name} to {filename.parent}")

                zip_entry_tile_filename_mapping = [(member, filename.parent / Path(member).name) for member in tif_members]
                scaled_tile_filenames = []

                for zip_entry, tile_filename in zip_entry_tile_filename_mapping:
                    if not tile_filename.exists():
                        with z.open(zip_entry) as src, open(tile_filename, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                            logger.debug(f"Written from {filename} tile: {tile_filename}")
                    else:
                        logger.debug(f"Skipping unpacking tile {tile_filename} from archive {original_filename}: file already exists")

                    scaled_tile_filename = tile_filename.parent / Path("scaled_" + str(tile_filename.name))
                    if not scaled_tile_filename.exists():
                        modify_tiff.downscale_and_write(tile_filename, scaled_tile_filename, TILES_SCALING_FACTOR)

                    else:
                        logger.debug(f"Skipping scaling tile {scaled_tile_filename} from archive {original_filename}: file already exists")
                    scaled_tile_filenames.append(scaled_tile_filename)

                modify_tiff.merge_and_write(scaled_tile_filenames, filename)

    else:
        logger.warning(f"Non-TIFF download file {original_filename.name}, attempting conversion")
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
    if args.config is not None and args.config.exists():
        with open(args.config, "r") as f:
            data = toml.load(f)
            config = DownloaderConfig(**data)
    else:
        logger.warning("No config file found")

    download(config.dem_url, args.output_dir / "dem.tif")
    download(config.surface_color_url, args.output_dir / "surface_color.tif")

    if config.clouds_download:
        path_generic_file = args.output_dir / "cds_clouds.nc"
        path_unique_file = args.output_dir / f"cds_clouds_{config.clouds_datetime.strftime(DATETIME_FORMAT)}.nc"
        retrieve_from_cdsapi(config.clouds_datetime, path_unique_file)

        if path_generic_file.exists():
            if path_generic_file.is_symlink():
                if path_generic_file.resolve() == path_unique_file.resolve():
                    # symlink exists and points to the right file, do nothing
                    pass
                else:
                    logger.debug(f"updating mtime of file behind symlink")
                    # the symlink has pointed to another unique file beforehand
                    # make only checks the mtime of the unique file and will not detect that data has changed
                    # due to a different symlink.
                    os.utime(path_unique_file, (os.stat(path_unique_file).st_atime, datetime.datetime.now().timestamp()))
                    path_generic_file.unlink()
                    os.symlink(path_unique_file.name, path_generic_file)
            else:
                logger.debug(f"removing generic CDS cloud data file: {path_generic_file} (expected a symlink, not a file)")
                os.remove(path_generic_file)
                os.symlink(path_unique_file.name, path_generic_file)
        else:
            os.symlink(path_unique_file.name, path_generic_file)


if __name__ == "__main__":
    main()
