import os
from pathlib import Path
from typing import Any

import rasterio
from rasterio.warp import calculate_default_transform, Resampling

import argparse
from pathlib import Path
import numpy as np
import cv2
import toml
from pydantic import BaseModel, Field
import psutil
import time
import random

from loguru import logger

LOW_MEMORY_SLEEP_DURATION = 10.0


class ModifyTiffConfig(BaseModel):
    # scaling_factor: float | None = None
    resize_width: int | None = None
    convert_uint8: bool = False

    contrast_stretching: list[float] | None = None  # [lower_value, upper_value]

    contrast_increase: float | None = None
    contrast_grid_size: int = 4

    blur: float | None = None  # kernel size as percentage of the longest side of the image
    floor: float | None = None
    ceil: float | None = None

    threshold: float | None = None


# def reproject(src: Path, dst: Path) -> None:
#     dst_crs = "ESRI:54029"
#
#     with rasterio.open(src) as src:
#         transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
#         kwargs = src.meta.copy()
#         kwargs.update({"crs": dst_crs, "transform": transform, "width": width, "height": height})
#
#         with rasterio.open(dst, "w", **kwargs) as dst:
#             for i in range(1, src.count + 1):
#                 band_arr = src.read(i)
#
#                 rasterio.warp.reproject(
#                     source=band_arr,
#                     destination=rasterio.band(dst, i),
#                     src_transform=src.transform,
#                     src_crs=src.crs,
#                     dst_transform=transform,
#                     dst_crs=dst_crs,
#                     resampling=Resampling.nearest,
#                 )


def _read(input_path: Path) -> tuple[np.ndarray, Any, Any]:
    with rasterio.open(input_path) as src:
        return src.read(), src.crs, src.transform


def _write(output_path: Path, data: np.ndarray, options: dict[str, Any] = {}) -> None:
    config = {
        "driver": "GTiff",
        "height": data.shape[-2],
        "width": data.shape[-1],
        "count": data.shape[0],
        "dtype": data.dtype,
    }

    config = config | options

    with rasterio.open(output_path, "w", **config) as dst:
        dst.write(data)


def _clip(data: np.ndarray, floor: float | None, ceil: float | None) -> np.ndarray:
    return np.clip(data, floor, ceil).astype(data.dtype)


def _contrast_stretch(data: np.ndarray, src_range: list[float], target_range: list[float] = [0.0, 255.0]) -> np.ndarray:
    clipped = np.clip(data, src_range[0], src_range[1])
    normalized = (clipped - src_range[0]) / (src_range[1] - src_range[0])
    stretched = normalized * (target_range[1] - target_range[0]) + target_range[0]
    return stretched.astype(data.dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="input filename")
    parser.add_argument("output", type=Path, help="output filename")
    parser.add_argument("--config", type=Path, help="Configuration file (TOML)")
    parser.add_argument(
        "--pause-below-minimum-available-memory",
        type=float,
        default=0.0,
        help="Do not start execution if available system memory is below the given threshold (MB)",
    )
    args = parser.parse_args()

    config = None
    if args.config is not None:
        with open(args.config, "r") as f:
            data = toml.load(f)
            config = ModifyTiffConfig(**data)

    os.makedirs(args.output.parent, exist_ok=True)

    if args.input is None or not args.input.exists():
        logger.warning(f"Empty input file {args.input}. Abort.")
        return

    if args.pause_below_minimum_available_memory > 0:
        # wait for a random amount of time so concurrent executions
        # don't try to read the file into memory at the same time
        wait_duration = random.uniform(0, LOW_MEMORY_SLEEP_DURATION)
        logger.debug(f"Initial wait: {wait_duration:5.2f}s")
        time.sleep(wait_duration)

        def _get_available_memory() -> float:
            return getattr(psutil.virtual_memory(), "available") * 2**-20  # megabytes

        available = _get_available_memory()

        while _get_available_memory() < args.pause_below_minimum_available_memory:
            logger.warning(f"Available memory {available:5.2f} is below the threshold {args.pause_below_minimum_available_memory:5.2f}")
            time.sleep(LOW_MEMORY_SLEEP_DURATION)
            available = _get_available_memory()
        else:
            logger.info(f"Available memory {available:5.2f} is sufficient, starting execution")

    data, data_crs, data_transform = _read(args.input)
    options = {"crs": data_crs, "transform": data_transform}
    data = np.transpose(data, (1, 2, 0))

    if config.convert_uint8:
        data = data.astype(np.uint8)

    if config.resize_width is not None:
        scaling_factor = config.resize_width / data.shape[1]
        new_size = (config.resize_width, int(data.shape[0] * scaling_factor))

        data = cv2.resize(data, new_size, interpolation=cv2.INTER_AREA).astype(data.dtype)
        options["transform"] = data_transform * data_transform.scale(1 / scaling_factor, 1 / scaling_factor)

    if config.contrast_stretching is not None:
        data = _contrast_stretch(data, config.contrast_stretching)

    if config.contrast_increase is not None and config.contrast_increase > 0:
        clahe = cv2.createCLAHE(clipLimit=config.contrast_increase, tileGridSize=(config.contrast_grid_size, config.contrast_grid_size))
        lab = cv2.cvtColor(data, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        lab_enhanced = cv2.merge((clahe.apply(l), a, b))
        data = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    if config.blur is not None and config.blur > 0:
        kernel_size = int(max(*data.shape) * (config.blur / 100.0))

        if kernel_size >= 3:
            data = cv2.blur(data, (kernel_size, kernel_size))
        else:
            logger.warning(f"blur kernel size below 3 (config.blur {config.blur})")

    if config.floor is not None or config.ceil is not None:
        data = _clip(data, config.floor, config.ceil)

    if config.threshold is not None:
        mask = data > config.threshold
        data[mask] = 255.0
        data[~mask] = 0.0

    if len(data.shape) == 2:
        data = data[:, :, np.newaxis]

    data = np.transpose(data, (2, 0, 1))
    _write(args.output, data, options=options)


if __name__ == "__main__":
    main()
