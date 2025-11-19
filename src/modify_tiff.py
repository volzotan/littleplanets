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

from loguru import logger

class ModifyDemConfig(BaseModel):
    scaling_factor: float | None = None
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


def _read(input_path: Path) -> np.ndarray:
    with rasterio.open(input_path) as src:
        return src.read(), src.crs, src.transform


def _rescale(data: np.ndarray, scaling_factor: float) -> np.ndarray:

    new_size = (
        max(int(data.shape[1] * scaling_factor), 1),
        max(int(data.shape[0] * scaling_factor), 1)
    )

    return cv2.resize(data, new_size, interpolation=cv2.INTER_AREA).astype(
        data.dtype
    )


def _write(output_path: Path, data: np.ndarray, options: dict[str, Any] = {}) -> None:
    config = {
        "driver": "GTiff",
        "height": data.shape[-2],
        "width": data.shape[-1],
        "count": 1,
        "dtype": data.dtype,
    }

    config = config | options

    with rasterio.open(output_path, "w", **config) as dst:
        dst.write(data)


def _clip(data: np.ndarray, floor: float | None, ceil: float | None) -> np.ndarray:
    return np.clip(data, floor, ceil).astype(data.dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="input filename")
    parser.add_argument("output", type=Path, help="output filename")
    parser.add_argument("--config", type=Path, help="Configuration file [TOML]")
    args = parser.parse_args()

    config = None
    if args.config is not None:
        with open(args.config, "r") as f:
            data = toml.load(f)
            config = ModifyDemConfig(**data)

    os.makedirs(args.output.parent, exist_ok=True)

    if args.input is None or not args.input.exists():
        logger.warning(f"Empty input file {args.input}. Abort.")
        return

    data, data_crs, data_transform = _read(args.input)
    options = {"crs": data_crs, "transform": data_transform}
    data = np.transpose(data, (1, 2, 0))

    if config.scaling_factor is not None and config.scaling_factor != 1.0:
        data = _rescale(data, config.scaling_factor)
        options["transform"] = data_transform * data_transform.scale(1 / config.scaling_factor, 1 / config.scaling_factor)

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
