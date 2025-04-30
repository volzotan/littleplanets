from pathlib import Path

import rasterio
from rasterio.warp import calculate_default_transform, Resampling


def downscale(input_path: Path, output_path: Path, scaling_factor: float) -> None:
    """
    Downscale GEBCO GeoTiff images
    """

    with rasterio.open(input_path) as src:
        data = src.read(
            out_shape=(
                src.count,
                int(src.height * scaling_factor),
                int(src.width * scaling_factor),
            ),
            resampling=Resampling.bilinear,
        )

        transform = src.transform * src.transform.scale(
            (src.width / data.shape[-1]), (src.height / data.shape[-2])
        )

        config = {
            "driver": "GTiff",
            "height": data.shape[-2],
            "width": data.shape[-1],
            "count": 1,
            "dtype": data.dtype,
            "crs": src.crs,
            "transform": transform,
        }

        with rasterio.open(output_path, "w", **config) as dst:
            dst.write(data)


def reproject(src: Path, dst: Path) -> None:
    dst_crs = "ESRI:54029"

    with rasterio.open(src) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update(
            {"crs": dst_crs, "transform": transform, "width": width, "height": height}
        )

        with rasterio.open(dst, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                band_arr = src.read(i)

                rasterio.warp.reproject(
                    source=band_arr,
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                )


if __name__ == "__main__":
    input_path = Path("assets", "Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tiff")
    output_path = Path("assets", "Lunar_DEM_resized.tif")
    scaling_factor = 0.25

    downscale(input_path, output_path, scaling_factor)
