import argparse
import itertools
import json
import os
from pathlib import Path

from loguru import logger

import cv2
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000, deltaE_cie76, deltaE_ciede94

RESIZE_SIZE = (600, 600)
DIR_DEBUG = Path("debug")

MIN_RATIO_THRESHOLD = None  # 0.15

ERROR_THRESHOLD = 0.050

# BASE_DIR = Path("build_earth")
# NUM_COLORS = 3

BASE_DIR = Path("build_mars")
NUM_COLORS = 2

# BASE_DIR = Path("build_jupiter")
# NUM_COLORS = 3


IMAGE_PATH = BASE_DIR / "image.tif"
BACKGROUND_PATH = BASE_DIR / "mapping_background.png"


def calculate(image_color: np.ndarray, colors: list[list[int]], bg_mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    palette = np.array(colors, dtype=int)
    palette = np.delete(palette, np.where(np.min(palette, axis=1) < 0), axis=0)  # remove invalid palette colors
    palette = palette.astype(np.uint8)

    # HSV H-ONLY

    palette_hsv = cv2.cvtColor(palette[:, np.newaxis, :], cv2.COLOR_RGB2HSV)[:, 0, :]
    mapping_color_hsv = cv2.cvtColor(image_color, cv2.COLOR_BGR2HSV)

    def diff_h(c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
        # OpenCV's HSV color conversion results in an image in the range of H: [0, 179], S: [0, 255], V: [0, 255]
        c1 = c1.astype(np.float32)[:, :, 0] / 179 * 360
        c2 = c2.astype(np.float32)[:, :, 0] / 179 * 360
        a = np.abs(c2 - c1)
        b = np.abs(np.full_like(c1, 360) - np.abs(c2 - c1))
        return np.minimum(a, b)

    color_distance_error = np.dstack([diff_h(mapping_color_hsv, np.array(c)[np.newaxis, np.newaxis, :]) for c in palette_hsv])
    color_distance_error[bg_mask, :] = 0

    similarity = 1 - (color_distance_error / 180.0)
    ratio = similarity / np.sum(similarity, axis=2)[:, :, np.newaxis]

    # if args.debug:
    #     for i in range(palette.shape[0]):
    #         cv2.imwrite(str(DIR_DEBUG / f"color_similarity_{color_names[i].strip()}.png"), (similarity[:, :, i] * 255).astype(np.uint8))

    # Low-value suppression:
    # if any pixel's color value is < MIN_RATIO_THRESHOLD, set it to 0
    if MIN_RATIO_THRESHOLD is not None:
        mask = ratio < MIN_RATIO_THRESHOLD
        ratio[mask] = 0
        ratio_diff = 1 - np.sum(ratio, axis=2)
        ratio_count_non_zero_colors = np.sum(ratio > 0, axis=2)
        ratio = ratio + (ratio_diff / ratio_count_non_zero_colors)[:, :, np.newaxis]
        ratio[mask] = 0

        logger.debug(f"low-value suppression affected pixels: {np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1]) * 100:5.2f}%")

    weighted_colors = np.full([color_distance_error.shape[0], color_distance_error.shape[1], palette.shape[0], 3], palette)
    weighted_colors = weighted_colors * ratio[:, :, :, np.newaxis]
    mapping_palette_avg_rgb = np.sum(weighted_colors, axis=2).astype(np.uint8)

    mapping_palette_avg_hsv = cv2.cvtColor(mapping_palette_avg_rgb, cv2.COLOR_RGB2HSV)

    # diff_v is the amount [0, 255] of how much brighter a pixel (if colored with a color mixed from the palette) is than it should be,
    # i.e. it is the "too bright" error value.
    # The error of "too dark" is not relevant, since making a pixel brighter than the mixed color
    # is not possible (assuming bright colors on dark surface)
    diff_v = np.clip(mapping_palette_avg_hsv[:, :, 2].astype(float) - mapping_color_hsv[:, :, 2], 0, 255).astype(np.uint8)

    # the image drawn with palette colors including V (only darker, not brighter)
    mapping_palette_preview_hsv = mapping_palette_avg_hsv.copy().astype(int)
    mapping_palette_preview_hsv[:, :, 2] = np.clip(mapping_palette_preview_hsv[:, :, 2] - diff_v, 0, 255)

    error_h_1 = np.abs(mapping_color_hsv[:, :, 0].astype(float) - mapping_palette_avg_hsv[:, :, 0].astype(float))
    error_h_2 = np.full(mapping_color_hsv.shape[0:2], 180) - error_h_1
    error_h = np.min(np.dstack([error_h_1, error_h_2]), axis=2) / 90

    error_s = np.abs(mapping_color_hsv[:, :, 1].astype(float) - mapping_palette_avg_hsv[:, :, 1].astype(float)) / 255
    error_v = np.abs(mapping_color_hsv[:, :, 2].astype(float) - mapping_palette_avg_hsv[:, :, 2].astype(float)) / 255

    return error_h, error_s, error_v, mapping_palette_preview_hsv


def main() -> None:
    parser = argparse.ArgumentParser()
    # parser.add_argument("image", type=Path, help="RGB image (TIFF)")
    # parser.add_argument("background", type=Path, help="Grayscale background mapping (PNG)")
    # parser.add_argument("--num-colors", type=int, default=3, help="Num Colors")
    parser.add_argument("--config", type=Path, help="Configuration file (TOML)")
    parser.add_argument("--debug", action="store_true", default=False, help="Write debug output")
    args = parser.parse_args()

    if args.debug:
        os.makedirs(DIR_DEBUG, exist_ok=True)

    image_path = IMAGE_PATH
    background_path = BACKGROUND_PATH
    num_colors = NUM_COLORS

    image_color = cv2.imread(str(image_path))
    mapping_background = cv2.imread(str(background_path), cv2.IMREAD_GRAYSCALE)

    if RESIZE_SIZE is not None:
        image_color = cv2.resize(image_color, RESIZE_SIZE)
        mapping_background = cv2.resize(mapping_background, RESIZE_SIZE)

    bg_mask = mapping_background > 0

    all_colors = []
    with open(Path("inks.json"), "rb") as f:
        inks = json.load(f)

        for key, value in inks.items():
            all_colors.append([key, value["on_black"]])

    combinations = list(itertools.combinations(all_colors, num_colors))

    for i_c, combination in enumerate(combinations):
        color_names = [c[0].replace(" ", "-") for c in combination]
        colors = [c[1] for c in combination]

        error_h, error_s, error_v, mapping_palette_preview_hsv = calculate(image_color, colors, bg_mask=bg_mask)

        fg_mask = ~bg_mask
        total_error = np.mean(np.dstack([error_h, error_s, error_v])[fg_mask])

        # ignore S and V for now

        combined_error = np.mean(error_h[fg_mask])  # + np.mean(error_s[fg_mask])

        prefix = f"{i_c}_{'_'.join(color_names)}_"
        suffix = f"_{total_error:.4f}"

        if combined_error < ERROR_THRESHOLD:
            debug_mapping = cv2.cvtColor(mapping_palette_preview_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            cv2.imwrite(str(DIR_DEBUG / (prefix + "palette_mapping_hsv_preview" + suffix + ".png")), debug_mapping)
            print("EXPORT")

        msg = "{:3d} | colors: {:<60s} ".format(i_c, " | ".join(color_names))
        msg += f""
        msg += f"H: {np.mean(error_h[fg_mask]):.4f} | "
        msg += f"S: {np.mean(error_s[fg_mask]):.4f} | "
        msg += f"V: {np.mean(error_v[fg_mask]):.4f} | "
        msg += f"HSV: {total_error:.4f}"
        print(msg)

        # if args.debug:
        #     cv2.imwrite(str(DIR_DEBUG / (prefix + "mapping_color" + suffix + ".png")), image_color)
        #
        #     debug_mapping = cv2.cvtColor(mapping_palette_avg_hsv, cv2.COLOR_HSV2BGR)
        #     debug_mapping[bg_mask, :] = 0
        #     cv2.imwrite(str(DIR_DEBUG / (prefix + "palette_mapping_hsv_avg" + suffix + ".png")), debug_mapping)
        #
        #     debug_mapping = diff_v.copy()
        #     debug_mapping[bg_mask] = 0
        #     cv2.imwrite(str(DIR_DEBUG / (prefix + "palette_mapping_hsv_diff_v" + suffix + ".png")), debug_mapping)
        #
        #     for c_i in range(ratio.shape[2]):
        #         cv2.imwrite(str(DIR_DEBUG / f"palette_ratio_{c_i}" + suffix + ".png"), (ratio[:, :, c_i] * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
