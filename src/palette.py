import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000, deltaE_cie76, deltaE_ciede94

RESIZE_SIZE = None  # (500, 500)
DIR_DEBUG = Path("debug")

MIN_RATIO_THRESHOLD = 0.15

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, default="image.tif", help="RGB image (TIFF)")
    parser.add_argument("--palette-mixture", type=Path, default="palette_mixture.npy", help="Output filename for palette mixture ratios [NPY]")
    parser.add_argument(
        "--palette-brightness-difference",
        type=Path,
        default="mapping_brightness_difference.png",
        help="Output filename for HSV value difference mapping [PNG]",
    )
    parser.add_argument("--color-model", default="hsv", choices=["hsv", "lab"], help="Choice of color model for ink to image comparisons")
    parser.add_argument("--palette-color", action="append", type=float, nargs=3, help="Palette color item [R, G, B], append once per color")
    parser.add_argument("--debug", action="store_true", default=False, help="Write debug output")
    args = parser.parse_args()

    mapping_color = cv2.imread(str(args.image))

    # inks = {}
    # with open(Path("inks.json"), "rb") as f:
    #     inks = json.load(f)

    if RESIZE_SIZE is not None:
        mapping_color = cv2.resize(mapping_color, RESIZE_SIZE)

    if args.debug:  # TODO: remove hardcoded paths
        mapping_distance = cv2.imread("build/mapping_color.png", cv2.IMREAD_GRAYSCALE)
        mapping_flat = cv2.imread("build/mapping_flat.png", cv2.IMREAD_GRAYSCALE)
        if RESIZE_SIZE is not None:
            mapping_distance = cv2.resize(mapping_distance, RESIZE_SIZE)
            mapping_flat = cv2.resize(mapping_flat, RESIZE_SIZE)

        cv2.imwrite(str(DIR_DEBUG / "palette_mapping_color.png"), mapping_color)
        cv2.imwrite(str(DIR_DEBUG / "palette_mapping_distance.png"), mapping_distance)

    # palette = np.array([
    #     [169, 103, 74],
    #     [50, 65, 92],
    #     # [0, 0, 0],
    #     # [255, 255, 255]
    # ], dtype=np.uint8)

    # palette = np.array([
    #     [220, 0, 0],
    #     [0, 0, 220],
    #     # [0, 0, 0],
    #     # [255, 255, 255]
    # ], dtype=np.uint8)

    # palette = np.array([
    #     [255, 255, 255],
    # ], dtype=np.uint8)

    # palette = np.array(
    #     [
    #         inks["naphtol orange"]["on_black"],
    #         # inks["ruby red"]["on_black"],
    #         inks["dunkelblau"]["on_black"],
    #         # inks["phthaloblau"]["on_black"],
    #         # inks["phthalotürkis"]["on_black"],
    #         # inks["kobaltblau"]["on_black"],
    #         # inks["pale grey"]["on_black"],
    #         # [0, 0, 0],
    #     ],
    #     dtype=np.uint8,
    # )

    palette = np.array(args.palette_color, dtype=int)
    palette = np.delete(palette, np.where(np.min(palette, axis=1) < 0), axis=0)  # remove invalid palette colors
    palette = palette.astype(np.uint8)

    # HSV H-ONLY

    palette_hsv = cv2.cvtColor(palette[:, np.newaxis, :], cv2.COLOR_RGB2HSV)[:, 0, :]
    mapping_color_hsv = cv2.cvtColor(mapping_color, cv2.COLOR_BGR2HSV)

    def diff_h(c1, c2):
        # OpenCV's HSV color conversion results in an image in the range of H: [0, 179], S: [0, 255], V: [0, 255]
        c1 = c1.astype(np.float32)[:, :, 0] / 179 * 360
        c2 = c2.astype(np.float32)[:, :, 0] / 179 * 360
        a = np.abs(c2 - c1)
        b = np.abs(np.full_like(c1, 360) - np.abs(c2 - c1))
        return np.minimum(a, b)

    color_distance_error = np.dstack([diff_h(mapping_color_hsv, np.array(c)[np.newaxis, np.newaxis, :]) for c in palette_hsv])

    # color_distance_error[:, :, 0] = [180]
    # cv2.imwrite(
    #     str(DIR_DEBUG / "palette_color_distance_error.png"),
    #     np.dstack([
    #         (1 - (color_distance_error / 180)) * 255,
    #         np.full([
    #             color_distance_error.shape[0],
    #             color_distance_error.shape[1],
    #             1
    #         ], 0)
    #     ]).astype(np.uint8)
    # )
    # exit()

    similarity = 1 - (color_distance_error / 180.0)
    ratio = similarity / np.sum(similarity, axis=2)[:, :, np.newaxis]

    # Low-value suppression:
    # if any pixel's color value is < MIN_RATIO_THRESHOLD, set it to 0
    mask = ratio < MIN_RATIO_THRESHOLD
    ratio[mask] = 0
    ratio_diff = 1 - np.sum(ratio, axis=2)
    ratio_count_non_zero_colors = np.sum(ratio > 0, axis=2)
    ratio = ratio + (ratio_diff / ratio_count_non_zero_colors)[:, :, np.newaxis]
    ratio[mask] = 0

    weighted_colors = np.full([color_distance_error.shape[0], color_distance_error.shape[1], palette.shape[0], 3], palette, dtype=np.float32)
    weighted_colors = weighted_colors * ratio[:, :, :, np.newaxis]
    mapping_palette_avg_rgb = np.sum(weighted_colors, axis=2).astype(np.uint8)

    mapping_palette_avg_hsv = cv2.cvtColor(mapping_palette_avg_rgb, cv2.COLOR_RGB2HSV)

    # diff_v is the amount [0, 255] of how much brighter a pixel (if colored with a color mixed from the palette) is than it should be,
    # i.e. it is the "too bright" error value.
    # The error of "too dark" is not relevant, since making a pixel brighter than the mixed color
    # is not possible (assuming bright colors on dark surface)
    diff_v = np.clip(mapping_palette_avg_hsv[:, :, 2].astype(np.float32) - mapping_color_hsv[:, :, 2], 0, 255).astype(np.uint8)

    # mapping_palette_avg_hsv[:, :, 2] -= diff_v

    if args.debug:
        cv2.imwrite(str(DIR_DEBUG / "palette_mapping_hsv_avg.png"), cv2.cvtColor(mapping_palette_avg_hsv, cv2.COLOR_HSV2BGR))
        cv2.imwrite(str(DIR_DEBUG / "palette_mapping_hsv_diff_v.png"), diff_v)

        for c_i in range(ratio.shape[2]):
            cv2.imwrite(str(DIR_DEBUG / f"palette_ratio_{c_i}.png"), (ratio[:, :, c_i] * 255).astype(np.uint8))

    if args.color_model == "hsv":
        np.save(args.palette_mixture, ratio)
        cv2.imwrite(args.palette_brightness_difference, ~diff_v)

        # diff_v_normalized = ((diff_v - np.min(diff_v)).astype(np.float32) / (np.max(diff_v) - np.min(diff_v)) * 255).astype(np.uint8)
        # cv2.imwrite(args.palette_brightness_difference, ~diff_v_normalized)

    exit()

    # HSV CLOSEST

    mapping_palette_closest = palette[np.argmin(color_distance_error, axis=2)]

    if args.debug:
        mapping_palette_closest[mapping_flat > 0] = [0, 0, 0]
        cv2.imwrite(str(DIR_DEBUG / "palette_mapping_hsv_closest.png"), cv2.cvtColor(mapping_palette_closest, cv2.COLOR_RGB2BGR))

    # LAB AVG

    palette_lab = [rgb2lab(np.array(c) / 255.0) for c in palette.tolist()]
    mapping_color_lab = rgb2lab(cv2.cvtColor(mapping_color, cv2.COLOR_BGR2RGB).astype(float) / 255.0)
    color_distance_error = np.dstack(
        # [deltaE_cie76(mapping_color_lab, c[np.newaxis, np.newaxis, :]) for c in palette_lab]
        [deltaE_ciede2000(mapping_color_lab, c[np.newaxis, np.newaxis, :]) for c in palette_lab]
        # [deltaE_ciede94(mapping_color_lab, c[np.newaxis, np.newaxis, :]) for c in palette_lab]
    )

    similarity = np.abs(1 - np.clip(color_distance_error, 0.1, 99.9) / 100.0)

    # increase chance if a candidate is very similar to a palette color
    # example: similarity color_1: 1.0 color_2: 0.5, should result in a very high chance for 1 and a low chance for 2,
    # unskewed this would be 2/3 vs. 1/3. Using exponentiation we can increase this gap.
    similarity_skewed = np.power(similarity, 4)

    ratio = similarity_skewed / np.sum(similarity_skewed, axis=2)[:, :, np.newaxis]
    weighted_colors = np.full([color_distance_error.shape[0], color_distance_error.shape[1], palette.shape[0], 3], palette, dtype=np.float32)
    weighted_colors = weighted_colors * ratio[:, :, :, np.newaxis]
    mapping_palette_avg_rgb = np.sum(weighted_colors, axis=2).astype(np.uint8)

    # mapping_palette_avg_rgb[mapping_flat > 0] = [0, 0, 0]

    mapping_palette_avg_hsv = cv2.cvtColor(mapping_palette_avg_rgb, cv2.COLOR_RGB2HSV)
    diff_v = np.clip(mapping_palette_avg_hsv[:, :, 2].astype(np.float32) - mapping_color_hsv[:, :, 2], 0, 255).astype(np.uint8)

    # mapping_palette_avg_hsv[:, :, 2] -= (diff_v / 2.0).astype(np.uint8)
    mapping_palette_avg_hsv[:, :, 2] -= diff_v

    if args.debug:
        cv2.imwrite(str(DIR_DEBUG / "palette_mapping_lab_avg.png"), cv2.cvtColor(mapping_palette_avg_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(DIR_DEBUG / "palette_mapping_lab_avg_with_diff.png"), cv2.cvtColor(mapping_palette_avg_hsv, cv2.COLOR_HSV2BGR))
        cv2.imwrite(str(DIR_DEBUG / "palette_mapping_lab_diff_v.png"), diff_v)

    if args.color_model == "lab":
        np.save(args.palette_mixture, ratio)
        cv2.imwrite(args.palette_brightness_difference, diff_v)

    # LAB CLOSEST

    mapping_palette_closest = palette[np.argmin(color_distance_error, axis=2)]

    if args.debug:
        mapping_palette_closest[mapping_flat > 0] = [0, 0, 0]
        cv2.imwrite(str(DIR_DEBUG / "palette_mapping_lab_closest.png"), cv2.cvtColor(mapping_palette_closest, cv2.COLOR_RGB2BGR))

    # HSV LAB AVG

    weighted_colors = np.full([color_distance_error.shape[0], color_distance_error.shape[1], palette.shape[0], 3], palette, dtype=np.float32)
    weighted_colors = weighted_colors * ratio[:, :, :, np.newaxis]
    mapping_palette_avg_rgb = np.sum(weighted_colors, axis=2).astype(np.uint8)

    mapping_palette_avg_comb = cv2.cvtColor(mapping_palette_avg_rgb, cv2.COLOR_RGB2HSV)
    diff_v = np.clip(mapping_palette_avg_comb[:, :, 2].astype(np.float32) - mapping_color_hsv[:, :, 2], 0, 255).astype(np.uint8)

    # mapping_palette_avg_comb[:, :, 2] -= diff_v

    if args.debug:
        cv2.imwrite(str(DIR_DEBUG / "palette_mapping_comb_avg.png"), cv2.cvtColor(mapping_palette_avg_comb, cv2.COLOR_HSV2BGR))

    # --------------------------------------------------

    # mapping_palette_avg_lab = rgb2lab(cv2.cvtColor(mapping_palette_avg, cv2.COLOR_BGR2RGB).astype(float) / 255.0)
    # color_distance_error = deltaE_ciede2000(mapping_palette_avg_lab, mapping_color_lab)
    # print(f"total error: {np.mean(color_distance_error):5.3f}")
