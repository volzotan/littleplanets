from pathlib import Path

import cv2
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000, deltaE_cie76, deltaE_ciede94

RESIZE_SIZE = (200, 200)
DIR_DEBUG = Path("debug")

mapping_color = cv2.imread("build/mapping_color.png")
# mapping_distance = cv2.imread("build/mapping_distance.png", cv2.IMREAD_GRAYSCALE)
mapping_distance = cv2.imread("build/mapping_color.png", cv2.IMREAD_GRAYSCALE)
mapping_flat = cv2.imread("build/mapping_flat.png", cv2.IMREAD_GRAYSCALE)

mapping_color = cv2.resize(mapping_color, RESIZE_SIZE)
mapping_distance = cv2.resize(mapping_distance, RESIZE_SIZE)
mapping_flat = cv2.resize(mapping_flat, RESIZE_SIZE)

palette = np.array([
    [169, 103, 74],
    [50, 65, 92],
    [0, 0, 0],
    # [255, 255, 255]
], dtype=np.uint8)

# palette = np.array([
#     [255, 0, 0],
#     [0, 0, 255],
#     [0, 0, 0],
#     [255, 255, 255]
# ], dtype=np.uint8)

palette_labColor = [rgb2lab(np.array(c) / 255.0) for c in palette.tolist()]

mapping_color_rgb = cv2.cvtColor(mapping_color, cv2.COLOR_BGR2RGB)
mapping_color_lab = rgb2lab(mapping_color_rgb.astype(float) / 255.0)

closest_palette_color = np.zeros([mapping_color_lab.shape[0], mapping_color_lab.shape[1]], dtype=int)
mapping_palette_avg = np.zeros_like(mapping_color)

for i in range(mapping_color_lab.shape[0]):
    for j in range(mapping_color_lab.shape[1]):
        # distance_error = np.array([deltaE_cie76(mapping_color_lab[i, j, :], palette_labColor[k]) for k in range(len(palette))])
        distance_error = np.array([deltaE_ciede2000(mapping_color_lab[i, j, :], palette_labColor[k]) for k in range(len(palette))])
        # distance_error = np.array([deltaE_ciede94(mapping_color_lab[i, j, :], palette_labColor[k]) for k in range(len(palette))])
        closest_palette_color[i, j] = np.argmin(distance_error)

        similarity = np.abs(1 - np.clip(distance_error, 0, 100) / 100.0)

        # increase chance if a candidate is very similar to a palette color
        # example: color_1: 1.0 color_2: 0.5, should result in a very high chance for 1 and a low chance for 2,
        # unskewed this would be 2/3 vs. 1/3. By exponentation we can increase this gap.
        similarity_skewed = np.power(similarity, 5)

        ratio = similarity_skewed / np.sum(similarity_skewed)
        weighted_colors = np.multiply(palette, ratio[:, np.newaxis])
        avg = np.sum(weighted_colors, axis=0)
        mapping_palette_avg[i, j, :] = avg

        if i == 50 and j == 50:
            mapping_palette_avg[i, j, :] = [255, 255, 255]

mapping_palette_closest = palette[closest_palette_color]

mapping_palette_closest[mapping_flat > 0] = [0, 0, 0]
mapping_palette_avg[mapping_flat > 0] = [0, 0, 0]

cv2.imwrite(str(DIR_DEBUG / "palette_mapping_closest.png"), cv2.cvtColor(mapping_palette_closest, cv2.COLOR_RGB2BGR))
cv2.imwrite(str(DIR_DEBUG / "palette_mapping_avg.png"), cv2.cvtColor(mapping_palette_avg, cv2.COLOR_RGB2BGR))

cv2.imwrite(str(DIR_DEBUG / "palette_mapping_color.png"), mapping_color)
cv2.imwrite(str(DIR_DEBUG / "palette_mapping_distance.png"), mapping_distance)