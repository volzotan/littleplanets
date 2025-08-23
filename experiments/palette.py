import json
from pathlib import Path

import cv2
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000, deltaE_cie76, deltaE_ciede94

RESIZE_SIZE = None # (200, 200)
DIR_DEBUG = Path("debug")

mapping_color = cv2.imread("build/mapping_color.png")
# mapping_distance = cv2.imread("build/mapping_distance.png", cv2.IMREAD_GRAYSCALE)
mapping_distance = cv2.imread("build/mapping_color.png", cv2.IMREAD_GRAYSCALE)
mapping_flat = cv2.imread("build/mapping_flat.png", cv2.IMREAD_GRAYSCALE)

inks = {}
with open(Path("inks.json"), "rb") as f:
    inks = json.load(f)

if RESIZE_SIZE is not None:
    mapping_color = cv2.resize(mapping_color, RESIZE_SIZE)
    mapping_distance = cv2.resize(mapping_distance, RESIZE_SIZE)
    mapping_flat = cv2.resize(mapping_flat, RESIZE_SIZE)

palette = np.array([
    [169, 103, 74],
    [50, 65, 92],
    [0, 0, 0],
    # [255, 255, 255]
], dtype=np.uint8)

palette = np.array([
    [255, 0, 0],
    [0, 0, 255],
    [0, 0, 0],
    # [255, 255, 255]
], dtype=np.uint8)

palette = np.array([
    inks["naphtol orange"]["on_black"],
    # inks["ruby red"]["on_black"],
    # inks["phthaloblau"]["on_black"],
    inks["dunkelblau"]["on_black"],
    # inks["phthalotürkis"]["on_black"],
    # inks["kobaltblau"]["on_black"],
    # inks["pale grey"]["on_black"],
    [0, 0, 0],
], dtype=np.uint8)


palette_labColor = [rgb2lab(np.array(c) / 255.0) for c in palette.tolist()]

mapping_color_lab = rgb2lab(cv2.cvtColor(mapping_color, cv2.COLOR_BGR2RGB).astype(float) / 255.0)

closest_palette_color = np.zeros([mapping_color_lab.shape[0], mapping_color_lab.shape[1]], dtype=int)
mapping_palette_avg = np.zeros_like(mapping_color)

color_distance_error = np.dstack(
    # [deltaE_cie76(mapping_color_lab, c[np.newaxis, np.newaxis, :]) for c in palette_labColor]
    [deltaE_ciede2000(mapping_color_lab, c[np.newaxis, np.newaxis, :]) for c in palette_labColor]
    # [deltaE_ciede94(mapping_color_lab, c[np.newaxis, np.newaxis, :]) for c in palette_labColor]
)

similarity = np.abs(1 - np.clip(color_distance_error, 0.1, 99.9) / 100.0)

# increase chance if a candidate is very similar to a palette color
# example: similarity color_1: 1.0 color_2: 0.5, should result in a very high chance for 1 and a low chance for 2,
# unskewed this would be 2/3 vs. 1/3. By exponentation we can increase this gap.
similarity_skewed = np.power(similarity, 5)

ratio = similarity_skewed / np.sum(similarity_skewed, axis=2)[:, :, np.newaxis]
weighted_colors = np.full([color_distance_error.shape[0], color_distance_error.shape[1], palette.shape[0], 3], palette, dtype=np.float32)
weighted_colors = weighted_colors * ratio[:, :, :, np.newaxis]

avg = np.sum(weighted_colors, axis=2).astype(np.uint8)
mapping_palette_avg = avg

mapping_palette_closest = palette[np.argmin(color_distance_error, axis=2)]

# mapping_palette_closest[mapping_flat > 0] = [0, 0, 0]
mapping_palette_avg[mapping_flat > 0] = [0, 0, 0]

cv2.imwrite(str(DIR_DEBUG / "palette_mapping_closest.png"), cv2.cvtColor(mapping_palette_closest, cv2.COLOR_RGB2BGR))
cv2.imwrite(str(DIR_DEBUG / "palette_mapping_avg.png"), cv2.cvtColor(mapping_palette_avg, cv2.COLOR_RGB2BGR))

cv2.imwrite(str(DIR_DEBUG / "palette_mapping_color.png"), mapping_color)
cv2.imwrite(str(DIR_DEBUG / "palette_mapping_distance.png"), mapping_distance)

mapping_palette_avg_lab = rgb2lab(cv2.cvtColor(mapping_palette_avg, cv2.COLOR_BGR2RGB).astype(float) / 255.0)
color_distance_error = deltaE_ciede2000(mapping_palette_avg_lab, mapping_color_lab)
print(f"total error: {np.mean(color_distance_error):5.3f}")