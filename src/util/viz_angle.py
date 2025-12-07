import argparse
from pathlib import Path
import math

import numpy as np
import cv2

import matplotlib.pyplot as plt

RESIZE_FACTOR = 5
NUM_X = 150
NUM_Y = 150

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mapping_angle", type=Path, help="(PNG)")
    args = parser.parse_args()

    mapping_angle = cv2.imread(args.mapping_angle, flags=cv2.IMREAD_GRAYSCALE)
    overlay_image = cv2.imread("earth.png")
    overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)

    # mapping_angle = mapping_angle[400:600, 400:600]
    # mapping_angle[500:, 500:] = 0.25 * 255

    if RESIZE_FACTOR is not None:
        mapping_angle = cv2.resize(mapping_angle, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

    overlay_image = cv2.resize(overlay_image, (mapping_angle.shape[1], mapping_angle.shape[0]))

    height, width = mapping_angle.shape[0:2]
    #
    # u = cv2.resize(u, size)
    # v = cv2.resize(v, size)

    x_distance = width / (NUM_X)
    y_distance = height / (NUM_Y)

    x = (np.arange(NUM_X) * x_distance) + (x_distance / 2)
    y = (np.arange(NUM_Y) * y_distance) + (y_distance / 2)

    X, Y = np.meshgrid(x, y)

    X = X.astype(int)
    Y = Y.astype(int)

    u = np.full_like(X, 1.0)
    v = np.full_like(X, 1.0)

    u = np.cos(mapping_angle[X, Y] / 255 * math.tau)
    v = np.sin(mapping_angle[X, Y] / 255 * math.tau)

    plt.figure(figsize=(50, 50))
    # plt.imshow(mapping_angle, cmap="gray", origin="upper")
    plt.imshow(overlay_image, cmap="gray", origin="upper")
    plt.quiver(X, Y, u, v, color="red" , headwidth=2)

    plt.axis("off")
    plt.savefig("plot.png", bbox_inches='tight')
    # plt.show()