"""Tutorial 8: dilation, erosion, and closing with a square structuring element."""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from image_utils import read_image, show_grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to an input image, such as broken.tif")
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()

    if args.kernel_size < 1:
        parser.error("--kernel-size must be positive")
    if args.iterations < 1:
        parser.error("--iterations must be positive")

    gray = read_image(args.image, grayscale=True)
    kernel = np.ones((args.kernel_size, args.kernel_size), dtype=np.uint8)
    dilation = cv2.dilate(gray, kernel, iterations=args.iterations)
    erosion = cv2.erode(gray, kernel, iterations=args.iterations)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=args.iterations)

    show_grid(
        [
            ("Original", gray, "gray"),
            ("Dilation", dilation, "gray"),
            ("Erosion", erosion, "gray"),
            ("Closing (dilation then erosion)", closing, "gray"),
        ],
        columns=2,
    )


if __name__ == "__main__":
    main()
