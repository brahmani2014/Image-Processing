"""Tutorial 7: Canny edge detection and five thresholding modes."""

from __future__ import annotations

import argparse

import cv2
import matplotlib.pyplot as plt

from image_utils import read_image, show_grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to an input image, such as Balloon.jpg")
    parser.add_argument("--threshold", type=int, default=140)
    parser.add_argument("--canny-low", type=int, default=100)
    parser.add_argument("--canny-high", type=int, default=200)
    args = parser.parse_args()

    gray = read_image(args.image, grayscale=True)
    edges = cv2.Canny(gray, args.canny_low, args.canny_high)

    threshold_modes = [
        ("Binary", cv2.THRESH_BINARY),
        ("Binary inverse", cv2.THRESH_BINARY_INV),
        ("Truncate", cv2.THRESH_TRUNC),
        ("To zero", cv2.THRESH_TOZERO),
        ("To zero inverse", cv2.THRESH_TOZERO_INV),
    ]
    results = [("Original grayscale", gray, "gray"), ("Canny edges", edges, "gray")]
    for title, mode in threshold_modes:
        _, thresholded = cv2.threshold(gray, args.threshold, 255, mode)
        results.append((title, thresholded, "gray"))

    show_grid(results, columns=3, figure_size=(14, 10))
    plt.hist(gray.ravel(), bins=256, range=(0, 256))
    plt.axvline(args.threshold, color="red", linestyle="--", label=f"Threshold {args.threshold}")
    plt.title("Grayscale intensity histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Pixel count")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
