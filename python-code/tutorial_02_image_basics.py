"""Tutorial 2: pixels, brightness, grayscale, resizing, rotation, and interpolation."""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from image_utils import read_image, show_grid, to_rgb


def increase_brightness(image_bgr: np.ndarray, amount: int = 100) -> np.ndarray:
    """Increase HSV value with saturation instead of uint8 wraparound."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = cv2.add(hsv[..., 2], amount)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def rotate(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to an input image")
    parser.add_argument("--brightness", type=int, default=100, help="HSV value increase")
    parser.add_argument("--angle", type=float, default=120, help="Rotation angle in degrees")
    args = parser.parse_args()

    bgr = read_image(args.image)
    rgb = to_rgb(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bright_rgb = to_rgb(increase_brightness(bgr, args.brightness))

    print("Numeric image array:")
    print(rgb)

    half = cv2.resize(rgb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    zoom = cv2.resize(rgb, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    rotated = rotate(rgb, args.angle)
    flipped = cv2.flip(rgb, 0)

    show_grid(
        [
            ("Original", rgb, None),
            (f"Brightness +{args.brightness}", bright_rgb, None),
            ("Grayscale", gray, "gray"),
            ("Half size", half, None),
            ("2x zoom", zoom, None),
            (f"Rotated {args.angle:g} degrees", rotated, None),
            ("Vertical flip", flipped, None),
        ],
        columns=3,
        figure_size=(15, 10),
    )

    target_size = (1024, 1024)
    interpolation_results = [
        ("Nearest neighbor", cv2.resize(rgb, target_size, interpolation=cv2.INTER_NEAREST), None),
        ("Bilinear", cv2.resize(rgb, target_size, interpolation=cv2.INTER_LINEAR), None),
        ("Bicubic", cv2.resize(rgb, target_size, interpolation=cv2.INTER_CUBIC), None),
    ]
    show_grid(interpolation_results, columns=3, figure_size=(15, 5))


if __name__ == "__main__":
    main()
