"""Tutorial 4: Fourier spectrum and spatial averaging filter."""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from image_utils import read_image, show_grid, to_rgb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to an input image, such as Snow.JPG")
    parser.add_argument("--kernel-size", type=int, default=3, help="Odd averaging-kernel size")
    args = parser.parse_args()

    if args.kernel_size < 1 or args.kernel_size % 2 == 0:
        parser.error("--kernel-size must be a positive odd integer")

    bgr = read_image(args.image)
    rgb = to_rgb(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    frequency = np.fft.fft2(gray)
    centered = np.fft.fftshift(frequency)
    spectrum = 20.0 * np.log1p(np.abs(centered))
    averaged = cv2.blur(rgb, (args.kernel_size, args.kernel_size))

    show_grid(
        [
            ("Original", rgb, None),
            ("Grayscale", gray, "gray"),
            ("Centered Fourier magnitude", spectrum, "gray"),
            (f"{args.kernel_size}x{args.kernel_size} averaging filter", averaged, None),
        ],
        columns=2,
    )


if __name__ == "__main__":
    main()
