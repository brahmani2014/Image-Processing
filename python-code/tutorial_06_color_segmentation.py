"""Tutorial 6: segment a color range with an HSV mask."""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from image_utils import read_image, show_grid, to_rgb


def hsv_triplet(value: str) -> tuple[int, int, int]:
    try:
        result = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use H,S,V integers separated by commas") from exc
    if len(result) != 3 or not (0 <= result[0] <= 179) or any(not 0 <= x <= 255 for x in result[1:]):
        raise argparse.ArgumentTypeError("H must be 0-179; S and V must be 0-255")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to an input image, such as Rose.jpeg")
    parser.add_argument("--lower", type=hsv_triplet, default=(0, 0, 200), help="Lower HSV as H,S,V")
    parser.add_argument("--upper", type=hsv_triplet, default=(145, 90, 255), help="Upper HSV as H,S,V")
    args = parser.parse_args()

    bgr = read_image(args.image)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(args.lower, dtype=np.uint8), np.array(args.upper, dtype=np.uint8))
    result_bgr = cv2.bitwise_and(bgr, bgr, mask=mask)

    show_grid(
        [
            ("Original", to_rgb(bgr), None),
            ("HSV mask", mask, "gray"),
            ("Segmented result", to_rgb(result_bgr), None),
        ],
        columns=3,
        figure_size=(15, 5),
    )


if __name__ == "__main__":
    main()
