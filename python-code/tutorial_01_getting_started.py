"""Tutorial 1: load and display an image with OpenCV and Matplotlib."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from image_utils import read_image, to_rgb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to an input image, such as tree.jpeg")
    args = parser.parse_args()

    image_bgr = read_image(args.image)
    image_rgb = to_rgb(image_bgr)

    print(f"Shape (height, width, channels): {image_rgb.shape}")
    print(f"Data type: {image_rgb.dtype}")
    print(f"Intensity range: {image_rgb.min()} to {image_rgb.max()}")

    plt.imshow(image_rgb)
    plt.title("Input image")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
