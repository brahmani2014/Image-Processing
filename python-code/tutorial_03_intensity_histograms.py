"""Tutorial 3: log/gamma transforms, histograms, and histogram equalization."""

from __future__ import annotations

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from image_utils import read_image, show_grid, to_rgb


def log_transform(image: np.ndarray) -> np.ndarray:
    image_float = image.astype(np.float32)
    scale = 255.0 / np.log1p(255.0)
    return np.clip(scale * np.log1p(image_float), 0, 255).astype(np.uint8)


def gamma_transform(image: np.ndarray, gamma: float = 2.5) -> np.ndarray:
    normalized = image.astype(np.float32) / 255.0
    return np.clip(255.0 * normalized**gamma, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to an input image, such as flower.jpg")
    parser.add_argument("--gamma", type=float, default=2.5)
    args = parser.parse_args()

    bgr = read_image(args.image)
    rgb = to_rgb(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    log_rgb = log_transform(rgb)
    gamma_rgb = gamma_transform(rgb, args.gamma)
    equalized = cv2.equalizeHist(gray)

    show_grid(
        [
            ("Original", rgb, None),
            ("Log transform", log_rgb, None),
            (f"Gamma transform (gamma={args.gamma:g})", gamma_rgb, None),
            ("Grayscale", gray, "gray"),
            ("Histogram equalized", equalized, "gray"),
        ],
        columns=3,
        figure_size=(15, 9),
    )

    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    cdf = hist.cumsum()
    cdf_scaled = cdf * hist.max() / max(cdf.max(), 1)
    plt.plot(cdf_scaled, color="blue", label="CDF (scaled)")
    plt.hist(gray.ravel(), bins=256, range=(0, 256), color="red", alpha=0.55, label="Histogram")
    plt.xlim(0, 255)
    plt.legend()
    plt.title("Grayscale histogram and cumulative distribution")
    plt.show()

    for channel, color in enumerate(("blue", "green", "red")):
        channel_hist = cv2.calcHist([bgr], [channel], None, [256], [0, 256])
        plt.plot(channel_hist, color=color)
    plt.xlim(0, 255)
    plt.title("Color-channel histograms")
    plt.xlabel("Intensity")
    plt.ylabel("Pixel count")
    plt.show()


if __name__ == "__main__":
    main()
