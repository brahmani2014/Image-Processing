"""Tutorial 5: salt-and-pepper, Gaussian, and Rayleigh noise."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from image_utils import read_image, show_grid, to_rgb


def salt_and_pepper(image: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    noisy = image.copy()
    affected = rng.random(image.shape[:2])
    noisy[affected < amount / 2] = 0.0
    noisy[(affected >= amount / 2) & (affected < amount)] = 1.0
    return noisy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to an input image, such as Rose.jpg")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--salt-pepper", type=float, default=0.30, help="Fraction of affected pixels")
    parser.add_argument("--gaussian-sigma", type=float, default=0.30)
    parser.add_argument("--rayleigh-scale", type=float, default=0.20)
    args = parser.parse_args()

    if not 0 <= args.salt_pepper <= 1:
        parser.error("--salt-pepper must be between 0 and 1")

    rgb = to_rgb(read_image(args.image)).astype(np.float32) / 255.0
    rng = np.random.default_rng(args.seed)

    sp_image = salt_and_pepper(rgb, args.salt_pepper, rng)
    gaussian_noise = rng.normal(0.0, args.gaussian_sigma, rgb.shape)
    gaussian_image = np.clip(rgb + gaussian_noise, 0.0, 1.0)
    rayleigh_noise = rng.rayleigh(args.rayleigh_scale, rgb.shape)
    rayleigh_image = np.clip(rgb + rayleigh_noise, 0.0, 1.0)

    show_grid(
        [
            ("Original", rgb, None),
            ("Salt-and-pepper noise", sp_image, None),
            ("Gaussian noise", gaussian_image, None),
            ("Rayleigh noise", rayleigh_image, None),
        ],
        columns=2,
    )

    _, axes = plt.subplots(1, 3, figsize=(15, 4))
    distributions = [
        (sp_image, "Salt-and-pepper image intensities"),
        (gaussian_noise, "Gaussian noise values"),
        (rayleigh_noise, "Rayleigh noise values"),
    ]
    for axis, (values, title) in zip(axes, distributions):
        axis.hist(values.ravel(), bins=100)
        axis.set_title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
