"""Shared helpers for the Image Processing tutorial scripts."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_image(path: str | Path, grayscale: bool = False) -> np.ndarray:
    """Read an image and raise a useful error when OpenCV cannot load it."""
    image_path = Path(path).expanduser()
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(image_path), flag)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path.resolve()}")
    return image


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR image to Matplotlib's RGB channel order."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def show_grid(
    images: list[tuple[str, np.ndarray, str | None]],
    columns: int = 2,
    figure_size: tuple[float, float] = (12, 8),
) -> None:
    """Display titled images in a compact Matplotlib grid."""
    rows = (len(images) + columns - 1) // columns
    _, axes = plt.subplots(rows, columns, figsize=figure_size, squeeze=False)
    for axis, (title, image, cmap) in zip(axes.flat, images):
        axis.imshow(image, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    for axis in axes.flat[len(images) :]:
        axis.axis("off")
    plt.tight_layout()
    plt.show()
