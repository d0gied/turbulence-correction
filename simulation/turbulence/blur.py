from __future__ import annotations

import cv2
import numpy as np


def create_and_blend_blurred_versions(
    image: np.ndarray,
    noise_map: np.ndarray,
    max_sigma: float = 11.0,
    num_levels: int = 11,
) -> np.ndarray:
    blurred_images = [image]
    for i in range(1, num_levels):
        sigma = i**0.5 * max_sigma / num_levels
        blurred_image = cv2.GaussianBlur(blurred_images[-1], (0, 0), sigma)
        blurred_images.append(blurred_image)

    normalized_noise_map = (noise_map - noise_map.min()) / (
        noise_map.max() - noise_map.min()
    )
    indices = (normalized_noise_map * (len(blurred_images) - 1)).astype(int)

    stacked_images = np.stack(blurred_images, axis=-1)

    x_coords, y_coords = np.meshgrid(
        np.arange(noise_map.shape[1]), np.arange(noise_map.shape[0])
    )

    if image.ndim == 2:
        return stacked_images[y_coords, x_coords, indices[y_coords, x_coords]]

    final_image = stacked_images[y_coords, x_coords, :, indices[y_coords, x_coords]]
    return final_image


def create_and_blend_blurred_versions_fast(
    image: np.ndarray,
    noise_map: np.ndarray,
    max_sigma: float = 11.0,
    num_levels: int = 11,
) -> np.ndarray:
    # 1) индексы уровней
    nmin, nmax = float(noise_map.min()), float(noise_map.max())
    denom = (nmax - nmin) if (nmax > nmin) else 1.0
    normalized = (noise_map - nmin) / denom
    idx = (normalized * (num_levels - 1)).astype(np.int32)

    out = np.empty_like(image, dtype=np.float32)

    cur = image.astype(np.float32)
    for i in range(num_levels):
        if i > 0:
            sigma = (i**0.5) * max_sigma / num_levels
            cur = cv2.GaussianBlur(cur, (0, 0), sigma)  # type: ignore

        m = idx == i
        if image.ndim == 2:
            out[m] = cur[m]
        else:
            out[m, :] = cur[m, :]

    return out
