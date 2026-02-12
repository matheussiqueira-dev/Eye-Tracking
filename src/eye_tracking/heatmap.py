from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class HeatmapConfig:
    height: int
    width: int
    sigma: float
    decay: float


class HeatmapAccumulator:
    def __init__(self, config: HeatmapConfig) -> None:
        self.config = config
        self.map = np.zeros((config.height, config.width), dtype=np.float32)
        self._radius = max(2, int(round(3.0 * config.sigma)))
        self._kernel = self._gaussian_kernel(self._radius, config.sigma)

    @staticmethod
    def _gaussian_kernel(radius: int, sigma: float) -> np.ndarray:
        coords = np.arange(-radius, radius + 1, dtype=np.float32)
        xx, yy = np.meshgrid(coords, coords)
        kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma * sigma))
        kernel /= max(kernel.max(), 1e-6)
        return kernel.astype(np.float32)

    def reset(self) -> None:
        self.map.fill(0.0)

    def update(self, point: Optional[np.ndarray], confidence: float, dt: float) -> None:
        dt = max(dt, 1e-4)
        self.map *= np.exp(-self.config.decay * dt)
        if point is None or confidence <= 0.0:
            return

        x = int(round(float(point[0]) * (self.config.width - 1)))
        y = int(round(float(point[1]) * (self.config.height - 1)))
        if x < 0 or y < 0 or x >= self.config.width or y >= self.config.height:
            return

        r = self._radius
        x0 = max(0, x - r)
        x1 = min(self.config.width, x + r + 1)
        y0 = max(0, y - r)
        y1 = min(self.config.height, y + r + 1)

        kx0 = x0 - (x - r)
        kx1 = kx0 + (x1 - x0)
        ky0 = y0 - (y - r)
        ky1 = ky0 + (y1 - y0)
        self.map[y0:y1, x0:x1] += self._kernel[ky0:ky1, kx0:kx1] * float(confidence)

    def overlay(self, frame: np.ndarray, alpha: float = 0.38) -> np.ndarray:
        max_value = float(self.map.max())
        if max_value <= 1e-5:
            return frame
        normalized = self.map / max_value
        normalized = np.power(normalized, 0.6, dtype=np.float32)
        heat = np.uint8(np.clip(normalized * 255.0, 0, 255))
        heat = cv2.resize(heat, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
        colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        return cv2.addWeighted(colored, alpha, frame, 1.0 - alpha, 0.0)
