from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from .config import RuntimeConfig


class LowPassFilter:
    def __init__(self) -> None:
        self._initialized = False
        self._value = 0.0

    def apply(self, value: float, alpha: float) -> float:
        if not self._initialized:
            self._initialized = True
            self._value = value
            return value
        self._value = alpha * value + (1.0 - alpha) * self._value
        return self._value

    @property
    def value(self) -> float:
        return self._value


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x = LowPassFilter()
        self._dx = LowPassFilter()

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def apply(self, value: float, dt: float) -> float:
        prev_x = self._x.value if self._x._initialized else value
        derivative = (value - prev_x) / max(dt, 1e-6)
        edx = self._dx.apply(derivative, self._alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x.apply(value, self._alpha(cutoff, dt))


class GazeStabilizer:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self._filter_x = OneEuroFilter(
            min_cutoff=config.one_euro_min_cutoff,
            beta=config.one_euro_beta,
            d_cutoff=config.one_euro_d_cutoff,
        )
        self._filter_y = OneEuroFilter(
            min_cutoff=config.one_euro_min_cutoff,
            beta=config.one_euro_beta,
            d_cutoff=config.one_euro_d_cutoff,
        )
        self._kalman = cv2.KalmanFilter(4, 2)
        self._kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self._kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]],
            dtype=np.float32,
        )
        self._kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
        self._kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2e-2
        self._kalman.errorCovPost = np.eye(4, dtype=np.float32)
        self._kalman.statePost = np.array([[0.5], [0.5], [0.0], [0.0]], dtype=np.float32)
        self._last_valid: Optional[np.ndarray] = None

    def update(self, measurement: Optional[np.ndarray], dt: float, confidence: float) -> np.ndarray:
        dt = max(dt, 1e-4)
        self._kalman.transitionMatrix[0, 2] = dt
        self._kalman.transitionMatrix[1, 3] = dt
        prediction = self._kalman.predict()
        predicted_xy = prediction[:2, 0].astype(np.float32)

        if measurement is not None:
            measurement = np.asarray(measurement, dtype=np.float32)
            if self._last_valid is not None:
                jump = float(np.linalg.norm(measurement - self._last_valid))
                if jump > self.config.outlier_threshold and confidence < 0.7:
                    measurement = self._last_valid
            corrected = self._kalman.correct(measurement.reshape(2, 1))
            raw_xy = corrected[:2, 0].astype(np.float32)
            self._last_valid = raw_xy
        else:
            raw_xy = predicted_xy

        x = self._filter_x.apply(float(raw_xy[0]), dt)
        y = self._filter_y.apply(float(raw_xy[1]), dt)
        stabilized = np.array([x, y], dtype=np.float32)
        return np.clip(stabilized, 0.0, 1.0)
