from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


class LinearCalibrator:
    def __init__(self, regularization: float = 1e-3) -> None:
        self.regularization = regularization
        self._features: List[np.ndarray] = []
        self._targets: List[np.ndarray] = []
        self._weights: Optional[np.ndarray] = None

    @property
    def is_ready(self) -> bool:
        return self._weights is not None

    def add_sample(self, features: np.ndarray, target_xy: np.ndarray) -> None:
        self._features.append(features.astype(np.float32))
        self._targets.append(target_xy.astype(np.float32))

    def fit(self) -> bool:
        if len(self._features) < 6:
            return False
        x = np.stack(self._features, axis=0)  # [N, F]
        y = np.stack(self._targets, axis=0)  # [N, 2]
        ones = np.ones((x.shape[0], 1), dtype=np.float32)
        x_aug = np.concatenate([x, ones], axis=1)  # [N, F+1]

        # Ridge regression closed-form: W = (X^T X + lI)^-1 X^T Y
        xtx = x_aug.T @ x_aug
        reg = np.eye(xtx.shape[0], dtype=np.float32) * self.regularization
        reg[-1, -1] = 0.0  # no regularization on bias
        w = np.linalg.solve(xtx + reg, x_aug.T @ y)  # [F+1,2]
        self._weights = w.astype(np.float32)
        return True

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("Calibrator is not fitted")
        x_aug = np.concatenate([features.astype(np.float32), np.array([1.0], dtype=np.float32)], axis=0)
        y = x_aug @ self._weights
        return np.clip(y, 0.0, 1.0).astype(np.float32)

    def reset(self) -> None:
        self._features.clear()
        self._targets.clear()
        self._weights = None


@dataclass
class CalibrationState:
    active: bool
    index: int
    total: int
    just_finished: bool


class CalibrationRoutine:
    def __init__(
        self,
        calibrator: LinearCalibrator,
        points: Optional[Sequence[Tuple[float, float]]] = None,
        settle_seconds: float = 0.35,
        samples_per_point: int = 16,
    ) -> None:
        self.calibrator = calibrator
        if points is None:
            points = (
                (0.15, 0.15),
                (0.50, 0.15),
                (0.85, 0.15),
                (0.15, 0.50),
                (0.50, 0.50),
                (0.85, 0.50),
                (0.15, 0.85),
                (0.50, 0.85),
                (0.85, 0.85),
            )
        self.points = [np.array(p, dtype=np.float32) for p in points]
        self.settle_seconds = float(settle_seconds)
        self.samples_per_point = int(samples_per_point)
        self._active = False
        self._index = 0
        self._start = 0.0
        self._buffer: List[np.ndarray] = []
        self._just_finished = False

    @property
    def state(self) -> CalibrationState:
        return CalibrationState(
            active=self._active,
            index=self._index,
            total=len(self.points),
            just_finished=self._just_finished,
        )

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> None:
        self.calibrator.reset()
        self._active = True
        self._index = 0
        self._start = time.perf_counter()
        self._buffer = []
        self._just_finished = False

    def _advance(self) -> None:
        self._index += 1
        self._start = time.perf_counter()
        self._buffer = []
        if self._index >= len(self.points):
            self._active = False
            self._just_finished = self.calibrator.fit()

    def update(self, raw_features: Optional[np.ndarray]) -> None:
        if not self._active or raw_features is None:
            return

        elapsed = time.perf_counter() - self._start
        if elapsed < self.settle_seconds:
            return

        self._buffer.append(raw_features.copy())
        if len(self._buffer) >= self.samples_per_point:
            features = np.mean(np.stack(self._buffer, axis=0), axis=0)
            target = self.points[self._index]
            self.calibrator.add_sample(features, target)
            self._advance()

    def consume_finished_flag(self) -> bool:
        if self._just_finished:
            self._just_finished = False
            return True
        return False

    def draw(self, frame: np.ndarray) -> None:
        if not self._active:
            return
        height, width = frame.shape[:2]
        target = self.points[self._index]
        px = int(target[0] * (width - 1))
        py = int(target[1] * (height - 1))

        cv2.circle(frame, (px, py), 16, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 4, (0, 200, 255), -1, cv2.LINE_AA)
        msg = f"Calibracao {self._index + 1}/{len(self.points)}: olhe para o ponto"
        cv2.putText(frame, msg, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(frame, msg, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
