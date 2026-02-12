from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

from .config import RuntimeConfig
from .landmarks import FaceObservation


@dataclass
class HeadPose:
    success: bool
    yaw: float
    pitch: float
    roll: float


@dataclass
class GazeEstimate:
    point: Optional[np.ndarray]
    confidence: float
    head_pose: HeadPose
    raw_features: Optional[np.ndarray]
    source: str


class HeadPoseEstimator:
    def __init__(self) -> None:
        # Canonical face model in millimeters (rough approximation).
        self._model_points = np.array(
            [
                (0.0, 0.0, 0.0),  # nose
                (0.0, -63.6, -12.5),  # chin
                (-43.3, 32.7, -26.0),  # right eye corner
                (43.3, 32.7, -26.0),  # left eye corner
                (-28.9, -28.9, -24.1),  # mouth left
                (28.9, -28.9, -24.1),  # mouth right
            ],
            dtype=np.float32,
        )

    def estimate(self, points: Dict[int, np.ndarray], width: int, height: int) -> HeadPose:
        image_points = np.array(
            [
                points[1],
                points[152],
                points[33],
                points[263],
                points[61],
                points[291],
            ],
            dtype=np.float32,
        )
        focal = float(width)
        camera_matrix = np.array(
            [[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1]],
            dtype=np.float32,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        success, rotation_vec, _ = cv2.solvePnP(
            self._model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return HeadPose(success=False, yaw=0.0, pitch=0.0, roll=0.0)

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        # cv2.RQDecomp3x3 returns x,y,z rotation angles in degrees.
        pitch_deg, yaw_deg, roll_deg = cv2.RQDecomp3x3(rotation_mat)[0]
        yaw = float(np.deg2rad(yaw_deg))
        pitch = float(np.deg2rad(pitch_deg))
        roll = float(np.deg2rad(roll_deg))
        return HeadPose(success=True, yaw=yaw, pitch=pitch, roll=roll)


class GazeEstimator:
    def __init__(self, config: RuntimeConfig, calibrator: "LinearCalibrator") -> None:
        self.config = config
        self.calibrator = calibrator
        self._head_pose = HeadPoseEstimator()

    @staticmethod
    def _weighted_average(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if values.shape[0] == 0:
            return np.zeros(2, dtype=np.float32)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-6:
            return np.mean(values, axis=0)
        return (values * weights[:, None]).sum(axis=0) / weight_sum

    def estimate(self, face: FaceObservation, width: int, height: int) -> GazeEstimate:
        head_pose = self._head_pose.estimate(face.points, width=width, height=height)
        left = face.left_eye
        right = face.right_eye

        eye_vectors = []
        eye_weights = []
        source = "none"
        if left.valid:
            eye_vectors.append(np.array([left.horizontal, left.vertical], dtype=np.float32))
            eye_weights.append(left.quality)
        if right.valid:
            eye_vectors.append(np.array([right.horizontal, right.vertical], dtype=np.float32))
            eye_weights.append(right.quality)

        mean_eye = np.zeros(2, dtype=np.float32)
        eye_conf = 0.0
        if eye_vectors:
            values = np.stack(eye_vectors, axis=0)
            weights = np.array(eye_weights, dtype=np.float32)
            mean_eye = self._weighted_average(values, weights)
            eye_conf = float(np.clip(weights.mean(), 0.0, 1.0))
            source = "eyes"

        # Raw feature vector is consumed by calibrator.
        raw_features = np.array(
            [
                left.horizontal if left.valid else 0.0,
                left.vertical if left.valid else 0.0,
                right.horizontal if right.valid else 0.0,
                right.vertical if right.valid else 0.0,
                head_pose.yaw if head_pose.success else 0.0,
                head_pose.pitch if head_pose.success else 0.0,
            ],
            dtype=np.float32,
        )

        if self.calibrator.is_ready:
            point = self.calibrator.predict(raw_features)
            source = "calibrated"
        else:
            # Heuristic mapping before calibration.
            x = (
                0.5
                - self.config.eye_gain_x * mean_eye[0]
                + self.config.head_gain_x * np.tanh(head_pose.yaw if head_pose.success else 0.0)
            )
            y = (
                0.5
                + self.config.eye_gain_y * mean_eye[1]
                + self.config.head_gain_y * np.tanh(head_pose.pitch if head_pose.success else 0.0)
            )
            point = np.array([x, y], dtype=np.float32)

        head_conf = 1.0 if head_pose.success else 0.0
        confidence = float(np.clip(0.15 + 0.65 * eye_conf + 0.2 * head_conf, 0.0, 1.0))
        point = np.clip(point, 0.0, 1.0).astype(np.float32)
        if eye_conf < 0.15 and head_conf < 0.1:
            return GazeEstimate(point=None, confidence=0.0, head_pose=head_pose, raw_features=None, source="lost")

        if source == "none" and head_pose.success:
            source = "head_pose"

        return GazeEstimate(
            point=point,
            confidence=confidence,
            head_pose=head_pose,
            raw_features=raw_features,
            source=source,
        )


# Avoid circular import in type checkers/runtime.
from .calibration import LinearCalibrator  # noqa: E402  pylint: disable=wrong-import-position
