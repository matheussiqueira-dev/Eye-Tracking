from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np


RIGHT_IRIS = (469, 470, 471, 472)
LEFT_IRIS = (474, 475, 476, 477)

# FaceMesh canonical indices used by downstream modules
KEY_INDICES = (
    1,  # nose
    152,  # chin
    33,  # right eye outer
    263,  # left eye outer
    61,  # mouth left
    291,  # mouth right
    133,  # right eye inner
    362,  # left eye inner
    159,  # right eye top
    145,  # right eye bottom
    386,  # left eye top
    374,  # left eye bottom
)


@dataclass
class EyeObservation:
    valid: bool
    center: np.ndarray
    iris: np.ndarray
    horizontal: float
    vertical: float
    openness: float
    quality: float


@dataclass
class FaceObservation:
    found: bool
    points: Dict[int, np.ndarray]
    left_eye: EyeObservation
    right_eye: EyeObservation


class FaceLandmarkTracker:
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

    def __init__(self) -> None:
        self._backend = "solutions" if self._has_solutions_backend() else "tasks"
        self._face_mesh = None
        self._face_landmarker = None

        if self._backend == "solutions":
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            model_path = self._ensure_task_model()
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    @staticmethod
    def _has_solutions_backend() -> bool:
        return hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")

    @classmethod
    def _ensure_task_model(cls) -> Path:
        root = Path(__file__).resolve().parents[2]
        model_dir = root / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "face_landmarker.task"
        if model_path.exists():
            return model_path

        tmp_path = model_dir / "face_landmarker.task.tmp"
        try:
            urlretrieve(cls.MODEL_URL, tmp_path)
            tmp_path.replace(model_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        return model_path

    @staticmethod
    def _denorm(landmark: object, width: int, height: int) -> np.ndarray:
        x = float(landmark.x) * width
        y = float(landmark.y) * height
        return np.array([x, y], dtype=np.float32)

    @staticmethod
    def _safe_norm(vector: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, float]:
        magnitude = float(np.linalg.norm(vector))
        if magnitude < eps:
            return np.zeros_like(vector), 0.0
        return vector / magnitude, magnitude

    def _build_eye(
        self,
        all_points: Dict[int, np.ndarray],
        iris_indices: Tuple[int, int, int, int],
        outer_idx: int,
        inner_idx: int,
        top_idx: int,
        bottom_idx: int,
    ) -> EyeObservation:
        if any(idx not in all_points for idx in (outer_idx, inner_idx, top_idx, bottom_idx)):
            zero = np.zeros(2, dtype=np.float32)
            return EyeObservation(False, zero, zero, 0.0, 0.0, 0.0, 0.0)

        outer = all_points[outer_idx]
        inner = all_points[inner_idx]
        top = all_points[top_idx]
        bottom = all_points[bottom_idx]

        iris_points = [all_points[idx] for idx in iris_indices if idx in all_points]
        center = (outer + inner) * 0.5
        has_iris = len(iris_points) >= 2
        iris = np.mean(np.array(iris_points, dtype=np.float32), axis=0) if has_iris else center.copy()

        axis_x, eye_width = self._safe_norm(inner - outer)
        axis_y, eye_height = self._safe_norm(bottom - top)
        openness = eye_height / max(eye_width, 1e-6)

        offset = iris - center
        horizontal = float(np.dot(offset, axis_x) / max(eye_width * 0.5, 1e-6))
        vertical = float(np.dot(offset, axis_y) / max(eye_height * 0.5, 1e-6))

        valid_geometry = eye_width > 3.0 and eye_height > 1.0 and openness > 0.08
        in_bounds = abs(horizontal) < 1.6 and abs(vertical) < 1.6
        quality = 1.0
        quality -= min(abs(horizontal), 1.6) / 2.2
        quality -= min(abs(vertical), 1.6) / 2.2
        if not has_iris:
            quality *= 0.1
        quality = float(np.clip(quality, 0.0, 1.0))

        valid = valid_geometry and in_bounds and has_iris and quality > 0.2
        return EyeObservation(
            valid=valid,
            center=center,
            iris=iris,
            horizontal=horizontal,
            vertical=vertical,
            openness=float(openness),
            quality=quality if valid else 0.0,
        )

    def process(self, frame_bgr: np.ndarray) -> Optional[FaceObservation]:
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self._backend == "solutions":
            assert self._face_mesh is not None
            result = self._face_mesh.process(frame_rgb)
            if not result.multi_face_landmarks:
                return None
            mesh = result.multi_face_landmarks[0].landmark
        else:
            assert self._face_landmarker is not None
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self._face_landmarker.detect(mp_image)
            if not result.face_landmarks:
                return None
            mesh = result.face_landmarks[0]

        needed = set(KEY_INDICES) | set(LEFT_IRIS) | set(RIGHT_IRIS)
        points: Dict[int, np.ndarray] = {}
        for idx in needed:
            if idx < len(mesh):
                points[idx] = self._denorm(mesh[idx], width, height)

        # Head pose needs this subset. If anything is missing, skip the frame.
        if any(idx not in points for idx in KEY_INDICES):
            return None

        right_eye = self._build_eye(
            all_points=points,
            iris_indices=RIGHT_IRIS,
            outer_idx=33,
            inner_idx=133,
            top_idx=159,
            bottom_idx=145,
        )
        left_eye = self._build_eye(
            all_points=points,
            iris_indices=LEFT_IRIS,
            outer_idx=263,
            inner_idx=362,
            top_idx=386,
            bottom_idx=374,
        )

        return FaceObservation(
            found=True,
            points=points,
            left_eye=left_eye,
            right_eye=right_eye,
        )

    def close(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()
        if self._face_landmarker is not None:
            self._face_landmarker.close()
