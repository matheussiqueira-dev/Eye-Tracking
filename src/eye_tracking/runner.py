from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .calibration import CalibrationRoutine, LinearCalibrator
from .capture import AsyncCamera
from .config import RuntimeConfig
from .filters import GazeStabilizer
from .gaze import GazeEstimate, GazeEstimator, HeadPose
from .heatmap import HeatmapAccumulator, HeatmapConfig
from .landmarks import FaceLandmarkTracker, FaceObservation


@dataclass
class GazeEvent:
    ts: float
    x: float
    y: float
    confidence: float
    source: str
    fps: float


class EventLogger:
    def __init__(self, output_path: Optional[str]) -> None:
        self._file = None
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("a", encoding="utf-8")

    def write(self, event: GazeEvent) -> None:
        if self._file is None:
            return
        self._file.write(json.dumps(asdict(event), ensure_ascii=True) + "\n")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class EyeTrackingApp:
    def __init__(self, config: RuntimeConfig, event_log_path: Optional[str] = None) -> None:
        self.config = config
        self.camera = AsyncCamera(config)
        self.tracker = FaceLandmarkTracker()
        self.calibrator = LinearCalibrator()
        self.calibration = CalibrationRoutine(self.calibrator)
        self.estimator = GazeEstimator(config=config, calibrator=self.calibrator)
        self.stabilizer = GazeStabilizer(config)
        self.heatmap = HeatmapAccumulator(
            HeatmapConfig(
                height=config.heatmap_height,
                width=config.heatmap_width,
                sigma=config.heatmap_sigma,
                decay=config.heatmap_decay,
            )
        )
        self.logger = EventLogger(event_log_path)
        self._fps_times: deque[float] = deque(maxlen=120)
        self._last_face: Optional[FaceObservation] = None
        self._last_estimate: Optional[GazeEstimate] = None
        self._last_point: Optional[np.ndarray] = None
        self._frame_index = 0
        self._calibration_done_at = 0.0

    def _compute_fps(self, now: float) -> float:
        self._fps_times.append(now)
        if len(self._fps_times) < 2:
            return 0.0
        span = self._fps_times[-1] - self._fps_times[0]
        if span <= 1e-6:
            return 0.0
        return (len(self._fps_times) - 1) / span

    @staticmethod
    def _to_px(point: np.ndarray, width: int, height: int) -> tuple[int, int]:
        x = int(np.clip(point[0], 0.0, 1.0) * (width - 1))
        y = int(np.clip(point[1], 0.0, 1.0) * (height - 1))
        return x, y

    def _draw_debug(
        self,
        frame: np.ndarray,
        face: Optional[FaceObservation],
        estimate: Optional[GazeEstimate],
        stabilized: Optional[np.ndarray],
        fps: float,
    ) -> None:
        height, width = frame.shape[:2]
        if face is not None:
            for eye in (face.left_eye, face.right_eye):
                c = (0, 255, 0) if eye.valid else (0, 70, 255)
                cv2.circle(frame, tuple(np.int32(eye.center)), 4, c, -1, cv2.LINE_AA)
                cv2.circle(frame, tuple(np.int32(eye.iris)), 5, c, 2, cv2.LINE_AA)

        if stabilized is not None:
            px = self._to_px(stabilized, width, height)
            cv2.circle(frame, px, 12, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(frame, px, (0, 30, 255), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)

        lines = [f"FPS: {fps:5.1f}"]
        if estimate is not None:
            lines.append(f"Source: {estimate.source}")
            lines.append(f"Conf: {estimate.confidence:.2f}")
            lines.append(
                "Head(y,p,r): "
                f"{np.rad2deg(estimate.head_pose.yaw):+.1f}, "
                f"{np.rad2deg(estimate.head_pose.pitch):+.1f}, "
                f"{np.rad2deg(estimate.head_pose.roll):+.1f}"
            )
        else:
            lines.append("Source: none")
            lines.append("Conf: 0.00")

        status = (
            "CALIBRADO"
            if self.calibrator.is_ready
            else ("CALIBRANDO" if self.calibration.active else "SEM CALIBRACAO")
        )
        lines.append(f"Modo: {status}")
        lines.append("Teclas: C calibrar | H heatmap | D debug | R reset | Q sair")

        y = 26
        for text in lines:
            cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
            y += 22

        if self._calibration_done_at > 0 and (time.perf_counter() - self._calibration_done_at) < 2.0:
            msg = "Calibracao concluida"
            cv2.putText(frame, msg, (width - 290, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 3, cv2.LINE_AA)
            cv2.putText(frame, msg, (width - 290, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 255, 130), 1, cv2.LINE_AA)

    def run(self) -> None:
        self.camera.start()
        last_ts = time.perf_counter()
        show_debug = self.config.show_debug
        show_heatmap = self.config.show_heatmap

        try:
            while True:
                frame, frame_ts = self.camera.read()
                if frame is None:
                    time.sleep(0.001)
                    continue

                now = frame_ts if frame_ts > 0 else time.perf_counter()
                dt = max(now - last_ts, 1e-3)
                last_ts = now
                fps = self._compute_fps(now)
                self._frame_index += 1

                do_process = (self._frame_index % max(self.config.process_every_n, 1)) == 0
                if do_process:
                    self._last_face = self.tracker.process(frame)
                    if self._last_face is not None:
                        h, w = frame.shape[:2]
                        self._last_estimate = self.estimator.estimate(self._last_face, width=w, height=h)
                    else:
                        self._last_estimate = None

                face = self._last_face
                estimate = self._last_estimate
                measurement = None
                confidence = 0.0
                source = "lost"

                if estimate is not None and estimate.point is not None:
                    measurement = estimate.point
                    confidence = estimate.confidence
                    source = estimate.source
                    self.calibration.update(estimate.raw_features)
                    if self.calibration.consume_finished_flag():
                        self._calibration_done_at = time.perf_counter()
                else:
                    self.calibration.update(None)

                stabilized = self.stabilizer.update(
                    measurement=measurement,
                    dt=dt,
                    confidence=confidence,
                )
                if confidence >= self.config.min_confidence and measurement is not None:
                    self._last_point = stabilized
                    self.heatmap.update(stabilized, confidence, dt)
                    self.logger.write(
                        GazeEvent(
                            ts=now,
                            x=float(stabilized[0]),
                            y=float(stabilized[1]),
                            confidence=float(confidence),
                            source=source,
                            fps=float(fps),
                        )
                    )
                else:
                    self.heatmap.update(None, 0.0, dt)

                output = frame.copy()
                if show_heatmap:
                    output = self.heatmap.overlay(output)
                self.calibration.draw(output)
                if show_debug:
                    self._draw_debug(output, face, estimate, self._last_point, fps)

                cv2.imshow("Eye Tracking", output)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    self.calibration.start()
                if key == ord("h"):
                    show_heatmap = not show_heatmap
                if key == ord("d"):
                    show_debug = not show_debug
                if key == ord("r"):
                    self.heatmap.reset()
                    self.calibrator.reset()
                if key == ord("s"):
                    filename = f"heatmap_{int(time.time())}.png"
                    cv2.imwrite(filename, self.heatmap.overlay(frame.copy()))

        finally:
            self.logger.close()
            self.tracker.close()
            self.camera.stop()
            cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real-time eye tracking with regular webcam.")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam index.")
    parser.add_argument("--width", type=int, default=1280, help="Capture width.")
    parser.add_argument("--height", type=int, default=720, help="Capture height.")
    parser.add_argument("--target-fps", type=int, default=120, help="Requested camera FPS.")
    parser.add_argument("--process-every-n", type=int, default=1, help="Run full inference every N frames.")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug overlay.")
    parser.add_argument("--no-heatmap", action="store_true", help="Disable heatmap overlay.")
    parser.add_argument("--log-events", type=str, default="", help="Optional NDJSON output path.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = RuntimeConfig(
        camera_id=args.camera_id,
        frame_width=args.width,
        frame_height=args.height,
        target_fps=args.target_fps,
        process_every_n=max(1, args.process_every_n),
        show_debug=not args.no_debug,
        show_heatmap=not args.no_heatmap,
    )
    app = EyeTrackingApp(config=config, event_log_path=args.log_events or None)
    app.run()
