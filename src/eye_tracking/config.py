"""Runtime configuration for the eye-tracking pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RuntimeConfig:
    """Centralised, immutable-by-convention configuration object.

    All numeric fields have sensible defaults that work with most consumer
    webcams.  Tweak :py:attr:`eye_gain_x` / :py:attr:`eye_gain_y` and the
    head-pose gain pair when the uncalibrated gaze point feels sluggish or
    over-sensitive.

    Args:
        camera_id: OS device index passed to ``cv2.VideoCapture``.
        frame_width: Requested capture width in pixels.
        frame_height: Requested capture height in pixels.
        target_fps: Requested capture frame-rate.
        show_debug: Render diagnostic overlays on the output window.
        show_heatmap: Render the attention heatmap on the output window.
        process_every_n: Run full MediaPipe inference every *N* frames.
            Values > 1 trade accuracy for throughput on slow hardware.
        min_confidence: Minimum gaze-estimate confidence required before a
            point is added to the heatmap and event log.
        outlier_threshold: Euclidean-distance threshold (normalised [0, 1])
            above which a low-confidence jump is rejected as an outlier.
        eye_gain_x: Horizontal eye-vector contribution to uncalibrated gaze.
        eye_gain_y: Vertical eye-vector contribution to uncalibrated gaze.
        head_gain_x: Horizontal head-pose contribution to uncalibrated gaze.
        head_gain_y: Vertical head-pose contribution to uncalibrated gaze.
        heatmap_height: Internal heatmap grid height (low-res for speed).
        heatmap_width: Internal heatmap grid width.
        heatmap_sigma: Gaussian kernel standard deviation in grid cells.
        heatmap_decay: Exponential decay rate (per second) for the heatmap.
        one_euro_min_cutoff: One Euro filter minimum cutoff frequency (Hz).
        one_euro_beta: One Euro filter speed coefficient.
        one_euro_d_cutoff: One Euro filter derivative cutoff frequency (Hz).
    """

    camera_id: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    target_fps: int = 120
    show_debug: bool = True
    show_heatmap: bool = True
    process_every_n: int = 1
    min_confidence: float = 0.30
    outlier_threshold: float = 0.20
    eye_gain_x: float = 0.65
    eye_gain_y: float = 0.65
    head_gain_x: float = 0.35
    head_gain_y: float = 0.40
    heatmap_height: int = 72
    heatmap_width: int = 128
    heatmap_sigma: float = 2.2
    heatmap_decay: float = 1.2
    one_euro_min_cutoff: float = 1.5
    one_euro_beta: float = 0.08
    one_euro_d_cutoff: float = 1.0

    def __post_init__(self) -> None:
        if self.process_every_n < 1:
            raise ValueError(f"process_every_n must be >= 1, got {self.process_every_n}")
        if not 0.0 < self.min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be in (0, 1], got {self.min_confidence}")
        if self.heatmap_sigma <= 0.0:
            raise ValueError(f"heatmap_sigma must be > 0, got {self.heatmap_sigma}")
