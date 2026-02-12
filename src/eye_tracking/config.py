from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuntimeConfig:
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
