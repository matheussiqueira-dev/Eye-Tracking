"""Eye tracking package: real-time gaze estimation with MediaPipe and OpenCV.

Public API
----------
RuntimeConfig : dataclass
    Pipeline configuration with sensible defaults.
EyeTrackingApp : class
    Orchestrates the full tracking pipeline.
GazeEstimate : dataclass
    Fused gaze estimate produced each frame.
CalibrationRoutine : class
    Guided 9-point calibration routine.

Example
-------
    from eye_tracking import EyeTrackingApp, RuntimeConfig

    config = RuntimeConfig(camera_id=0, show_heatmap=True)
    app = EyeTrackingApp(config=config)
    app.run()
"""

from .calibration import CalibrationRoutine, LinearCalibrator
from .config import RuntimeConfig
from .gaze import GazeEstimate, HeadPose
from .runner import EyeTrackingApp

__all__ = [
    "RuntimeConfig",
    "EyeTrackingApp",
    "GazeEstimate",
    "HeadPose",
    "LinearCalibrator",
    "CalibrationRoutine",
]
