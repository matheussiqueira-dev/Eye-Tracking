from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import RuntimeConfig


class AsyncCamera:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self._capture: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts = 0.0

    def start(self) -> None:
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        self._capture = cv2.VideoCapture(self.config.camera_id, backend)
        if not self._capture.isOpened():
            raise RuntimeError(f"Unable to open webcam id={self.config.camera_id}")

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        self._capture.set(cv2.CAP_PROP_FPS, self.config.target_fps)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self) -> None:
        assert self._capture is not None
        while self._running:
            ok, frame = self._capture.read()
            if not ok:
                time.sleep(0.001)
                continue
            ts = time.perf_counter()
            with self._lock:
                self._latest_frame = frame
                self._latest_ts = ts

    def read(self) -> Tuple[Optional[np.ndarray], float]:
        with self._lock:
            if self._latest_frame is None:
                return None, 0.0
            return self._latest_frame.copy(), self._latest_ts

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
