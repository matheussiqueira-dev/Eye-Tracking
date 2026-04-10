"""Asynchronous webcam capture using a background reader thread."""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import RuntimeConfig

_log = logging.getLogger(__name__)


class AsyncCamera:
    """Non-blocking webcam reader that always exposes the latest frame.

    A daemon thread continuously reads frames from the capture device so
    the main loop never blocks waiting for a new frame.  Callers should
    pair :py:meth:`start` / :py:meth:`stop` with a ``try/finally`` block
    or use the context-manager interface.

    Args:
        config: Runtime configuration object.
    """

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self._capture: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts = 0.0
        self._consecutive_failures = 0
        self._MAX_CONSECUTIVE_FAILURES = 30

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "AsyncCamera":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the capture device and start the background reader thread.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        self._capture = cv2.VideoCapture(self.config.camera_id, backend)
        if not self._capture.isOpened():
            raise RuntimeError(
                f"Unable to open webcam id={self.config.camera_id}. "
                "Check that the device is connected and not in use by another process."
            )

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        self._capture.set(cv2.CAP_PROP_FPS, self.config.target_fps)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._capture.get(cv2.CAP_PROP_FPS)
        _log.info(
            "Camera %d opened: %dx%d @ %.0f FPS",
            self.config.camera_id,
            actual_w,
            actual_h,
            actual_fps,
        )

        self._running = True
        self._consecutive_failures = 0
        self._thread = threading.Thread(target=self._reader_loop, daemon=True, name="camera-reader")
        self._thread.start()

    def _reader_loop(self) -> None:
        assert self._capture is not None  # guaranteed by start()
        while self._running:
            ok, frame = self._capture.read()
            if not ok:
                self._consecutive_failures += 1
                if self._consecutive_failures == self._MAX_CONSECUTIVE_FAILURES:
                    _log.warning(
                        "Camera read has failed %d times consecutively; "
                        "check cable/driver health.",
                        self._consecutive_failures,
                    )
                time.sleep(0.001)
                continue

            if self._consecutive_failures > 0:
                _log.info("Camera read recovered after %d failures.", self._consecutive_failures)
                self._consecutive_failures = 0

            ts = time.perf_counter()
            with self._lock:
                self._latest_frame = frame
                self._latest_ts = ts

    def read(self) -> Tuple[Optional[np.ndarray], float]:
        """Return a copy of the latest frame and its timestamp.

        Returns:
            ``(frame, timestamp)`` where *frame* is ``None`` and
            *timestamp* is ``0.0`` if no frame has been captured yet.
        """
        with self._lock:
            if self._latest_frame is None:
                return None, 0.0
            return self._latest_frame.copy(), self._latest_ts

    def stop(self) -> None:
        """Stop the background thread and release the capture device."""
        _log.debug("Stopping camera reader thread.")
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                _log.warning("Camera reader thread did not terminate cleanly.")
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        _log.info("Camera released.")
