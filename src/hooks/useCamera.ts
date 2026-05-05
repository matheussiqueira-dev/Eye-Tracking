"use client";

import { useCallback, useEffect, useState } from "react";

import type { CameraStatus } from "@/types/eye-tracking";

interface UseCameraResult {
  errorMessage: string | null;
  startCamera: () => Promise<boolean>;
  status: CameraStatus;
  stopCamera: () => void;
  stream: MediaStream | null;
}

function toCameraError(error: unknown): { status: CameraStatus; message: string } {
  if (error instanceof DOMException && error.name === "NotAllowedError") {
    return {
      status: "denied",
      message: "Camera permission was denied. Enable camera access to run the analysis.",
    };
  }

  if (error instanceof DOMException && error.name === "NotFoundError") {
    return {
      status: "unavailable",
      message: "No camera device was found on this browser.",
    };
  }

  return {
    status: "unavailable",
    message: "Unable to initialize the camera preview.",
  };
}

export function useCamera(): UseCameraResult {
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [status, setStatus] = useState<CameraStatus>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const stopCamera = useCallback(() => {
    setStream((currentStream) => {
      currentStream?.getTracks().forEach((track) => track.stop());
      return null;
    });
    setStatus("idle");
  }, []);

  const startCamera = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus("unsupported");
      setErrorMessage("This browser does not support navigator.mediaDevices.getUserMedia.");
      return false;
    }

    setStatus("requesting");
    setErrorMessage(null);

    try {
      const nextStream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          frameRate: { ideal: 30 },
          height: { ideal: 720 },
          width: { ideal: 1280 },
        },
      });

      setStream((currentStream) => {
        currentStream?.getTracks().forEach((track) => track.stop());
        return nextStream;
      });
      setStatus("ready");
      return true;
    } catch (error) {
      const cameraError = toCameraError(error);
      setStatus(cameraError.status);
      setErrorMessage(cameraError.message);
      return false;
    }
  }, []);

  useEffect(() => stopCamera, [stopCamera]);

  return { errorMessage, startCamera, status, stopCamera, stream };
}
