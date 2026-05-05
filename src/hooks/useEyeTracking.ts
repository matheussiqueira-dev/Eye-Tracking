"use client";

import { useEffect, useRef, useState } from "react";

import {
  FACE_LANDMARKER_MODEL_URL,
  MEDIAPIPE_WASM_URL,
  TRACKING_INTERVAL_MS,
} from "@/lib/constants";
import { estimateGazeFromLandmarks } from "@/lib/eye-tracking/gaze-estimator";
import type {
  GazeEstimate,
  GazePoint,
  ModelStatus,
  NormalizedLandmark,
} from "@/types/eye-tracking";

interface UseEyeTrackingParams {
  enabled: boolean;
  onPoint: (point: GazePoint, estimate: GazeEstimate) => void;
  sessionId: string;
  sessionStartedAt: number | null;
  videoRef: React.RefObject<HTMLVideoElement | null>;
}

interface UseEyeTrackingResult {
  errorMessage: string | null;
  lastEstimate: GazeEstimate | null;
  modelStatus: ModelStatus;
}

interface FaceLandmarkerLike {
  detectForVideo: (
    video: HTMLVideoElement,
    timestamp: number,
  ) => { faceLandmarks?: NormalizedLandmark[][] };
}

export function useEyeTracking({
  enabled,
  onPoint,
  sessionId,
  sessionStartedAt,
  videoRef,
}: UseEyeTrackingParams): UseEyeTrackingResult {
  const [modelStatus, setModelStatus] = useState<ModelStatus>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [lastEstimate, setLastEstimate] = useState<GazeEstimate | null>(null);
  const landmarkerRef = useRef<FaceLandmarkerLike | null>(null);
  const previousEstimateRef = useRef<GazeEstimate | null>(null);
  const onPointRef = useRef(onPoint);

  useEffect(() => {
    onPointRef.current = onPoint;
  }, [onPoint]);

  useEffect(() => {
    let cancelled = false;

    async function loadModel(): Promise<void> {
      if (!enabled || landmarkerRef.current || modelStatus !== "idle") {
        return;
      }

      setModelStatus("loading");
      setErrorMessage(null);

      try {
        const { FaceLandmarker, FilesetResolver } = await import("@mediapipe/tasks-vision");
        const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);
        const landmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            delegate: "GPU",
            modelAssetPath: FACE_LANDMARKER_MODEL_URL,
          },
          numFaces: 1,
          outputFaceBlendshapes: false,
          outputFacialTransformationMatrixes: false,
          runningMode: "VIDEO",
        });

        if (!cancelled) {
          landmarkerRef.current = landmarker as FaceLandmarkerLike;
          setModelStatus("ready");
        }
      } catch {
        if (!cancelled) {
          setModelStatus("error");
          setErrorMessage(
            "Face landmark model could not be loaded. Check network and WebGL support.",
          );
        }
      }
    }

    void loadModel();

    return () => {
      cancelled = true;
    };
  }, [enabled, modelStatus]);

  useEffect(() => {
    if (!enabled || modelStatus !== "ready" || !sessionStartedAt) {
      return;
    }

    let frameId = 0;
    let lastCaptureAt = 0;

    const detect = (timestamp: number) => {
      const video = videoRef.current;
      const landmarker = landmarkerRef.current;

      if (
        video &&
        landmarker &&
        video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
        timestamp - lastCaptureAt >= TRACKING_INTERVAL_MS
      ) {
        lastCaptureAt = timestamp;
        const result = landmarker.detectForVideo(video, timestamp);
        const landmarks = result.faceLandmarks?.[0];
        const estimate = landmarks
          ? estimateGazeFromLandmarks(landmarks, previousEstimateRef.current)
          : null;

        if (estimate) {
          previousEstimateRef.current = estimate;
          setLastEstimate(estimate);
          onPointRef.current(
            {
              confidence: estimate.confidence,
              elapsedMs: Date.now() - sessionStartedAt,
              id: `${sessionId}-${Date.now()}`,
              sessionId,
              timestamp: Date.now(),
              x: estimate.x,
              y: estimate.y,
            },
            estimate,
          );
        }
      }

      frameId = requestAnimationFrame(detect);
    };

    frameId = requestAnimationFrame(detect);

    return () => cancelAnimationFrame(frameId);
  }, [enabled, modelStatus, sessionId, sessionStartedAt, videoRef]);

  return { errorMessage, lastEstimate, modelStatus };
}
