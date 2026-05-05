export type TrackingStatus = "idle" | "requesting-camera" | "running" | "paused" | "stopped";

export type ModelStatus = "idle" | "loading" | "ready" | "error";

export type CameraStatus =
  | "idle"
  | "requesting"
  | "ready"
  | "unsupported"
  | "denied"
  | "unavailable";

export type AttentionRegion =
  | "Top left"
  | "Top center"
  | "Top right"
  | "Middle left"
  | "Center"
  | "Middle right"
  | "Bottom left"
  | "Bottom center"
  | "Bottom right"
  | "No data";

export interface GazePoint {
  id: string;
  timestamp: number;
  elapsedMs: number;
  x: number;
  y: number;
  confidence: number;
  sessionId: string;
}

export interface NormalizedLandmark {
  x: number;
  y: number;
  z?: number;
}

export interface GazeEstimate {
  x: number;
  y: number;
  confidence: number;
  stability: number;
  source: "face-landmarks" | "fallback";
}

export interface HeatmapGrid {
  width: number;
  height: number;
  values: Float32Array;
}

export interface SessionMetrics {
  elapsedMs: number;
  pointsCaptured: number;
  dominantArea: AttentionRegion;
  averageStability: number;
  captureRate: number;
}

export interface SessionEvent {
  id: string;
  timestamp: number;
  label: string;
  description: string;
}
