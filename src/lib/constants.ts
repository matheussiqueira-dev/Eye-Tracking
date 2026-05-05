export const APP_NAME = "Eye Tracking UX Intelligence";
export const AUTHOR_CREDIT = "Desenvolvido por Matheus Siqueira - www.matheussiqueira.dev";
export const AUTHOR_URL = "https://www.matheussiqueira.dev/";
export const SITE_URL =
  process.env.NEXT_PUBLIC_SITE_URL ?? "https://eye-tracking-ux-intelligence.vercel.app";

export const MEDIAPIPE_VERSION = "0.10.35";
export const MEDIAPIPE_WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;
export const FACE_LANDMARKER_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";

export const HEATMAP_WIDTH = 64;
export const HEATMAP_HEIGHT = 36;
export const MAX_SESSION_POINTS = 5000;
export const TRACKING_INTERVAL_MS = 80;
