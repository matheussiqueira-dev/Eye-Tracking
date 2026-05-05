import { clamp } from "@/lib/utils";
import type { GazeEstimate, NormalizedLandmark } from "@/types/eye-tracking";

import { average, distance, landmarkAt, midpoint } from "./vector-utils";

const LEFT_EYE = {
  outer: 33,
  inner: 133,
  top: 159,
  bottom: 145,
  iris: [469, 470, 471, 472],
} as const;

const RIGHT_EYE = {
  outer: 263,
  inner: 362,
  top: 386,
  bottom: 374,
  iris: [474, 475, 476, 477],
} as const;

interface EyeVector {
  horizontal: number;
  vertical: number;
  openness: number;
  confidence: number;
}

function estimateEyeVector(
  landmarks: readonly NormalizedLandmark[],
  eye: typeof LEFT_EYE | typeof RIGHT_EYE,
): EyeVector | null {
  const outer = landmarkAt(landmarks, eye.outer);
  const inner = landmarkAt(landmarks, eye.inner);
  const top = landmarkAt(landmarks, eye.top);
  const bottom = landmarkAt(landmarks, eye.bottom);
  const irisPoints = eye.iris
    .map((index) => landmarkAt(landmarks, index))
    .filter((point): point is NormalizedLandmark => Boolean(point));

  if (!outer || !inner || !top || !bottom) {
    return null;
  }

  const center = midpoint(outer, inner);
  const iris = irisPoints.length >= 2 ? average(irisPoints) : center;
  const eyeWidth = Math.max(distance(outer, inner), 0.001);
  const eyeHeight = Math.max(distance(top, bottom), 0.001);

  const horizontal = (iris.x - center.x) / eyeWidth;
  const vertical = (iris.y - center.y) / eyeHeight;
  const openness = clamp(eyeHeight / eyeWidth, 0, 1);
  const irisConfidence = irisPoints.length >= 2 ? 1 : 0.45;
  const confidence = clamp(0.35 + openness * 1.8, 0, 1) * irisConfidence;

  return { horizontal, vertical, openness, confidence };
}

export function estimateGazeFromLandmarks(
  landmarks: readonly NormalizedLandmark[],
  previous?: GazeEstimate | null,
): GazeEstimate | null {
  const left = estimateEyeVector(landmarks, LEFT_EYE);
  const right = estimateEyeVector(landmarks, RIGHT_EYE);
  const eyes = [left, right].filter((eye): eye is EyeVector => Boolean(eye));

  if (eyes.length === 0) {
    return null;
  }

  const horizontal = eyes.reduce((sum, eye) => sum + eye.horizontal, 0) / eyes.length;
  const vertical = eyes.reduce((sum, eye) => sum + eye.vertical, 0) / eyes.length;
  const confidence = eyes.reduce((sum, eye) => sum + eye.confidence, 0) / eyes.length;

  const rawX = 0.5 - horizontal * 1.9;
  const rawY = 0.5 + vertical * 1.15;
  const x = clamp(previous ? previous.x * 0.58 + rawX * 0.42 : rawX);
  const y = clamp(previous ? previous.y * 0.58 + rawY * 0.42 : rawY);
  const movement = previous ? Math.hypot(x - previous.x, y - previous.y) : 0;
  const stability = clamp(1 - movement * 8);

  return {
    x,
    y,
    confidence: clamp(confidence),
    stability,
    source: "face-landmarks",
  };
}
