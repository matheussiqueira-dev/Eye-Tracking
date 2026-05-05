import { estimateGazeFromLandmarks } from "@/lib/eye-tracking/gaze-estimator";
import type { NormalizedLandmark } from "@/types/eye-tracking";

function createLandmarks(irisOffsetX = 0, irisOffsetY = 0): NormalizedLandmark[] {
  const landmarks: NormalizedLandmark[] = Array.from({ length: 478 }, () => ({
    x: 0.5,
    y: 0.5,
    z: 0,
  }));

  landmarks[33] = { x: 0.38, y: 0.42 };
  landmarks[133] = { x: 0.5, y: 0.42 };
  landmarks[159] = { x: 0.44, y: 0.39 };
  landmarks[145] = { x: 0.44, y: 0.45 };
  landmarks[263] = { x: 0.62, y: 0.42 };
  landmarks[362] = { x: 0.5, y: 0.42 };
  landmarks[386] = { x: 0.56, y: 0.39 };
  landmarks[374] = { x: 0.56, y: 0.45 };

  for (const index of [469, 470, 471, 472]) {
    landmarks[index] = { x: 0.44 + irisOffsetX, y: 0.42 + irisOffsetY };
  }

  for (const index of [474, 475, 476, 477]) {
    landmarks[index] = { x: 0.56 + irisOffsetX, y: 0.42 + irisOffsetY };
  }

  return landmarks;
}

describe("estimateGazeFromLandmarks", () => {
  it("returns a normalized gaze estimate from eye and iris landmarks", () => {
    const estimate = estimateGazeFromLandmarks(createLandmarks());

    expect(estimate).not.toBeNull();
    expect(estimate?.x).toBeGreaterThanOrEqual(0);
    expect(estimate?.x).toBeLessThanOrEqual(1);
    expect(estimate?.y).toBeGreaterThanOrEqual(0);
    expect(estimate?.y).toBeLessThanOrEqual(1);
    expect(estimate?.confidence).toBeGreaterThan(0.5);
  });

  it("smooths movement when a previous estimate is available", () => {
    const previous = estimateGazeFromLandmarks(createLandmarks());
    const next = estimateGazeFromLandmarks(createLandmarks(0.04, 0.02), previous);

    expect(next).not.toBeNull();
    expect(next?.stability).toBeGreaterThanOrEqual(0);
    expect(next?.stability).toBeLessThanOrEqual(1);
  });

  it("returns null when required landmarks are absent", () => {
    expect(estimateGazeFromLandmarks([])).toBeNull();
  });
});
