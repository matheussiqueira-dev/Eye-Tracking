import { clamp } from "@/lib/utils";
import type { NormalizedLandmark } from "@/types/eye-tracking";

export interface Vector2 {
  x: number;
  y: number;
}

export function midpoint(a: Vector2, b: Vector2): Vector2 {
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
  };
}

export function distance(a: Vector2, b: Vector2): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export function average(points: readonly Vector2[]): Vector2 {
  if (points.length === 0) {
    return { x: 0.5, y: 0.5 };
  }

  const sum = points.reduce((acc, point) => ({ x: acc.x + point.x, y: acc.y + point.y }), {
    x: 0,
    y: 0,
  });

  return {
    x: sum.x / points.length,
    y: sum.y / points.length,
  };
}

export function landmarkAt(
  landmarks: readonly NormalizedLandmark[],
  index: number,
): NormalizedLandmark | undefined {
  return landmarks[index];
}

export function normalizedRegion(x: number, y: number): string {
  const columns = ["left", "center", "right"];
  const rows = ["Top", "Middle", "Bottom"];
  const col = columns[Math.min(2, Math.floor(clamp(x) * 3))] ?? "center";
  const row = rows[Math.min(2, Math.floor(clamp(y) * 3))] ?? "Middle";

  return row === "Middle" && col === "center" ? "Center" : `${row} ${col}`;
}
