"use client";

import { useMemo } from "react";

import { getDominantArea } from "@/lib/eye-tracking/heatmap-utils";
import type { GazePoint, HeatmapGrid, SessionMetrics, TrackingStatus } from "@/types/eye-tracking";

interface UseSessionMetricsParams {
  grid: HeatmapGrid;
  now: number;
  points: readonly GazePoint[];
  sessionStartedAt: number | null;
  status: TrackingStatus;
}

export function useSessionMetrics({
  grid,
  now,
  points,
  sessionStartedAt,
  status,
}: UseSessionMetricsParams): SessionMetrics {
  return useMemo(() => {
    const elapsedMs = sessionStartedAt ? now - sessionStartedAt : 0;
    const captureRate = elapsedMs > 0 ? points.length / (elapsedMs / 1000) : 0;
    const averageStability =
      points.length > 1
        ? points.reduce((sum, point, index) => {
            const previous = points[index - 1];
            if (!previous) {
              return sum;
            }
            const movement = Math.hypot(point.x - previous.x, point.y - previous.y);
            return sum + Math.max(0, 1 - movement * 8);
          }, 0) / Math.max(points.length - 1, 1)
        : status === "running"
          ? 1
          : 0;

    return {
      elapsedMs,
      pointsCaptured: points.length,
      dominantArea: getDominantArea(grid),
      averageStability,
      captureRate,
    };
  }, [grid, now, points, sessionStartedAt, status]);
}
