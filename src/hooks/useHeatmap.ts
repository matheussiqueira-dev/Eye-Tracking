"use client";

import { useCallback, useState } from "react";

import {
  addPointToHeatmap,
  createHeatmapGrid,
  getDominantArea,
} from "@/lib/eye-tracking/heatmap-utils";
import type { GazePoint, HeatmapGrid } from "@/types/eye-tracking";

interface UseHeatmapResult {
  addPoint: (point: GazePoint) => void;
  dominantArea: ReturnType<typeof getDominantArea>;
  grid: HeatmapGrid;
  resetHeatmap: () => void;
}

export function useHeatmap(): UseHeatmapResult {
  const [grid, setGrid] = useState(() => createHeatmapGrid());

  const addPoint = useCallback((point: GazePoint) => {
    setGrid((currentGrid) => addPointToHeatmap(currentGrid, point));
  }, []);

  const resetHeatmap = useCallback(() => {
    setGrid(createHeatmapGrid());
  }, []);

  return {
    addPoint,
    dominantArea: getDominantArea(grid),
    grid,
    resetHeatmap,
  };
}
