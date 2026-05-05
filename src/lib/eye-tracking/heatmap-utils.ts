import { HEATMAP_HEIGHT, HEATMAP_WIDTH } from "@/lib/constants";
import { clamp } from "@/lib/utils";
import type { AttentionRegion, GazePoint, HeatmapGrid } from "@/types/eye-tracking";

const DEFAULT_RADIUS = 4;

export function createHeatmapGrid(width = HEATMAP_WIDTH, height = HEATMAP_HEIGHT): HeatmapGrid {
  return {
    width,
    height,
    values: new Float32Array(width * height),
  };
}

export function cloneHeatmapGrid(grid: HeatmapGrid): HeatmapGrid {
  return {
    width: grid.width,
    height: grid.height,
    values: new Float32Array(grid.values),
  };
}

export function addPointToHeatmap(
  grid: HeatmapGrid,
  point: Pick<GazePoint, "x" | "y" | "confidence">,
  radius = DEFAULT_RADIUS,
): HeatmapGrid {
  const next = cloneHeatmapGrid(grid);
  const centerX = Math.round(clamp(point.x) * (grid.width - 1));
  const centerY = Math.round(clamp(point.y) * (grid.height - 1));
  const strength = clamp(point.confidence, 0.05, 1);

  for (let y = centerY - radius; y <= centerY + radius; y += 1) {
    for (let x = centerX - radius; x <= centerX + radius; x += 1) {
      if (x < 0 || y < 0 || x >= grid.width || y >= grid.height) {
        continue;
      }

      const dx = x - centerX;
      const dy = y - centerY;
      const distanceSquared = dx * dx + dy * dy;
      const falloff = Math.exp(-distanceSquared / (2 * radius));
      const valueIndex = y * grid.width + x;
      next.values[valueIndex] = (next.values[valueIndex] ?? 0) + strength * falloff;
    }
  }

  return next;
}

export function getMaxHeatmapValue(grid: HeatmapGrid): number {
  return grid.values.reduce((max, value) => Math.max(max, value), 0);
}

export function getDominantArea(grid: HeatmapGrid): AttentionRegion {
  if (getMaxHeatmapValue(grid) <= 0) {
    return "No data";
  }

  const buckets = new Array<number>(9).fill(0);

  for (let y = 0; y < grid.height; y += 1) {
    for (let x = 0; x < grid.width; x += 1) {
      const col = Math.min(2, Math.floor((x / grid.width) * 3));
      const row = Math.min(2, Math.floor((y / grid.height) * 3));
      const bucketIndex = row * 3 + col;
      buckets[bucketIndex] = (buckets[bucketIndex] ?? 0) + (grid.values[y * grid.width + x] ?? 0);
    }
  }

  const dominantIndex = buckets.reduce(
    (best, value, index) => (value > (buckets[best] ?? 0) ? index : best),
    0,
  );

  const labels: AttentionRegion[] = [
    "Top left",
    "Top center",
    "Top right",
    "Middle left",
    "Center",
    "Middle right",
    "Bottom left",
    "Bottom center",
    "Bottom right",
  ];

  return labels[dominantIndex] ?? "No data";
}
