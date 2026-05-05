import {
  addPointToHeatmap,
  createHeatmapGrid,
  getDominantArea,
  getMaxHeatmapValue,
} from "@/lib/eye-tracking/heatmap-utils";

describe("heatmap utilities", () => {
  it("creates an empty grid with no dominant area", () => {
    const grid = createHeatmapGrid(9, 9);

    expect(grid.values).toHaveLength(81);
    expect(getMaxHeatmapValue(grid)).toBe(0);
    expect(getDominantArea(grid)).toBe("No data");
  });

  it("accumulates weighted points into the expected region", () => {
    const grid = createHeatmapGrid(30, 30);
    const nextGrid = addPointToHeatmap(grid, { confidence: 0.9, x: 0.12, y: 0.14 }, 3);

    expect(getMaxHeatmapValue(nextGrid)).toBeGreaterThan(0);
    expect(getDominantArea(nextGrid)).toBe("Top left");
  });

  it("does not mutate the source grid", () => {
    const grid = createHeatmapGrid(12, 12);
    const nextGrid = addPointToHeatmap(grid, { confidence: 1, x: 0.5, y: 0.5 }, 2);

    expect(getMaxHeatmapValue(grid)).toBe(0);
    expect(getMaxHeatmapValue(nextGrid)).toBeGreaterThan(0);
  });
});
