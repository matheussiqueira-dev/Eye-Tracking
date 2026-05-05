"use client";

import { Layers3, LocateFixed } from "lucide-react";
import { useEffect } from "react";

import { Badge } from "@/components/ui/Badge";
import { Card } from "@/components/ui/Card";
import { getMaxHeatmapValue } from "@/lib/eye-tracking/heatmap-utils";
import type { GazeEstimate, HeatmapGrid } from "@/types/eye-tracking";

interface HeatmapCanvasProps {
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  grid: HeatmapGrid;
  lastEstimate: GazeEstimate | null;
  pointsCaptured: number;
}

function heatmapColor(value: number): string {
  if (value > 0.78) {
    return `rgba(248, 113, 113, ${0.28 + value * 0.6})`;
  }

  if (value > 0.48) {
    return `rgba(251, 191, 36, ${0.18 + value * 0.54})`;
  }

  if (value > 0.2) {
    return `rgba(45, 212, 191, ${0.12 + value * 0.48})`;
  }

  return `rgba(56, 189, 248, ${0.08 + value * 0.36})`;
}

export function HeatmapCanvas({
  canvasRef,
  grid,
  lastEstimate,
  pointsCaptured,
}: HeatmapCanvasProps) {
  useEffect(() => {
    const canvas = canvasRef.current;

    if (!canvas) {
      return;
    }

    const context = canvas.getContext("2d");

    if (!context) {
      return;
    }

    const pixelRatio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || 960;
    const height = canvas.clientHeight || 540;
    canvas.width = Math.floor(width * pixelRatio);
    canvas.height = Math.floor(height * pixelRatio);
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);

    context.clearRect(0, 0, width, height);
    const background = context.createLinearGradient(0, 0, width, height);
    background.addColorStop(0, "#0b1720");
    background.addColorStop(0.52, "#10232a");
    background.addColorStop(1, "#142118");
    context.fillStyle = background;
    context.fillRect(0, 0, width, height);

    context.strokeStyle = "rgba(255,255,255,0.06)";
    context.lineWidth = 1;
    for (let col = 1; col < 3; col += 1) {
      const x = (width / 3) * col;
      context.beginPath();
      context.moveTo(x, 0);
      context.lineTo(x, height);
      context.stroke();
    }
    for (let row = 1; row < 3; row += 1) {
      const y = (height / 3) * row;
      context.beginPath();
      context.moveTo(0, y);
      context.lineTo(width, y);
      context.stroke();
    }

    const maxValue = getMaxHeatmapValue(grid);

    if (maxValue > 0) {
      const cellWidth = width / grid.width;
      const cellHeight = height / grid.height;

      for (let y = 0; y < grid.height; y += 1) {
        for (let x = 0; x < grid.width; x += 1) {
          const rawValue = grid.values[y * grid.width + x] ?? 0;

          if (rawValue <= 0) {
            continue;
          }

          const normalizedValue = Math.min(rawValue / maxValue, 1);
          context.fillStyle = heatmapColor(normalizedValue);
          context.fillRect(x * cellWidth, y * cellHeight, cellWidth + 1, cellHeight + 1);
        }
      }
    }

    if (lastEstimate) {
      const markerX = lastEstimate.x * width;
      const markerY = lastEstimate.y * height;
      context.beginPath();
      context.arc(markerX, markerY, 11, 0, Math.PI * 2);
      context.fillStyle = "rgba(251,191,36,0.86)";
      context.fill();
      context.strokeStyle = "rgba(255,255,255,0.92)";
      context.lineWidth = 2;
      context.stroke();
    }
  }, [canvasRef, grid, lastEstimate]);

  return (
    <Card className="overflow-hidden" role="region" aria-label="Attention heatmap">
      <div className="flex flex-col gap-4 border-b border-white/10 p-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-lg font-semibold text-white">Attention heatmap</h2>
          <p className="text-sm text-slate-400">
            Accumulated focus intensity over the analysis surface.
          </p>
        </div>
        <Badge className="border-amber-300/35 bg-amber-300/10 text-amber-100">
          {pointsCaptured} points
        </Badge>
      </div>

      <div className="relative aspect-video min-h-64 overflow-hidden">
        <canvas
          ref={canvasRef}
          aria-label="Real-time attention heatmap canvas"
          className="heatmap-canvas h-full w-full"
        />
        {pointsCaptured === 0 ? (
          <div className="pointer-events-none absolute inset-0 grid place-items-center p-6 text-center">
            <div className="grid max-w-sm gap-3 rounded-lg border border-white/10 bg-slate-950/72 p-5 backdrop-blur">
              <Layers3 aria-hidden="true" className="mx-auto text-cyan-200" size={28} />
              <p className="text-sm leading-6 text-slate-300">
                Heatmap points appear here as the model estimates attention direction.
              </p>
            </div>
          </div>
        ) : null}
        <div className="pointer-events-none absolute left-3 top-3 flex items-center gap-2 rounded-lg border border-white/10 bg-slate-950/70 px-3 py-2 text-xs text-slate-200 backdrop-blur">
          <LocateFixed aria-hidden="true" size={15} />
          Analysis surface
        </div>
      </div>
    </Card>
  );
}
