"use client";

import { Download, FileJson, FileSpreadsheet, Pause, Play, RotateCcw, Square } from "lucide-react";

import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import type { TrackingStatus } from "@/types/eye-tracking";

interface TrackingControlsProps {
  hasData: boolean;
  onExportCsv: () => void;
  onExportHeatmap: () => void;
  onExportJson: () => void;
  onPause: () => void;
  onReset: () => void;
  onStart: () => void;
  onStop: () => void;
  status: TrackingStatus;
}

export function TrackingControls({
  hasData,
  onExportCsv,
  onExportHeatmap,
  onExportJson,
  onPause,
  onReset,
  onStart,
  onStop,
  status,
}: TrackingControlsProps) {
  const isRunning = status === "running";
  const isRequestingCamera = status === "requesting-camera";

  return (
    <Card className="p-4" role="region" aria-label="Tracking controls">
      <div className="grid gap-4">
        <div>
          <h2 className="text-lg font-semibold text-white">Session controls</h2>
          <p className="text-sm text-slate-400">
            Capture, pause, reset and export UX attention data.
          </p>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <Button
            aria-label="Start analysis"
            className="col-span-2"
            disabled={isRunning || isRequestingCamera}
            onClick={onStart}
            variant="primary"
          >
            <Play aria-hidden="true" size={17} />
            {status === "paused" ? "Resume analysis" : "Start analysis"}
          </Button>
          <Button aria-label="Pause analysis" disabled={!isRunning} onClick={onPause}>
            <Pause aria-hidden="true" size={17} />
            Pause
          </Button>
          <Button
            aria-label="Stop analysis"
            disabled={status === "idle" || status === "stopped" || isRequestingCamera}
            onClick={onStop}
          >
            <Square aria-hidden="true" size={17} />
            Stop
          </Button>
          <Button aria-label="Reset heatmap and session" className="col-span-2" onClick={onReset}>
            <RotateCcw aria-hidden="true" size={17} />
            Reset session
          </Button>
        </div>

        <div className="grid gap-3 border-t border-white/10 pt-4">
          <Button disabled={!hasData} onClick={onExportJson} variant="secondary">
            <FileJson aria-hidden="true" size={17} />
            Export JSON
          </Button>
          <Button disabled={!hasData} onClick={onExportCsv} variant="secondary">
            <FileSpreadsheet aria-hidden="true" size={17} />
            Export CSV
          </Button>
          <Button disabled={!hasData} onClick={onExportHeatmap} variant="secondary">
            <Download aria-hidden="true" size={17} />
            Export heatmap
          </Button>
        </div>
      </div>
    </Card>
  );
}
