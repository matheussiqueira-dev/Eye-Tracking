"use client";

import { Cpu, Eye, ShieldAlert, Video } from "lucide-react";

import { Badge } from "@/components/ui/Badge";
import { Card } from "@/components/ui/Card";
import { formatPercent } from "@/lib/utils";
import type { CameraStatus, GazeEstimate, ModelStatus, TrackingStatus } from "@/types/eye-tracking";

interface ModelStatusPanelProps {
  cameraStatus: CameraStatus;
  errorMessage: string | null;
  lastEstimate: GazeEstimate | null;
  modelStatus: ModelStatus;
  trackingStatus: TrackingStatus;
}

const modelLabels: Record<ModelStatus, string> = {
  error: "Model error",
  idle: "Model idle",
  loading: "Loading model",
  ready: "Model ready",
};

const trackingLabels: Record<TrackingStatus, string> = {
  idle: "Idle",
  paused: "Paused",
  "requesting-camera": "Requesting camera",
  running: "Running",
  stopped: "Stopped",
};

export function ModelStatusPanel({
  cameraStatus,
  errorMessage,
  lastEstimate,
  modelStatus,
  trackingStatus,
}: ModelStatusPanelProps) {
  return (
    <Card className="p-4" role="region" aria-label="Model status">
      <div className="grid gap-4">
        <div>
          <h2 className="text-lg font-semibold text-white">Model status</h2>
          <p className="text-sm text-slate-400">
            MediaPipe Face Landmarker runs locally in the browser.
          </p>
        </div>

        <div className="grid gap-3">
          <div className="flex items-center justify-between gap-3">
            <span className="flex items-center gap-2 text-sm text-slate-300">
              <Cpu aria-hidden="true" size={16} />
              Detection model
            </span>
            <Badge
              className={
                modelStatus === "ready"
                  ? "border-emerald-300/40 bg-emerald-300/10 text-emerald-100"
                  : ""
              }
            >
              {modelLabels[modelStatus]}
            </Badge>
          </div>
          <div className="flex items-center justify-between gap-3">
            <span className="flex items-center gap-2 text-sm text-slate-300">
              <Video aria-hidden="true" size={16} />
              Camera
            </span>
            <span className="text-sm font-medium text-white">{cameraStatus}</span>
          </div>
          <div className="flex items-center justify-between gap-3">
            <span className="flex items-center gap-2 text-sm text-slate-300">
              <Eye aria-hidden="true" size={16} />
              Tracking
            </span>
            <span className="text-sm font-medium text-white">{trackingLabels[trackingStatus]}</span>
          </div>
        </div>

        <div className="rounded-lg border border-white/10 bg-white/6 p-3">
          <p className="text-xs uppercase text-slate-500">Current estimate</p>
          <div className="mt-2 grid grid-cols-3 gap-2 font-mono text-sm text-white">
            <span>x {lastEstimate ? lastEstimate.x.toFixed(2) : "--"}</span>
            <span>y {lastEstimate ? lastEstimate.y.toFixed(2) : "--"}</span>
            <span>{lastEstimate ? formatPercent(lastEstimate.confidence) : "--"}</span>
          </div>
        </div>

        <div className="flex gap-3 rounded-lg border border-amber-300/25 bg-amber-300/8 p-3 text-sm leading-6 text-amber-50">
          <ShieldAlert aria-hidden="true" className="mt-1 shrink-0" size={17} />
          <p>
            Prototype analytics only. The gaze estimate is approximate and must not be treated as
            clinical or scientific measurement.
          </p>
        </div>

        {errorMessage ? (
          <p className="rounded-lg border border-red-400/30 bg-red-500/10 p-3 text-sm leading-6 text-red-100">
            {errorMessage}
          </p>
        ) : null}
      </div>
    </Card>
  );
}
