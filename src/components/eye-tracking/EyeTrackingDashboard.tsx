"use client";

import { BrainCircuit, Database, Flame, ShieldCheck } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { CameraPreview } from "@/components/eye-tracking/CameraPreview";
import { HeatmapCanvas } from "@/components/eye-tracking/HeatmapCanvas";
import { MetricsCards } from "@/components/eye-tracking/MetricsCards";
import { ModelStatusPanel } from "@/components/eye-tracking/ModelStatusPanel";
import { SessionTimeline } from "@/components/eye-tracking/SessionTimeline";
import { TrackingControls } from "@/components/eye-tracking/TrackingControls";
import { Badge } from "@/components/ui/Badge";
import { MAX_SESSION_POINTS } from "@/lib/constants";
import {
  downloadDataUrl,
  downloadTextFile,
  serializeSessionToCsv,
  serializeSessionToJson,
} from "@/lib/eye-tracking/session-exporter";
import { createSessionId } from "@/lib/utils";
import { useCamera } from "@/hooks/useCamera";
import { useEyeTracking } from "@/hooks/useEyeTracking";
import { useHeatmap } from "@/hooks/useHeatmap";
import { useSessionMetrics } from "@/hooks/useSessionMetrics";
import type { GazeEstimate, GazePoint, SessionEvent, TrackingStatus } from "@/types/eye-tracking";

function createEvent(label: string, description: string): SessionEvent {
  return {
    description,
    id: `event-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`,
    label,
    timestamp: Date.now(),
  };
}

export function EyeTrackingDashboard() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const heatmapCanvasRef = useRef<HTMLCanvasElement>(null);
  const [sessionId, setSessionId] = useState(() => createSessionId());
  const [sessionStartedAt, setSessionStartedAt] = useState<number | null>(null);
  const [status, setStatus] = useState<TrackingStatus>("idle");
  const [points, setPoints] = useState<GazePoint[]>([]);
  const [events, setEvents] = useState<SessionEvent[]>([]);
  const [now, setNow] = useState(() => Date.now());
  const {
    errorMessage: cameraError,
    startCamera,
    status: cameraStatus,
    stopCamera,
    stream,
  } = useCamera();
  const { addPoint, grid, resetHeatmap } = useHeatmap();

  const recordEvent = useCallback((label: string, description: string) => {
    setEvents((currentEvents) => [createEvent(label, description), ...currentEvents].slice(0, 12));
  }, []);

  const handlePoint = useCallback(
    (point: GazePoint) => {
      addPoint(point);
      setNow(Date.now());
      setPoints((currentPoints) => [...currentPoints.slice(-(MAX_SESSION_POINTS - 1)), point]);
    },
    [addPoint],
  );

  const {
    errorMessage: modelError,
    lastEstimate,
    modelStatus,
  } = useEyeTracking({
    enabled: status === "running",
    onPoint: handlePoint,
    sessionId,
    sessionStartedAt,
    videoRef,
  });

  const metrics = useSessionMetrics({
    grid,
    now,
    points,
    sessionStartedAt,
    status,
  });

  useEffect(() => {
    if (status !== "running") {
      return;
    }

    const intervalId = window.setInterval(() => setNow(Date.now()), 500);
    return () => window.clearInterval(intervalId);
  }, [status]);

  const handleStart = useCallback(async () => {
    setStatus("requesting-camera");

    const cameraReady = stream ? true : await startCamera();

    if (!cameraReady) {
      setStatus("idle");
      recordEvent("Camera unavailable", "The browser could not grant a webcam stream.");
      return;
    }

    const startedAt = sessionStartedAt ?? Date.now();
    setSessionStartedAt(startedAt);
    setNow(Date.now());
    setStatus("running");
    recordEvent(
      status === "paused" ? "Session resumed" : "Session started",
      "Webcam preview, face landmarks and heatmap capture are active.",
    );
  }, [recordEvent, sessionStartedAt, startCamera, status, stream]);

  const handlePause = useCallback(() => {
    if (status !== "running") {
      return;
    }

    setStatus("paused");
    recordEvent(
      "Session paused",
      "Heatmap accumulation was paused without deleting captured data.",
    );
  }, [recordEvent, status]);

  const handleStop = useCallback(() => {
    if (status === "idle" || status === "stopped") {
      return;
    }

    setStatus("stopped");
    stopCamera();
    recordEvent("Session stopped", "Camera stream was closed and captured data remains available.");
  }, [recordEvent, status, stopCamera]);

  const handleReset = useCallback(() => {
    stopCamera();
    resetHeatmap();
    setPoints([]);
    setSessionId(createSessionId());
    setSessionStartedAt(null);
    setStatus("idle");
    setNow(Date.now());
    setEvents([createEvent("Session reset", "Heatmap, metrics and samples were cleared.")]);
  }, [resetHeatmap, stopCamera]);

  const handleExportJson = useCallback(() => {
    downloadTextFile(
      `${sessionId}.json`,
      serializeSessionToJson(points),
      "application/json;charset=utf-8",
    );
    recordEvent("JSON exported", "A structured session export was generated.");
  }, [points, recordEvent, sessionId]);

  const handleExportCsv = useCallback(() => {
    downloadTextFile(`${sessionId}.csv`, serializeSessionToCsv(points), "text/csv;charset=utf-8");
    recordEvent("CSV exported", "A tabular session export was generated.");
  }, [points, recordEvent, sessionId]);

  const handleExportHeatmap = useCallback(() => {
    const canvas = heatmapCanvasRef.current;

    if (!canvas) {
      return;
    }

    downloadDataUrl(`${sessionId}-heatmap.png`, canvas.toDataURL("image/png"));
    recordEvent("Heatmap exported", "The current canvas heatmap was exported as PNG.");
  }, [recordEvent, sessionId]);

  const highlightedEstimate: GazeEstimate | null = lastEstimate;
  const modelPanelError = modelError ?? cameraError;

  return (
    <div className="min-h-screen">
      <section className="mx-auto grid w-full max-w-7xl gap-8 px-4 pb-8 pt-10 sm:px-6 lg:grid-cols-[minmax(0,1fr)_24rem] lg:px-8 lg:pt-14">
        <div className="grid content-center gap-5">
          <Badge>Advanced UX analytics prototype</Badge>
          <h1 className="max-w-4xl text-4xl font-semibold text-white sm:text-6xl">
            Eye Tracking UX Intelligence
          </h1>
          <p className="max-w-3xl text-base leading-8 text-slate-300 sm:text-lg">
            Turn a regular webcam into a real-time behavioral analytics surface. The browser
            estimates face and eye landmarks, maps approximate attention coordinates, and builds a
            heatmap for product research sessions.
          </p>
        </div>

        <aside className="grid gap-3 self-end rounded-lg border border-white/10 bg-slate-950/72 p-4 backdrop-blur">
          <div className="flex items-start gap-3">
            <BrainCircuit aria-hidden="true" className="mt-1 text-cyan-200" size={20} />
            <p className="text-sm leading-6 text-slate-300">
              Client-side computer vision with no server upload.
            </p>
          </div>
          <div className="flex items-start gap-3">
            <Flame aria-hidden="true" className="mt-1 text-amber-200" size={20} />
            <p className="text-sm leading-6 text-slate-300">
              Real-time heatmap accumulation through Canvas API.
            </p>
          </div>
          <div className="flex items-start gap-3">
            <Database aria-hidden="true" className="mt-1 text-emerald-200" size={20} />
            <p className="text-sm leading-6 text-slate-300">
              JSON, CSV and PNG exports for UX research handoff.
            </p>
          </div>
          <div className="flex items-start gap-3">
            <ShieldCheck aria-hidden="true" className="mt-1 text-slate-200" size={20} />
            <p className="text-sm leading-6 text-slate-300">
              Approximate prototype, not biometric diagnosis.
            </p>
          </div>
        </aside>
      </section>

      <section className="mx-auto grid w-full max-w-7xl gap-6 px-4 pb-16 sm:px-6 lg:px-8">
        <MetricsCards metrics={metrics} />

        <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_24rem]">
          <div className="grid gap-6">
            <CameraPreview
              errorMessage={cameraError}
              lastEstimate={highlightedEstimate}
              status={cameraStatus}
              stream={stream}
              videoRef={videoRef}
            />
            <HeatmapCanvas
              canvasRef={heatmapCanvasRef}
              grid={grid}
              lastEstimate={highlightedEstimate}
              pointsCaptured={points.length}
            />
          </div>

          <aside className="grid content-start gap-6">
            <TrackingControls
              hasData={points.length > 0}
              onExportCsv={handleExportCsv}
              onExportHeatmap={handleExportHeatmap}
              onExportJson={handleExportJson}
              onPause={handlePause}
              onReset={handleReset}
              onStart={handleStart}
              onStop={handleStop}
              status={status}
            />
            <ModelStatusPanel
              cameraStatus={cameraStatus}
              errorMessage={modelPanelError}
              lastEstimate={highlightedEstimate}
              modelStatus={modelStatus}
              trackingStatus={status}
            />
            <SessionTimeline events={events} points={points} />
          </aside>
        </div>
      </section>
    </div>
  );
}
