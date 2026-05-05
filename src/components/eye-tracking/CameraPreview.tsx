"use client";

import { Camera, CameraOff, ScanFace } from "lucide-react";
import { useEffect } from "react";

import { Badge } from "@/components/ui/Badge";
import { Card } from "@/components/ui/Card";
import type { CameraStatus, GazeEstimate } from "@/types/eye-tracking";

interface CameraPreviewProps {
  errorMessage: string | null;
  lastEstimate: GazeEstimate | null;
  status: CameraStatus;
  stream: MediaStream | null;
  videoRef: React.RefObject<HTMLVideoElement | null>;
}

const cameraLabels: Record<CameraStatus, string> = {
  denied: "Permission denied",
  idle: "Camera idle",
  ready: "Camera online",
  requesting: "Requesting camera",
  unavailable: "Camera unavailable",
  unsupported: "Browser unsupported",
};

export function CameraPreview({
  errorMessage,
  lastEstimate,
  status,
  stream,
  videoRef,
}: CameraPreviewProps) {
  useEffect(() => {
    const video = videoRef.current;

    if (!video) {
      return;
    }

    video.srcObject = stream;

    if (stream) {
      void video.play().catch(() => undefined);
    }
  }, [stream, videoRef]);

  const hasCameraError =
    status === "denied" || status === "unavailable" || status === "unsupported";

  return (
    <Card className="overflow-hidden" role="region" aria-label="Webcam preview">
      <div className="flex flex-col gap-4 border-b border-white/10 p-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-lg font-semibold text-white">Webcam analysis</h2>
          <p className="text-sm text-slate-400">Live preview with an estimated attention marker.</p>
        </div>
        <Badge
          className={
            status === "ready" ? "border-emerald-300/40 bg-emerald-300/10 text-emerald-100" : ""
          }
        >
          {cameraLabels[status]}
        </Badge>
      </div>

      <div className="relative aspect-video min-h-64 overflow-hidden bg-slate-950">
        <video
          ref={videoRef}
          aria-label="Live webcam preview"
          autoPlay
          className="h-full w-full scale-x-[-1] object-cover"
          muted
          playsInline
        />

        {!stream ? (
          <div className="absolute inset-0 grid place-items-center bg-slate-950/92 p-6 text-center">
            <div className="grid max-w-sm gap-3">
              <span className="mx-auto flex size-14 items-center justify-center rounded-lg border border-white/10 bg-white/6 text-cyan-200">
                {hasCameraError ? (
                  <CameraOff aria-hidden="true" size={26} />
                ) : (
                  <Camera aria-hidden="true" size={26} />
                )}
              </span>
              <p className="text-base font-medium text-white">
                {hasCameraError ? "Camera cannot be initialized" : "Camera preview is waiting"}
              </p>
              <p className="text-sm leading-6 text-slate-400">
                {errorMessage ??
                  "Start the analysis to request camera permission and begin landmark detection."}
              </p>
            </div>
          </div>
        ) : null}

        {stream && lastEstimate ? (
          <span
            aria-label="Estimated gaze point"
            className="pointer-events-none absolute size-8 rounded-full border-2 border-white bg-amber-300/80 shadow-[0_0_40px_rgba(251,191,36,0.65)]"
            style={{
              left: `${(1 - lastEstimate.x) * 100}%`,
              top: `${lastEstimate.y * 100}%`,
              transform: "translate(-50%, -50%)",
            }}
          />
        ) : null}

        <div className="pointer-events-none absolute left-3 top-3 flex items-center gap-2 rounded-lg border border-white/10 bg-slate-950/70 px-3 py-2 text-xs text-slate-200 backdrop-blur">
          <ScanFace aria-hidden="true" size={15} />
          Approximate gaze
        </div>
      </div>
    </Card>
  );
}
