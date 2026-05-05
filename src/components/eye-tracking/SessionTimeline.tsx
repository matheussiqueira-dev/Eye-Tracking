"use client";

import { ListChecks } from "lucide-react";

import { Card } from "@/components/ui/Card";
import { formatDuration, formatPercent } from "@/lib/utils";
import type { GazePoint, SessionEvent } from "@/types/eye-tracking";

interface SessionTimelineProps {
  events: readonly SessionEvent[];
  points: readonly GazePoint[];
}

export function SessionTimeline({ events, points }: SessionTimelineProps) {
  const latestPoints = points.slice(-5).reverse();

  return (
    <Card className="p-4" role="region" aria-label="Session history">
      <div className="grid gap-4">
        <div>
          <h2 className="flex items-center gap-2 text-lg font-semibold text-white">
            <ListChecks aria-hidden="true" size={18} />
            Session history
          </h2>
          <p className="text-sm text-slate-400">Recent events and captured attention points.</p>
        </div>

        <div className="grid gap-3">
          {events.length === 0 ? (
            <p className="rounded-lg border border-white/10 bg-white/6 p-3 text-sm text-slate-400">
              No session events yet.
            </p>
          ) : (
            events.slice(0, 4).map((event) => (
              <article className="rounded-lg border border-white/10 bg-white/6 p-3" key={event.id}>
                <div className="flex items-center justify-between gap-3">
                  <h3 className="text-sm font-semibold text-white">{event.label}</h3>
                  <time
                    className="font-mono text-xs text-slate-500"
                    dateTime={new Date(event.timestamp).toISOString()}
                  >
                    {new Date(event.timestamp).toLocaleTimeString("en-US", {
                      hour: "2-digit",
                      minute: "2-digit",
                      second: "2-digit",
                    })}
                  </time>
                </div>
                <p className="mt-1 text-sm leading-6 text-slate-400">{event.description}</p>
              </article>
            ))
          )}
        </div>

        <div className="grid gap-2 border-t border-white/10 pt-4">
          <h3 className="text-sm font-semibold text-white">Latest samples</h3>
          {latestPoints.length === 0 ? (
            <p className="text-sm text-slate-500">Waiting for gaze samples.</p>
          ) : (
            <ul className="grid gap-2">
              {latestPoints.map((point) => (
                <li
                  className="grid grid-cols-[1fr_auto] items-center gap-3 rounded-lg border border-white/10 bg-slate-900/50 px-3 py-2 text-sm"
                  key={point.id}
                >
                  <span className="font-mono text-slate-200">
                    {formatDuration(point.elapsedMs)} / x {point.x.toFixed(2)} / y{" "}
                    {point.y.toFixed(2)}
                  </span>
                  <span className="font-mono text-cyan-200">{formatPercent(point.confidence)}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </Card>
  );
}
