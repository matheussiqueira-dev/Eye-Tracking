"use client";

import { Activity, Clock3, Gauge, MapPinned, SignalHigh } from "lucide-react";

import { Card } from "@/components/ui/Card";
import { formatDuration, formatPercent } from "@/lib/utils";
import type { SessionMetrics } from "@/types/eye-tracking";

interface MetricsCardsProps {
  metrics: SessionMetrics;
}

export function MetricsCards({ metrics }: MetricsCardsProps) {
  const cards = [
    {
      description: "Active analysis time",
      icon: Clock3,
      label: "Session time",
      value: formatDuration(metrics.elapsedMs),
    },
    {
      description: "Captured gaze samples",
      icon: Activity,
      label: "Data points",
      value: metrics.pointsCaptured.toLocaleString("en-US"),
    },
    {
      description: "Most observed region",
      icon: MapPinned,
      label: "Hot area",
      value: metrics.dominantArea,
    },
    {
      description: "Average gaze steadiness",
      icon: Gauge,
      label: "Stability",
      value: formatPercent(metrics.averageStability),
    },
    {
      description: "Samples per second",
      icon: SignalHigh,
      label: "Capture rate",
      value: `${metrics.captureRate.toFixed(1)}/s`,
    },
  ];

  return (
    <section aria-label="Session metrics" className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
      {cards.map((card) => {
        const Icon = card.icon;

        return (
          <Card className="grid min-h-36 gap-3 p-4" key={card.label}>
            <div className="flex items-center justify-between gap-3">
              <span className="text-sm font-medium text-slate-400">{card.label}</span>
              <span className="flex size-9 items-center justify-center rounded-lg border border-white/10 bg-white/6 text-cyan-200">
                <Icon aria-hidden="true" size={18} />
              </span>
            </div>
            <strong className="break-words text-2xl font-semibold text-white">{card.value}</strong>
            <p className="text-sm leading-6 text-slate-400">{card.description}</p>
          </Card>
        );
      })}
    </section>
  );
}
