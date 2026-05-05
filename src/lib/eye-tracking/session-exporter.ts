import type { GazePoint } from "@/types/eye-tracking";

const CSV_HEADERS = ["timestamp", "x", "y", "confidence", "sessionId"] as const;

function escapeCsvValue(value: string): string {
  return /[",\n\r]/.test(value) ? `"${value.replaceAll('"', '""')}"` : value;
}

export function serializeSessionToJson(points: readonly GazePoint[]): string {
  return JSON.stringify(
    {
      exportedAt: new Date().toISOString(),
      totalPoints: points.length,
      points,
    },
    null,
    2,
  );
}

export function serializeSessionToCsv(points: readonly GazePoint[]): string {
  const rows = points.map((point) =>
    [
      new Date(point.timestamp).toISOString(),
      point.x.toFixed(5),
      point.y.toFixed(5),
      point.confidence.toFixed(4),
      point.sessionId,
    ]
      .map(escapeCsvValue)
      .join(","),
  );

  return [CSV_HEADERS.join(","), ...rows].join("\n");
}

export function downloadTextFile(filename: string, content: string, type: string): void {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

export function downloadDataUrl(filename: string, dataUrl: string): void {
  const link = document.createElement("a");
  link.href = dataUrl;
  link.download = filename;
  link.click();
}
