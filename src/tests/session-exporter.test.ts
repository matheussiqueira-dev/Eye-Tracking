import { serializeSessionToCsv, serializeSessionToJson } from "@/lib/eye-tracking/session-exporter";
import type { GazePoint } from "@/types/eye-tracking";

const points: GazePoint[] = [
  {
    confidence: 0.8123,
    elapsedMs: 120,
    id: "point-1",
    sessionId: "session-test",
    timestamp: Date.UTC(2026, 4, 4, 12, 0, 0),
    x: 0.25,
    y: 0.75,
  },
];

describe("session exporter", () => {
  it("serializes session data to JSON", () => {
    const parsed = JSON.parse(serializeSessionToJson(points)) as { totalPoints: number };

    expect(parsed.totalPoints).toBe(1);
  });

  it("serializes session data to CSV with required columns", () => {
    const csv = serializeSessionToCsv(points);

    expect(csv).toContain("timestamp,x,y,confidence,sessionId");
    expect(csv).toContain("2026-05-04T12:00:00.000Z,0.25000,0.75000,0.8123,session-test");
  });
});
