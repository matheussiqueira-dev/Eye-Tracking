import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { TrackingControls } from "@/components/eye-tracking/TrackingControls";

const handlers = {
  onExportCsv: jest.fn(),
  onExportHeatmap: jest.fn(),
  onExportJson: jest.fn(),
  onPause: jest.fn(),
  onReset: jest.fn(),
  onStart: jest.fn(),
  onStop: jest.fn(),
};

describe("TrackingControls", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("starts a session from the idle state", async () => {
    const user = userEvent.setup();

    render(<TrackingControls {...handlers} hasData={false} status="idle" />);

    await user.click(screen.getByRole("button", { name: /start analysis/i }));

    expect(handlers.onStart).toHaveBeenCalledTimes(1);
  });

  it("disables exports until there is captured data", () => {
    render(<TrackingControls {...handlers} hasData={false} status="idle" />);

    expect(screen.getByRole("button", { name: /export json/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /export csv/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /export heatmap/i })).toBeDisabled();
  });
});
