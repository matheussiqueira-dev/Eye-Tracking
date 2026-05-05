# Eye Tracking UX Intelligence

Desenvolvido por Matheus Siqueira - www.matheussiqueira.dev

Eye Tracking UX Intelligence is a browser-based UX analytics prototype that turns a regular webcam into a real-time attention analysis surface. It uses webcam Web APIs, MediaPipe Face Landmarker, vector-based gaze estimation and Canvas rendering to capture approximate attention points and build heatmaps for UX research sessions.

This project is an advanced prototype. It does not provide clinical, biometric or scientific-grade gaze accuracy.

## Features

- Webcam permission flow with explicit unsupported, denied and unavailable camera states.
- Real-time webcam preview.
- Client-side face and ocular landmark detection with `@mediapipe/tasks-vision`.
- Approximate gaze estimation from iris and eye landmarks.
- Live visual indicator for the estimated attention point.
- Real-time Canvas heatmap accumulation.
- Session controls for start, pause, stop and reset.
- Metrics for session time, captured points, most observed area, gaze stability and capture rate.
- Session history with recent events and samples.
- Export to JSON, CSV and PNG heatmap.
- Next.js App Router, TypeScript, Tailwind CSS v4, ESLint, Prettier and Jest.

## Architecture

```txt
src/
  app/
    globals.css
    layout.tsx
    page.tsx
  components/
    layout/
      Header.tsx
      Footer.tsx
    eye-tracking/
      CameraPreview.tsx
      EyeTrackingDashboard.tsx
      HeatmapCanvas.tsx
      MetricsCards.tsx
      ModelStatusPanel.tsx
      SessionTimeline.tsx
      TrackingControls.tsx
    ui/
      Badge.tsx
      Button.tsx
      Card.tsx
      Section.tsx
  hooks/
    useCamera.ts
    useEyeTracking.ts
    useHeatmap.ts
    useSessionMetrics.ts
  lib/
    eye-tracking/
      gaze-estimator.ts
      heatmap-utils.ts
      session-exporter.ts
      vector-utils.ts
    constants.ts
    utils.ts
  tests/
    gaze-estimator.test.ts
    heatmap-utils.test.ts
    session-exporter.test.ts
    tracking-controls.test.tsx
  types/
    eye-tracking.ts
```

## Getting Started

```bash
npm install
npm run dev
```

Open `http://127.0.0.1:3000`.

For webcam access, use `localhost`, `127.0.0.1` or HTTPS. Modern browsers block camera access on insecure remote origins.

## Quality Commands

```bash
npm run lint
npm run type-check
npm run test
npm run build
```

## Vercel Deploy

The app is compatible with Vercel zero-config Next.js deployments.

Required environment variables: none.

Optional environment variables:

```bash
NEXT_PUBLIC_SITE_URL=https://your-production-domain.com
```

The MediaPipe model and WASM runtime are loaded from public Google/CDN URLs configured in `src/lib/constants.ts`.

## Export Format

CSV columns:

```txt
timestamp,x,y,confidence,sessionId
```

JSON includes export metadata, total points and all captured gaze points.

## Credits

Desenvolvido por Matheus Siqueira - www.matheussiqueira.dev
