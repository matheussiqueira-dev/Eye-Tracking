import { EncomPanel } from "../components/ui/EncomPanel";
import { TronButton } from "../components/ui/TronButton";
import { TronCard } from "../components/ui/TronCard";
import styles from "./page.module.css";

const telemetryCards = [
  {
    eyebrow: "Acquisition",
    heading: "Optical Capture Matrix",
    description:
      "Frames, landmarks, and iris vectors are staged through a clean control surface built for quick operator scans.",
    thumbnailLabel: "CAPTURE",
    tags: ["Low latency", "Async input", "Telemetry"],
    details: [
      "Accommodates camera, fps, and resolution controls.",
      "Highlights confidence shifts without overwhelming the operator.",
      "Keeps the panel layout stable across desktop and mobile widths.",
    ],
  },
  {
    eyebrow: "Calibration",
    heading: "Nine-Point Alignment Grid",
    description:
      "Calibration states are grouped into high-contrast cards with expandable diagnostics for confidence and regression quality.",
    thumbnailLabel: "ALIGN",
    tags: ["Regression", "Heatmap", "Guided flow"],
    details: [
      "Expandable diagnostics keep advanced data available on demand.",
      "The scan-line effect reinforces state changes without JavaScript-heavy animation.",
      "Typography and spacing preserve readability under neon contrast.",
    ],
  },
  {
    eyebrow: "Analytics",
    heading: "Attention Heatmap Console",
    description:
      "Session summaries, export states, and visual attention overlays sit inside modular ENCOM containers for future scaling.",
    thumbnailLabel: "HEATMAP",
    tags: ["NDJSON", "Session view", "Dashboard"],
    details: [
      "Panels are reusable for dashboards, cards, and future modal shells.",
      "Cards reorganize automatically for tablet and mobile breakpoints.",
      "No heavy runtime dependencies are required for the motion language.",
    ],
  },
];

const stripMetrics = [
  { value: "00.9ms", label: "UI transition budget" },
  { value: "CSS", label: "Primary animation stack" },
  { value: "SSR", label: "App Router ready" },
  { value: "Vercel", label: "Deployment target" },
];

const signalStates = [
  { label: "Theme tokens", value: "Online" },
  { label: "Grid renderer", value: "Nominal" },
  { label: "Panel system", value: "Reusable" },
  { label: "Responsive shell", value: "Adaptive" },
];

export default function HomePage() {
  return (
    <div className={styles.page}>
      <section className={styles.hero}>
        <EncomPanel
          className={styles.heroPanel}
          label="ENCOM UI System"
          heading="Grid-Ready Command Center"
          headingLevel={1}
        >
          <div className={`${styles.heroCopy} encom-fade-in`}>
            <span className={styles.kicker}>Tron Legacy visual layer</span>
            <h1 className={styles.headline}>Eye Tracking Interface Protocol</h1>
            <p className={styles.lead}>
              A dark cyber control surface for calibration, telemetry, and heatmap
              workflows, built as a lean App Router shell with reusable components and
              CSS-driven motion.
            </p>

            <div className={styles.actions}>
              <TronButton href="#modules">Inspect modules</TronButton>
              <TronButton
                href="https://www.matheussiqueira.dev/"
                target="_blank"
                rel="noreferrer"
              >
                Open author link
              </TronButton>
            </div>

            <div className={styles.stats}>
              <div className={styles.stat}>
                <span className={styles.statValue}>3</span>
                <span className={styles.statLabel}>Core primitives</span>
              </div>
              <div className={styles.stat}>
                <span className={styles.statValue}>100%</span>
                <span className={styles.statLabel}>CSS token driven</span>
              </div>
              <div className={styles.stat}>
                <span className={styles.statValue}>0</span>
                <span className={styles.statLabel}>Heavy UI libs added</span>
              </div>
            </div>
          </div>
        </EncomPanel>

        <EncomPanel className={styles.sidePanel} label="System Status" heading="Signal Integrity">
          <div className={`${styles.ring} encom-glow-pulse`} aria-hidden="true" />
          <div className={styles.statusList}>
            {signalStates.map((item) => (
              <div key={item.label} className={styles.statusItem}>
                <span className={styles.statusKey}>{item.label}</span>
                <span className={styles.statusValue}>{item.value}</span>
              </div>
            ))}
          </div>
        </EncomPanel>
      </section>

      <section className={styles.strip}>
        {stripMetrics.map((metric) => (
          <div key={metric.label} className={styles.stripCard}>
            <span className={styles.stripValue}>{metric.value}</span>
            <span className={styles.stripLabel}>{metric.label}</span>
          </div>
        ))}
      </section>

      <section id="modules" className={styles.modules}>
        <div className={styles.sectionHeader}>
          <div className={styles.sectionCopy}>
            <span className={styles.kicker}>Reusable components</span>
            <h2 className={styles.sectionTitle}>ENCOM modules for dashboards and containers</h2>
          </div>
          <p className={styles.sectionLead}>
            Each card is designed for dashboards, state summaries, and expandable
            diagnostics while preserving a restrained bundle profile.
          </p>
        </div>

        <div className={styles.cards}>
          {telemetryCards.map((card) => (
            <TronCard key={card.heading} {...card} />
          ))}
        </div>
      </section>

      <section className={styles.notes}>
        <EncomPanel label="Deployment" heading="Vercel-safe implementation" headingLevel={2}>
          <div className={styles.noteList}>
            <div className={styles.note}>
              Fonts are loaded through <code>next/font/google</code> to avoid manual asset
              handling and keep the pipeline App Router friendly.
            </div>
            <div className={styles.note}>
              The visual language relies on CSS variables, gradients, and keyframes instead
              of animation libraries.
            </div>
            <div className={styles.note}>
              Footer and WhatsApp access sit in the shared layout so every route inherits
              the system shell.
            </div>
          </div>
        </EncomPanel>

        <EncomPanel label="Scalability" heading="Extension points" headingLevel={2}>
          <div className={styles.noteList}>
            <div className={styles.note}>
              Panels already cover dashboard, card, and container use cases. The same shell
              can wrap modal content later without redesign.
            </div>
            <div className={styles.note}>
              Components are isolated by CSS modules, so moving this shell into an existing
              Next.js app won&apos;t bleed styles into backend or analytics code.
            </div>
            <div className={styles.note}>
              The current landing page is representative only. Existing routes can adopt the
              same primitives incrementally.
            </div>
          </div>
        </EncomPanel>
      </section>
    </div>
  );
}
