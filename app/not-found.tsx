import Link from "next/link";

/**
 * Custom 404 page matching the ENCOM visual language.
 */
export default function NotFoundPage() {
  return (
    <div
      style={{
        minHeight: "60vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: "24px",
        fontFamily: "var(--font-body, sans-serif)",
        color: "var(--encom-text-primary, #e6ffff)",
        textAlign: "center",
        padding: "0 24px",
      }}
    >
      <h1
        style={{
          fontFamily: "var(--font-heading, sans-serif)",
          fontSize: "clamp(3rem, 12vw, 8rem)",
          letterSpacing: "0.12em",
          color: "var(--encom-neon, #00e5ff)",
          textShadow: "0 0 28px rgba(0, 229, 255, 0.5)",
          margin: 0,
          lineHeight: 1,
        }}
      >
        404
      </h1>

      <h2
        style={{
          fontFamily: "var(--font-heading, sans-serif)",
          fontSize: "clamp(1rem, 3vw, 1.6rem)",
          letterSpacing: "0.12em",
          textTransform: "uppercase",
          margin: 0,
        }}
      >
        Sector Not Found
      </h2>

      <p
        style={{
          maxWidth: "44ch",
          color: "var(--encom-text-secondary, #8fa6b2)",
        }}
      >
        The requested grid sector does not exist or has been decommissioned.
      </p>

      <Link
        href="/"
        style={{
          padding: "12px 28px",
          border: "1px solid #00e5ff",
          borderRadius: "999px",
          color: "#00e5ff",
          fontFamily: "var(--font-label, sans-serif)",
          fontSize: "0.95rem",
          letterSpacing: "0.18em",
          textTransform: "uppercase",
        }}
      >
        Return to Grid
      </Link>
    </div>
  );
}
