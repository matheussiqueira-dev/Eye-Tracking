"use client";

import type { FC } from "react";
import { useEffect } from "react";

interface ErrorPageProps {
  error: Error & { digest?: string };
  reset: () => void;
}

/**
 * Next.js App Router error boundary.
 *
 * Rendered whenever an unhandled error is thrown inside a route segment.
 * Provides a reset action that re-renders the segment without a full reload.
 */
const ErrorPage: FC<ErrorPageProps> = ({ error, reset }) => {
  useEffect(() => {
    // Log the error to an error-reporting service in production
    console.error("[ErrorBoundary]", error);
  }, [error]);

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
      <h2
        style={{
          fontFamily: "var(--font-heading, sans-serif)",
          fontSize: "clamp(1.4rem, 4vw, 2.2rem)",
          letterSpacing: "0.12em",
          textTransform: "uppercase",
          color: "var(--encom-neon, #00e5ff)",
          textShadow: "0 0 18px rgba(0, 229, 255, 0.4)",
          margin: 0,
        }}
      >
        System Error
      </h2>

      <p
        style={{
          maxWidth: "44ch",
          color: "var(--encom-text-secondary, #8fa6b2)",
        }}
      >
        An unexpected error occurred in the application. The system has logged
        the fault and is ready to reinitialise.
      </p>

      <button
        onClick={reset}
        style={{
          padding: "12px 28px",
          border: "1px solid #00e5ff",
          borderRadius: "999px",
          background: "transparent",
          color: "#00e5ff",
          fontFamily: "var(--font-label, sans-serif)",
          fontSize: "0.95rem",
          letterSpacing: "0.18em",
          textTransform: "uppercase",
          cursor: "pointer",
        }}
      >
        Reinitialise
      </button>
    </div>
  );
};

export default ErrorPage;
