"use client";

import type { HTMLAttributes, ReactNode } from "react";

function HeadingTag({
  level,
  className,
  children,
}: {
  level: 1 | 2 | 3 | 4;
  className?: string;
  children: ReactNode;
}) {
  const Tag = (
    { 1: "h1", 2: "h2", 3: "h3", 4: "h4" } as const
  )[level];
  return <Tag className={className}>{children}</Tag>;
}


import { cx } from "../../lib/cx";
import styles from "./EncomPanel.module.css";

export type EncomPanelProps = HTMLAttributes<HTMLElement> & {
  as?: "section" | "article" | "div";
  /** Semantic heading level rendered inside the panel. Defaults to h2. */
  headingLevel?: 1 | 2 | 3 | 4;
  label?: string;
  heading?: string;
  children: ReactNode;
};

/**
 * ENCOM-branded panel container with corner decorations and scan-line effect.
 *
 * Renders as section, article, or div (controlled by the as prop).
 * Optionally displays a label badge and heading in a header row.
 */
export function EncomPanel({
  as = "section",
  label,
  heading,
  children,
  className,
  ...props
}: EncomPanelProps) {
  const Component = as;

  return (
    <Component className={cx(styles.panel, "encom-scan-line", className)} {...props}>
      <span className={styles.cornerTop} aria-hidden="true" />
      <span className={styles.cornerBottom} aria-hidden="true" />
      {label || heading ? (
        <header className={styles.header}>
          <div className={styles.copy}>
            {label ? <span className={styles.label}>{label}</span> : null}
            {heading ? (
            <HeadingTag level={headingLevel} className={styles.heading}>
              {heading}
            </HeadingTag>
          ) : null}
          </div>
        </header>
      ) : null}
      <div className={styles.body}>{children}</div>
    </Component>
  );
}
