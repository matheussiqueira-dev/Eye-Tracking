"use client";

import type { HTMLAttributes, ReactNode } from "react";

import { cx } from "../../lib/cx";
import styles from "./EncomPanel.module.css";

export type EncomPanelProps = HTMLAttributes<HTMLElement> & {
  as?: "section" | "article" | "div";
  label?: string;
  heading?: string;
  children: ReactNode;
};

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
            {heading ? <h2 className={styles.heading}>{heading}</h2> : null}
          </div>
        </header>
      ) : null}
      <div className={styles.body}>{children}</div>
    </Component>
  );
}
