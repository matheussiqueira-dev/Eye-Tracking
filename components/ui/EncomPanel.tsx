"use client";

import type { HTMLAttributes, ReactNode } from "react";

import styles from "./EncomPanel.module.css";

export type EncomPanelProps = HTMLAttributes<HTMLElement> & {
  as?: "section" | "article" | "div";
  label?: string;
  heading?: string;
  children: ReactNode;
};

function cx(...names: Array<string | false | null | undefined>) {
  return names.filter(Boolean).join(" ");
}

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
