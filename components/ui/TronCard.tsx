"use client";

import type { HTMLAttributes } from "react";

import styles from "./TronCard.module.css";

export type TronCardProps = HTMLAttributes<HTMLElement> & {
  eyebrow?: string;
  heading: string;
  description: string;
  thumbnailLabel?: string;
  tags?: string[];
  details?: string[];
  ctaHref?: string;
  ctaLabel?: string;
};

function cx(...names: Array<string | false | null | undefined>) {
  return names.filter(Boolean).join(" ");
}

export function TronCard({
  eyebrow,
  heading,
  description,
  thumbnailLabel = "ENCOM",
  tags,
  details,
  ctaHref,
  ctaLabel,
  className,
  ...props
}: TronCardProps) {
  return (
    <article className={cx(styles.card, className)} {...props}>
      <div className={styles.thumbnail} aria-hidden="true">
        <span className={styles.thumbnailLabel}>{thumbnailLabel}</span>
      </div>

      <div className={styles.header}>
        {eyebrow ? <span className={styles.eyebrow}>{eyebrow}</span> : null}
        <h3 className={styles.heading}>{heading}</h3>
        <p className={styles.description}>{description}</p>
      </div>

      {tags?.length ? (
        <div className={styles.meta}>
          {tags.map((tag) => (
            <span key={tag} className={styles.badge}>
              {tag}
            </span>
          ))}
        </div>
      ) : null}

      {details?.length ? (
        <details className={styles.details}>
          <summary>Expand diagnostics</summary>
          <ul className={styles.list}>
            {details.map((detail) => (
              <li key={detail}>{detail}</li>
            ))}
          </ul>
        </details>
      ) : null}

      {ctaHref && ctaLabel ? (
        <a className={styles.link} href={ctaHref}>
          {ctaLabel}
        </a>
      ) : null}
    </article>
  );
}
