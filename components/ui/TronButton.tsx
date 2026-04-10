"use client";

import type {
  AnchorHTMLAttributes,
  ButtonHTMLAttributes,
  ReactNode,
} from "react";

import { cx } from "../../lib/cx";
import styles from "./TronButton.module.css";

type CommonProps = {
  children: ReactNode;
  className?: string;
};

type ButtonProps = CommonProps &
  ButtonHTMLAttributes<HTMLButtonElement> & {
    href?: undefined;
  };

type LinkProps = CommonProps &
  AnchorHTMLAttributes<HTMLAnchorElement> & {
    href: string;
  };

export type TronButtonProps = ButtonProps | LinkProps;

/**
 * Polymorphic Tron-styled button/link.
 *
 * Renders as an anchor when href is provided, otherwise as a button.
 * Includes a sweep shimmer animation on hover via a CSS pseudo-element.
 */
export function TronButton(props: TronButtonProps) {
  const className = cx(styles.button, props.className);

  if ("href" in props && props.href) {
    const { children, className: _className, href, ...rest } = props;

    return (
      <a className={className} href={href} {...rest}>
        <span className={styles.label}>{children}</span>
      </a>
    );
  }

  const buttonProps = props as ButtonProps;
  const {
    children,
    className: _className,
    ...rest
  } = buttonProps;
  const type = buttonProps.type ?? "button";

  return (
    <button className={className} type={type} {...rest}>
      <span className={styles.label}>{children}</span>
    </button>
  );
}
