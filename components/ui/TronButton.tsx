"use client";

import type {
  AnchorHTMLAttributes,
  ButtonHTMLAttributes,
  ReactNode,
} from "react";

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

function cx(...names: Array<string | false | null | undefined>) {
  return names.filter(Boolean).join(" ");
}

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
