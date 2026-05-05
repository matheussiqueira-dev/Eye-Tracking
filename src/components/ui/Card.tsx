import type { HTMLAttributes, ReactNode } from "react";

import { cn } from "@/lib/utils";

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
}

export function Card({ children, className, ...props }: CardProps) {
  return (
    <div
      className={cn(
        "rounded-lg border border-white/10 bg-slate-950/70 shadow-2xl shadow-cyan-950/20 backdrop-blur",
        className,
      )}
      {...props}
    >
      {children}
    </div>
  );
}
