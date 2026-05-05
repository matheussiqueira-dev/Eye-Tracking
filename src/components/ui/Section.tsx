import type { ReactNode } from "react";

interface SectionProps {
  children: ReactNode;
  eyebrow?: string;
  id?: string;
  title: string;
}

export function Section({ children, eyebrow, id, title }: SectionProps) {
  return (
    <section aria-labelledby={id} className="grid gap-6">
      <div className="grid gap-2">
        {eyebrow ? (
          <span className="text-xs font-semibold uppercase text-cyan-200">{eyebrow}</span>
        ) : null}
        <h2 className="text-balance text-2xl font-semibold text-white md:text-3xl" id={id}>
          {title}
        </h2>
      </div>
      {children}
    </section>
  );
}
