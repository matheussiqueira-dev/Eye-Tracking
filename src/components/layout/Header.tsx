import { Activity, GitBranch } from "lucide-react";
import Link from "next/link";

import { APP_NAME } from "@/lib/constants";

export function Header() {
  return (
    <header className="sticky top-0 z-30 border-b border-white/10 bg-slate-950/80 backdrop-blur-xl">
      <div className="mx-auto flex w-full max-w-7xl items-center justify-between gap-4 px-4 py-4 sm:px-6 lg:px-8">
        <Link className="flex items-center gap-3 text-white" href="/#dashboard">
          <span className="flex size-10 items-center justify-center rounded-lg border border-cyan-300/30 bg-cyan-300/10 text-cyan-200">
            <Activity aria-hidden="true" size={20} />
          </span>
          <span className="grid">
            <strong className="text-sm font-semibold">{APP_NAME}</strong>
            <span className="hidden text-xs text-slate-400 sm:block">
              Webcam UX analytics prototype
            </span>
          </span>
        </Link>

        <a
          className="inline-flex items-center gap-2 rounded-lg border border-white/10 px-3 py-2 text-sm text-slate-200 transition hover:bg-white/8"
          href="https://github.com/matheussiqueira-dev/Eye-Tracking"
          rel="noopener noreferrer"
          target="_blank"
        >
          <GitBranch aria-hidden="true" size={16} />
          GitHub
        </a>
      </div>
    </header>
  );
}
