import Link from "next/link";

export default function NotFound() {
  return (
    <main className="mx-auto flex min-h-[70vh] w-full max-w-3xl flex-col items-start justify-center gap-6 px-4 py-16 sm:px-6 lg:px-8">
      <p className="font-mono text-sm uppercase text-cyan-200">404</p>
      <h1 className="text-3xl font-semibold text-white sm:text-5xl">Analysis route not found</h1>
      <p className="max-w-xl text-base leading-7 text-slate-300">
        The requested view is outside the current UX intelligence workspace.
      </p>
      <Link
        className="inline-flex min-h-11 items-center justify-center rounded-lg border border-cyan-300/60 bg-cyan-300 px-5 py-2 text-sm font-semibold text-slate-950 transition hover:bg-cyan-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-300"
        href="/"
      >
        Return to dashboard
      </Link>
    </main>
  );
}
