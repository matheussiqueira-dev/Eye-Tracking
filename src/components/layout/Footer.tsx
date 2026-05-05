import { AUTHOR_CREDIT, AUTHOR_URL } from "@/lib/constants";

export function Footer() {
  return (
    <footer className="border-t border-white/10 bg-slate-950">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-2 px-4 py-8 text-sm text-slate-400 sm:px-6 lg:px-8">
        <p>
          Desenvolvido por Matheus Siqueira -{" "}
          <a
            className="font-medium text-cyan-200 underline-offset-4 hover:underline"
            href={AUTHOR_URL}
            rel="noopener noreferrer"
            target="_blank"
          >
            www.matheussiqueira.dev
          </a>
        </p>
        <p className="max-w-3xl">
          {AUTHOR_CREDIT}. Este sistema e um prototipo avancado de UX analytics e nao deve ser
          interpretado como medicao clinica ou cientifica absoluta.
        </p>
      </div>
    </footer>
  );
}
