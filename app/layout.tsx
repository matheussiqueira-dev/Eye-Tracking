import type { Metadata } from "next";
import { Exo_2, Orbitron, Rajdhani } from "next/font/google";
import type { ReactNode } from "react";

import { BackgroundGrid } from "../components/system/BackgroundGrid";
import { Footer } from "../components/system/Footer";
import { WhatsAppButton } from "../components/system/WhatsAppButton";
import "../styles/encom-theme.css";

const orbitron = Orbitron({
  subsets: ["latin"],
  variable: "--font-orbitron",
  weight: ["700", "800"],
});

const rajdhani = Rajdhani({
  subsets: ["latin"],
  variable: "--font-rajdhani",
  weight: ["500", "600", "700"],
});

const exo2 = Exo_2({
  subsets: ["latin"],
  variable: "--font-exo-2",
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "ENCOM Eye Tracking Interface",
  description: "Tron Legacy inspired App Router interface for the Eye Tracking project.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: ReactNode;
}>) {
  return (
    <html lang="pt-BR">
      <body className={`${orbitron.variable} ${rajdhani.variable} ${exo2.variable}`}>
        <BackgroundGrid />
        <div className="encom-shell">
          <main className="encom-main">{children}</main>
          <Footer />
        </div>
        <WhatsAppButton />
      </body>
    </html>
  );
}
