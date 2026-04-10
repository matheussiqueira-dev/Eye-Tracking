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

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? "https://eye-tracking.vercel.app";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: "ENCOM Eye Tracking Interface",
    template: "%s | ENCOM Eye Tracking",
  },
  description:
    "Real-time eye tracking with a regular webcam — built with MediaPipe and OpenCV, presented through a Tron Legacy inspired interface.",
  openGraph: {
    type: "website",
    locale: "en_US",
    url: SITE_URL,
    siteName: "ENCOM Eye Tracking",
    title: "ENCOM Eye Tracking Interface",
    description:
      "Real-time eye tracking pipeline with calibration, heatmap, and gaze analytics.",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "ENCOM Eye Tracking dashboard",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "ENCOM Eye Tracking Interface",
    description:
      "Real-time eye tracking pipeline with calibration, heatmap, and gaze analytics.",
    images: ["/og-image.png"],
  },
  robots: {
    index: true,
    follow: true,
  },
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
