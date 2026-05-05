import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import type { ReactNode } from "react";

import { Footer } from "@/components/layout/Footer";
import { Header } from "@/components/layout/Header";
import { APP_NAME, AUTHOR_CREDIT, SITE_URL } from "@/lib/constants";

import "./globals.css";

const geistSans = Geist({
  subsets: ["latin"],
  variable: "--font-geist-sans",
});

const geistMono = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-geist-mono",
});

export const metadata: Metadata = {
  applicationName: APP_NAME,
  authors: [{ name: AUTHOR_CREDIT, url: "https://www.matheussiqueira.dev/" }],
  description:
    "Advanced UX analytics prototype that estimates attention from webcam face landmarks and renders real-time heatmaps.",
  metadataBase: new URL(SITE_URL),
  title: {
    default: APP_NAME,
    template: `%s | ${APP_NAME}`,
  },
};

export const viewport: Viewport = {
  initialScale: 1,
  themeColor: "#071013",
  width: "device-width",
};

interface RootLayoutProps {
  children: ReactNode;
}

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html className={`${geistSans.variable} ${geistMono.variable}`} lang="en">
      <body className="font-sans antialiased">
        <Header />
        {children}
        <Footer />
      </body>
    </html>
  );
}
