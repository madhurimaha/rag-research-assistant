import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter", display: "swap" });

export const metadata: Metadata = {
  title: "RAG Research Assistant",
  description:
    "Ask questions across uploaded documents and get answers grounded in the sources, with citations and a full retrieval trace.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="font-sans antialiased">
        {/* First tab stop: lets keyboard users jump past the source list straight to the
            question input, which is the primary task. */}
        <a
          href="#ask"
          className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-50 focus:bg-accent focus:text-white focus:shadow-lg"
        >
          Skip to question input
        </a>
        {children}
      </body>
    </html>
  );
}
