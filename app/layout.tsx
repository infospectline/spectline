import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
});

const mono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "Spectline",
  description: "Automatizovaný systém AI reportov pre firmy",
  metadataBase: new URL("http://localhost:3000"),
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
    return (
    <html lang="sk" suppressHydrationWarning>
      <body className={`${inter.variable} ${mono.variable} min-h-dvh`} suppressHydrationWarning>
        {children}
      </body>
    </html>
  );
}