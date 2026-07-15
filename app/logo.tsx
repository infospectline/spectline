"use client";

import { useEffect, useRef, useState } from "react";

type LogoProps = {
  className?: string;
};

export default function Logo({ className = "" }: LogoProps) {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const [svgMarkup, setSvgMarkup] = useState("");

  useEffect(() => {
    let cancelled = false;

    const loadSvg = async () => {
      const res = await fetch("/images/launcher.svg");

      if (!res.ok) {
        console.error("[Logo] SVG load failed:", res.status);
        return;
      }

      const raw = await res.text();

      const cleaned = raw
        .replace(/<\?xml[\s\S]*?\?>/g, "")
        .replace(/<!DOCTYPE[\s\S]*?>/g, "")
        .replace(
          /<svg\b([^>]*)>/,
          '<svg$1 class="launcher-logo__svg" aria-hidden="true" focusable="false">'
        );

      if (!cancelled) {
        setSvgMarkup(cleaned);
      }
    };

    loadSvg();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const root = rootRef.current;
    if (!root || !svgMarkup) return;

    const svg = root.querySelector("svg");
    if (svg) {
      svg.removeAttribute("width");
      svg.removeAttribute("height");
    }

    const paths = Array.from(root.querySelectorAll<SVGPathElement>("path"));

    paths.forEach((path, index) => {
      const length = path.getTotalLength();

      path.classList.add("launcher-logo__path");
      path.style.setProperty("--path-length", `${length}`);
      path.style.setProperty("--path-index", `${index}`);
    });
  }, [svgMarkup]);

  return (
    <div
      ref={rootRef}
      className={["launcher-logo", className].filter(Boolean).join(" ")}
      aria-hidden="true"
      dangerouslySetInnerHTML={{ __html: svgMarkup }}
    />
  );
}