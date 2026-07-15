"use client";

import { useState } from "react";
import Android from "./android";
import Launcher from "./launcher";

export default function Home() {
  const [showLauncher, setShowLauncher] = useState(true);

  return (
    <div className="bg-[#0A0A0A] text-white">
      <Android glbUrl="/android.glb" />

      {showLauncher ? (
        <Launcher
          durationMs={4500}
          onDone={() => setShowLauncher(false)}
        />
      ) : null}
    </div>
  );
}