"use client";

import { useEffect, useState } from "react";

type LauncherProps = {
  onDone?: () => void;
  durationMs?: number;
};

const LOGO_TRACE_PATH = `
M247.0 260.9 L243.3 262.1 L239.3 265.8 L237.2 270.7 L237.2 274.8 L238.9 278.9 L241.7 282.1 L245.8 284.2 L251.1 284.6 L255.2 282.9 L258.4 280.1 L260.5 276.0 L260.9 271.5 L259.3 266.6 L256.4 263.3 L252.7 261.3 Z
M247.0 265.8 L251.9 266.2 L254.8 268.7 L256.0 271.5 L256.0 274.4 L254.8 276.8 L250.7 279.7 L245.4 278.9 L242.1 274.4 L242.9 269.1 Z
M187.4 189.9 L181.7 191.9 L176.4 196.4 L172.7 204.1 L172.7 211.9 L176.4 219.7 L182.1 224.6 L188.2 226.6 L195.2 226.2 L200.5 223.7 L205.8 218.4 L208.6 211.5 L208.6 205.0 L206.2 198.4 L201.3 193.1 L193.9 189.9 Z
M188.2 196.0 L193.1 196.0 L196.4 197.2 L200.5 200.9 L202.5 205.0 L202.9 209.5 L201.3 214.4 L197.6 218.4 L193.1 220.5 L188.2 220.5 L184.1 218.8 L180.1 214.8 L178.4 210.7 L178.4 205.8 L180.1 201.7 L182.9 198.4 Z
M188.2 169.9 L181.3 171.1 L173.9 173.9 L166.2 179.2 L161.3 184.5 L154.7 197.6 L153.5 203.7 L153.5 212.3 L157.6 225.8 L161.3 231.5 L166.6 237.2 L173.1 241.7 L179.6 244.6 L187.4 246.2 L193.9 246.2 L200.5 245.0 L209.0 241.3 L219.3 232.7 L222.5 228.2 L226.6 218.8 L227.8 213.1 L227.8 203.3 L225.4 193.9 L222.1 187.4 L214.4 178.4 L209.0 174.7 L201.7 171.5 L193.5 169.9 Z
M188.2 173.9 L196.0 174.3 L202.1 176.0 L212.7 182.5 L218.8 189.9 L221.7 195.6 L223.7 203.7 L223.7 212.7 L222.5 218.4 L219.7 225.0 L216.0 230.3 L207.4 237.6 L202.5 240.1 L194.3 242.1 L182.5 241.3 L176.8 239.3 L169.9 234.8 L161.7 225.0 L157.6 213.1 L158.4 199.2 L164.1 187.4 L168.6 182.5 L174.7 178.0 L181.7 175.2 Z
M198.4 63.3 L184.5 69.4 L172.3 76.8 L162.9 84.1 L153.1 93.9 L144.5 105.3 L138.8 115.1 L132.3 129.8 L128.2 142.5 L124.9 158.4 L123.7 169.9 L123.7 193.1 L126.6 213.9 L131.5 231.5 L138.8 246.6 L146.6 257.2 L153.9 264.6 L162.1 270.7 L160.1 299.7 L154.7 343.0 L147.4 379.3 L123.7 406.7 L116.0 412.0 L102.5 417.7 L101.7 419.7 L104.9 420.5 L121.7 417.7 L136.0 417.7 L159.6 421.0 L180.1 427.1 L196.8 434.8 L212.3 444.6 L227.0 456.1 L254.0 456.1 L254.0 453.6 L249.9 446.7 L231.1 421.0 L214.4 403.8 L198.0 390.7 L196.4 370.7 L193.9 355.6 L198.4 353.2 L228.2 345.4 L265.0 455.2 L265.8 456.1 L271.5 456.1 L247.0 382.6 L224.2 309.5 L219.3 303.8 L212.3 300.9 L205.0 301.7 L198.8 305.8 L196.4 309.5 L195.2 313.6 L195.6 320.5 L198.4 325.8 L202.1 329.1 L207.0 331.1 L214.8 330.7 L222.5 325.8 L226.6 339.7 L193.5 349.1 L192.3 347.5 L189.9 334.8 L183.3 311.1 L167.4 266.6 L152.7 254.0 L144.1 242.5 L137.6 229.1 L133.5 215.2 L130.7 197.6 L129.8 175.2 L131.9 156.0 L136.0 139.2 L144.5 118.8 L154.3 103.3 L168.2 88.6 L181.3 79.6 L187.0 93.1 L187.0 95.1 L180.1 100.8 L174.3 107.4 L170.7 113.9 L169.0 120.4 L169.9 131.1 L179.6 151.5 L179.6 153.1 L166.6 158.8 L158.0 165.0 L150.7 172.7 L144.1 183.3 L140.9 191.9 L138.8 201.7 L138.8 214.4 L140.5 222.5 L143.7 231.9 L149.8 242.5 L159.2 252.7 L167.4 258.4 L182.9 264.6 L195.2 298.5 L197.6 298.5 L198.8 296.4 L187.8 265.8 L188.6 265.0 L195.6 265.4 L207.0 263.8 L221.3 293.2 L225.0 298.9 L230.3 304.2 L282.1 331.1 L305.0 344.6 L300.5 350.7 L296.8 358.5 L294.8 367.5 L294.4 374.0 L295.2 385.8 L301.3 425.0 L300.9 426.7 L291.5 425.4 L290.3 423.4 L285.4 372.0 L254.8 363.8 L252.7 365.4 L253.1 367.5 L254.8 369.9 L278.5 375.6 L280.9 377.3 L285.4 427.5 L289.5 456.1 L294.8 455.7 L291.5 434.4 L292.3 431.2 L305.8 434.0 L338.1 454.8 L344.6 455.7 L346.2 454.8 L346.2 453.2 L307.9 427.5 L301.7 387.1 L300.9 369.5 L303.8 358.5 L308.3 350.7 L310.7 348.3 L312.8 348.3 L321.7 351.9 L342.1 356.8 L356.8 359.3 L367.1 359.3 L371.5 358.1 L377.7 354.4 L381.8 349.9 L384.2 345.0 L385.0 339.3 L382.6 327.9 L382.6 322.6 L385.0 316.8 L389.9 312.3 L391.6 309.5 L392.0 303.4 L390.3 298.1 L395.2 293.2 L396.9 289.1 L396.9 286.2 L394.0 279.3 L393.2 273.6 L395.6 270.7 L405.8 266.6 L410.7 262.1 L412.8 258.0 L413.2 252.7 L411.2 247.8 L395.6 225.8 L387.5 212.3 L387.5 204.6 L392.8 191.9 L393.2 182.9 L386.7 152.3 L379.7 129.8 L374.8 118.8 L368.3 107.8 L361.7 99.2 L353.2 90.2 L343.8 82.5 L329.9 73.5 L309.5 64.1 L286.2 57.6 L261.7 54.3 L231.9 55.1 L214.4 58.4 Z
M127.0 412.0 L150.7 385.4 L153.1 384.6 L164.1 384.6 L173.5 386.7 L181.3 389.5 L191.9 395.2 L210.7 409.5 L228.2 427.9 L238.9 441.8 L243.3 449.5 L229.5 449.9 L214.4 438.5 L202.1 430.7 L188.2 423.8 L172.3 418.1 L151.5 413.6 Z
M208.6 306.2 L215.2 307.4 L218.4 310.7 L220.1 314.4 L220.1 318.1 L218.4 321.7 L215.6 324.6 L211.1 326.2 L207.0 325.8 L204.1 324.2 L201.3 320.9 L200.1 316.8 L202.1 310.3 L204.6 307.9 Z
M167.8 286.2 L176.4 309.9 L183.3 334.8 L189.0 363.8 L191.5 384.6 L191.1 387.1 L180.9 382.2 L171.9 379.3 L154.7 377.7 L160.5 348.3 L165.4 309.5 L167.0 287.8 Z
M213.9 261.3 L227.4 253.1 L232.3 248.7 L241.3 237.6 L268.7 241.3 L272.3 242.5 L309.9 274.0 L300.5 315.6 L301.3 319.7 L343.8 349.9 L331.9 347.9 L317.2 343.4 L281.7 323.4 L236.0 300.1 L229.5 294.0 L227.0 290.3 L213.9 263.8 Z
M333.2 200.9 L345.4 197.2 L353.2 196.8 L358.5 197.6 L361.3 198.8 L362.6 200.5 L361.7 208.2 L360.5 209.9 L348.7 209.5 L340.1 205.8 Z
M249.1 215.6 L279.7 203.3 L305.8 201.7 L324.6 202.1 L344.6 213.1 L349.9 214.4 L364.2 213.9 L365.4 213.1 L367.1 198.8 L365.8 196.4 L362.2 193.9 L355.6 192.3 L346.2 192.3 L336.4 194.3 L326.2 198.0 L317.2 197.6 L316.8 196.4 L320.5 191.1 L328.7 183.3 L335.6 179.6 L342.1 178.0 L358.9 178.8 L385.0 184.1 L386.7 185.4 L386.7 191.1 L381.3 204.6 L380.9 209.5 L381.8 213.9 L383.4 218.0 L407.1 252.7 L407.1 256.8 L403.8 260.9 L398.1 263.8 L396.0 263.8 L393.6 260.9 L390.7 260.9 L377.7 264.6 L375.6 263.8 L374.0 262.1 L373.6 259.7 L372.0 258.9 L370.7 262.1 L372.8 266.2 L376.9 268.2 L384.2 267.8 L387.9 270.3 L388.3 279.7 L391.6 287.8 L389.1 290.7 L380.9 289.5 L367.9 294.0 L358.1 295.2 L351.9 301.3 L352.8 303.0 L354.0 303.0 L360.9 298.5 L383.0 298.1 L385.4 300.9 L386.7 306.6 L383.4 309.9 L372.0 307.0 L369.1 307.4 L369.1 309.1 L370.7 310.7 L378.9 314.8 L376.4 322.1 L376.4 327.5 L378.9 336.8 L378.9 341.3 L376.9 346.2 L372.8 350.3 L365.0 353.2 L356.0 352.4 L305.4 316.8 L314.8 274.4 L314.4 271.5 L274.4 238.4 L271.1 237.2 L247.4 234.0 L246.2 231.9 L247.8 216.8 Z
M185.0 158.8 L199.2 158.4 L211.5 161.7 L220.9 167.0 L229.1 174.3 L233.1 179.6 L238.0 189.4 L240.9 200.1 L241.3 214.4 L240.1 221.3 L237.6 228.6 L231.9 238.9 L225.8 245.8 L218.0 251.9 L209.9 256.0 L200.9 258.4 L189.9 258.9 L181.3 257.2 L172.7 254.0 L164.5 248.7 L156.8 240.9 L151.9 234.0 L148.2 226.2 L145.4 215.2 L144.9 204.6 L146.2 196.8 L148.6 189.0 L153.9 179.2 L163.3 169.0 L169.0 165.0 L177.6 160.9 Z
M174.3 125.3 L174.7 119.6 L176.8 113.9 L180.5 108.6 L185.8 103.3 L192.3 98.4 L206.6 96.8 L232.7 97.6 L254.4 101.3 L280.1 109.4 L289.9 113.9 L295.6 118.0 L294.4 120.9 L266.2 159.6 L237.2 173.1 L227.0 163.3 L213.9 155.6 L199.7 151.9 L186.6 151.9 L177.2 134.7 Z
M187.8 75.9 L201.3 69.4 L217.6 64.5 L238.9 61.2 L258.0 60.8 L275.2 62.5 L292.3 65.7 L308.7 71.0 L325.4 78.8 L337.7 86.6 L347.0 93.9 L357.7 104.9 L365.0 115.1 L370.7 125.8 L376.9 141.7 L383.4 166.6 L385.4 178.0 L384.2 179.2 L381.8 179.2 L363.4 175.2 L350.7 173.5 L338.1 174.3 L331.1 176.8 L324.6 180.9 L319.7 185.4 L312.3 196.0 L310.3 197.6 L280.1 198.8 L248.2 210.7 L246.6 196.0 L240.5 178.4 L245.0 175.6 L269.5 164.5 L302.1 120.0 L303.0 116.4 L296.8 111.1 L285.0 104.9 L269.9 99.2 L250.7 94.3 L225.4 91.0 L204.6 91.0 L193.9 92.3 L191.9 90.6 L187.4 78.4 Z
`;

type AnimatedLauncherLogoProps = {
  progress: number;
};

const LOGO_DASH_LENGTH = 1000;

function getLogoDashOffset(progress: number) {
  const safeProgress = Math.min(100, Math.max(0, progress));
  return LOGO_DASH_LENGTH - safeProgress * 10;
}

export default function Launcher({
  onDone,
  durationMs = 4500,
}: LauncherProps) {
  const [progress, setProgress] = useState(0);
  const [leaving, setLeaving] = useState(false);

  useEffect(() => {
    let frame = 0;
    let doneTimer: ReturnType<typeof setTimeout> | null = null;

    const start = performance.now();

    const animate = (now: number) => {
      const elapsed = now - start;
      const p = Math.min(100, Math.round((elapsed / durationMs) * 100));

      setProgress(p);

      if (p < 100) {
        frame = requestAnimationFrame(animate);
      } else {
        setLeaving(true);

        doneTimer = setTimeout(() => {
          onDone?.();
        }, 650);
      }
    };

    frame = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(frame);
      if (doneTimer) clearTimeout(doneTimer);
    };
  }, [durationMs, onDone]);

  return (
    <div className={["launcher", leaving ? "launcher--leaving" : ""].join(" ")}>
      <div className="launcher__inner">
        <div className="launcher__figure_wrap" aria-hidden="true">
          <AnimatedLauncherLogo progress={progress} />
        </div>

        <div className="launcher__content">
          <div className="launcher__label">Loading system</div>

          <div className="launcher__progress_text">{progress}%</div>

          <div
            className="launcher__bar"
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={progress}
            aria-label="Načítavanie"
          >
            <div
              className="launcher__bar_fill"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      </div>

      <style>{`
        .launcher__figure_wrap {
          animation: none !important;
          transform: none !important;
          width: min(300px, 72vw);
          margin: 0 auto 28px;
        }

        .launcher-ai-logo {
          display: block;
          width: 100%;
          height: auto;
          overflow: visible;
        }

        .launcher-ai-logo__shape {
          fill: #3E3E3F;
          fill-rule: evenodd;
          clip-rule: evenodd;
          opacity: 0.98;
        }

        .launcher-ai-logo__reveal {
          fill: none;
          stroke: #3E3E3F;
          stroke-linecap: round;
          stroke-linejoin: round;
          stroke-dasharray: 1000;
          stroke-dashoffset: 1000;
        }

        .launcher-ai-logo__reveal--outer {
          stroke-width: 58;
        }

        .launcher-ai-logo__reveal--front {
          stroke-width: 58;
        }

        .launcher-ai-logo__reveal--details {
          stroke-width: 70;
        }
      `}</style>
    </div>
  );
}

function AnimatedLauncherLogo({ progress }: AnimatedLauncherLogoProps) {
  const dashOffset = getLogoDashOffset(progress);

  return (
    <svg
      className="launcher-ai-logo"
      viewBox="0 0 512 512"
      role="img"
      aria-label="Animované logo"
    >
      <defs>
        <path
          id="launcherLogoTrace"
          d={LOGO_TRACE_PATH}
          fillRule="evenodd"
          clipRule="evenodd"
        />

        <mask
          id="launcherLogoRevealMask"
          maskUnits="userSpaceOnUse"
          x="0"
          y="0"
          width="512"
          height="512"
        >
          <rect width="512" height="512" fill="black" />

          {/* 1. Starting point: bottom/rear silhouette upwards + lower arm */}
          <path
            className="launcher-ai-logo__reveal launcher-ai-logo__reveal--outer"
            pathLength={1000}
            style={{ strokeDashoffset: dashOffset }}
            d="
              M104 420

              C122 416 139 416 156 418
              C178 421 200 430 229 450
              C237 455 247 456 255 455
              C246 439 232 420 214 403
              C198 390 179 381 153 384
              C139 398 126 412 104 420

              C130 416 146 401 153 381
              C160 355 164 318 164 270
              C140 255 128 223 128 185
              C130 122 169 78 216 62
              C274 42 341 69 374 119
              C388 142 393 165 390 184
            "
          />

          {/* 2. Starting point: neck, jaw, face, front part of the head + connection of the lower inner part */}
          <path
            className="launcher-ai-logo__reveal launcher-ai-logo__reveal--front"
            pathLength={1000}
            style={{ strokeDashoffset: dashOffset }}
            d="
              M343 455
              C325 445 307 429 301 406
              C293 378 294 358 309 345
              C326 352 352 359 369 356
              C381 353 387 342 381 329
              C378 320 382 314 389 309
              C397 302 390 292 394 287
              C390 276 395 270 406 265
              C418 259 411 248 399 231
              C389 217 386 208 391 192
              C394 186 392 181 386 180
              C365 176 341 174 327 187
              C317 197 309 200 294 200
              C276 201 260 207 247 213

              C246 226 245 242 249 258
              C260 265 276 279 314 274
              C311 289 307 304 305 318
              C319 329 337 342 356 353

              C338 351 321 347 306 341
              C281 327 255 312 232 300
              C224 292 217 276 209 259

              C205 279 208 298 217 314
              C228 332 244 349 255 370
              C263 392 266 419 266 455

              C260 430 253 405 245 384
              C236 361 226 340 217 323
              C211 307 208 291 209 259
            "
          />

          {/* 3. Starting point: circles, panels, and small technical details */}
          <path
            className="launcher-ai-logo__reveal launcher-ai-logo__reveal--details"
            pathLength={1000}
            style={{ strokeDashoffset: dashOffset }}
            d="
              M213 316
              A19 19 0 1 1 212 315
              C205 292 198 267 192 247
              A44 44 0 1 1 193 246
              A64 64 0 1 1 193 246
              C209 262 231 258 249 272
              A17 17 0 1 1 248 271
              C244 252 245 229 249 214
              C274 202 304 198 324 202
              C340 196 356 193 364 201
              C352 184 334 176 320 190
              C301 197 272 199 244 211
              C239 183 218 158 187 156
              C180 145 173 130 173 122
              C174 106 188 96 205 94
              C245 92 281 104 301 119
              C285 144 265 166 237 176
              C231 165 213 157 187 156
            "
          />
        </mask>
      </defs>

      <use
        href="#launcherLogoTrace"
        className="launcher-ai-logo__shape"
        fillRule="evenodd"
        clipRule="evenodd"
        mask="url(#launcherLogoRevealMask)"
      />
    </svg>
  );
}