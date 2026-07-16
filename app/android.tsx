"use client";

import Image from "next/image";
import { useRouter } from "next/navigation";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  Home as HomeIcon,
  Settings,
  Monitor,
  Images,
  Star,
  Mail,
  LogIn,
  UserPlus,
  Menu,
  type LucideIcon,
} from "lucide-react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { DRACOLoader } from "three/examples/jsm/loaders/DRACOLoader.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { OutlinePass } from "three/examples/jsm/postprocessing/OutlinePass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass.js";
import sk from "@/lib/json/sk.json";
import en from "@/lib/json/en.json";

type AndroidBasicProps = {
  glbUrl?: string;
  background?: number | string;
};

type Lang = "sk" | "en";
type SiteCopy = typeof sk;
type NavLabelKey = keyof SiteCopy["nav"];

const COPY_BY_LANG: Record<Lang, SiteCopy> = {
  sk,
  en,
};

const SECTION_IDS = ["home", "process", "pricing", "demo", "reviews", "contact", "footer"] as const;
const SECTION_COUNT = SECTION_IDS.length;
const CONTACT_INDEX = 5;
const CONTACT_T = CONTACT_INDEX + 0.5;

type RadialState = { open: boolean; x: number; y: number };
type SectionId = (typeof SECTION_IDS)[number];

type ContactFormStatus =
  | "idle"
  | "submitting"
  | "success"
  | "error";

type ContactStatusCopy = {
  submitting: string;
  success: string;
  error: string;
};

type RadialItem =
  | { key: string; kind: "section"; id: SectionId; label: string; icon: LucideIcon }
  | { key: string; kind: "route"; href: string; label: string; icon: LucideIcon };

type RadialItemConfig =
  | { key: string; kind: "section"; id: SectionId; labelKey: NavLabelKey; icon: LucideIcon }
  | { key: string; kind: "route"; href: string; labelKey: NavLabelKey; icon: LucideIcon };

const RADIAL_ITEM_CONFIG: readonly RadialItemConfig[] = [
  { key: "home", kind: "section", id: "home", labelKey: "home", icon: HomeIcon },
  { key: "process", kind: "section", id: "process", labelKey: "process", icon: Settings },
  { key: "pricing", kind: "section", id: "pricing", labelKey: "pricing", icon: Monitor },
  { key: "demo", kind: "section", id: "demo", labelKey: "security", icon: Images },
  { key: "reviews", kind: "section", id: "reviews", labelKey: "reviews", icon: Star },
  { key: "contact", kind: "section", id: "contact", labelKey: "contact", icon: Mail },
  { key: "login", kind: "route", href: "/login", labelKey: "login", icon: LogIn },
  { key: "signup", kind: "route", href: "/signup", labelKey: "signup", icon: UserPlus },
] as const;

function RadialMenu({
  open,
  x,
  y,
  items,
  onPick,
  onClose,
}: {
  open: boolean;
  x: number;
  y: number;
  items: readonly RadialItem[];
  onPick: (item: RadialItem) => void;
  onClose: () => void;
}) {
  if (!open) return null;

  const radius = 104;

  return (
    <div
      className="radial-menu__overlay fixed inset-0 z-[9999] pointer-events-auto"
      onPointerDown={(e) => {
        const t = e.target as HTMLElement | null;
        if (!t) return;
        if (!t.closest("[data-radial-menu]")) onClose();
      }}
    >
      <div
        data-radial-menu
        className="absolute pointer-events-auto"
        style={{ left: x, top: y, transform: "translate(-50%, -50%)" }}
      >
        <div className="relative h-0 w-0">
          <div className="radial-menu__center absolute left-0 top-0 -translate-x-1/2 -translate-y-1/2 h-3 w-3 rounded-full border" />

          {items.map((it, idx) => {
            const a = (idx / items.length) * Math.PI * 2 - Math.PI / 2;
            const dx = Math.cos(a) * radius;
            const dy = Math.sin(a) * radius;
            const Icon = it.icon;

            return (
              <button
                key={it.key}
                type="button"
                aria-label={it.label}
                title={it.label}
                className="radial-menu__button absolute -translate-x-1/2 -translate-y-1/2 h-14 w-14 rounded-full
                          border backdrop-blur
                          grid place-items-center
                          transition-all duration-200
                          hover:scale-110"
                style={{ left: dx, top: dy }}
                onPointerDown={(e) => e.stopPropagation()}
                onClick={() => onPick(it)}
              >
                <Icon className="h-6 w-6 text-current" />
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export default function AndroidBasic({
  glbUrl = "/android.glb",
  background = 0x0b0b0c,
}: AndroidBasicProps) {
  const router = useRouter();
  const [lang, setLang] = useState<Lang>("sk");
  const copy = COPY_BY_LANG[lang];

  const contactStatusCopy = (
    copy.contact as SiteCopy["contact"] & {
      status?: ContactStatusCopy;
    }
  ).status;

  const documentHref = (
    document: "terms" | "gdpr" | "complaint" | "invoice"
  ) => `/documents/${document}_${lang}.pdf`;

  const processSteps = copy.process.steps;
  const pricingPlans = copy.pricing.plans;
  const reviews = copy.reviews.items;

  const radialItems = useMemo<readonly RadialItem[]>(() => {
    return RADIAL_ITEM_CONFIG.map((it) => {
      if (it.kind === "section") {
        return {
          key: it.key,
          kind: it.kind,
          id: it.id,
          label: copy.nav[it.labelKey],
          icon: it.icon,
        };
      }

      return {
        key: it.key,
        kind: it.kind,
        href: it.href,
        label: copy.nav[it.labelKey],
        icon: it.icon,
      };
    });
  }, [copy]);

  const [radial, setRadial] = useState<RadialState>({ open: false, x: 0, y: 0 });
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [ready, setReady] = useState(false);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      const saved = window.localStorage.getItem("spectline-lang");
      if (saved === "sk" || saved === "en") setLang(saved);
      setHydrated(true);
    });

    return () => window.cancelAnimationFrame(frame);
  }, []);

  const toggleLang = () => {
    setLang((current) => {
      const next = current === "sk" ? "en" : "sk";
      window.localStorage.setItem("spectline-lang", next);
      return next;
    });
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setRadial((s) => ({ ...s, open: false }));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const scrollToSection = (id: SectionId) => {
    const idx = SECTION_IDS.indexOf(id);
    if (idx < 0) return;

    const vh = Math.max(1, window.innerHeight);

    const FADE = 0.25;
    const sweet = Math.max(0, Math.min(1, 1 - FADE - 0.02));

    window.scrollTo({ top: (idx + sweet) * vh, behavior: "smooth" });
    history.replaceState(null, "", `#${id}`);
  };

  const openRadialAt = (clientX: number, clientY: number) => {
    const safe = 104 + 28 + 12;
    const x = Math.min(window.innerWidth - safe, Math.max(safe, clientX));
    const y = Math.min(window.innerHeight - safe, Math.max(safe, clientY));
    setRadial({ open: true, x, y });
  };

  const openMobileRadial = () => {
    openRadialAt(
      window.innerWidth / 2,
      window.innerHeight / 2
    );
  };

  const onBackgroundPointerDownCapture = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.pointerType !== "mouse") return;

    if (!window.matchMedia("(hover: hover) and (pointer: fine)").matches) return;

    if (e.button !== 0) return;

    const t = e.target as HTMLElement | null;
    if (!t) return;

    const blockedRadialSelector = [
      "[data-radial-menu]",
      "[data-no-radial]",

      "a",
      "button",
      "input",
      "textarea",
      "select",
      "label",
      "summary",
      "details",
      "[role='button']",

      ".rounded-2xl",
      ".first_section__panel",
      ".first_section__card",
      ".first_section__card_wrap",
    ].join(",");

    if (t.closest(blockedRadialSelector)) return;
    if (t.isContentEditable) return;

    openRadialAt(e.clientX, e.clientY);
  };

  const [isMobile, setIsMobile] = useState(false);
  const [isMobileNav, setIsMobileNav] = useState(false);
  const isMobileRef = useRef(false);
  const [scrollT, setScrollT] = useState(0);
  const [isProcessStacked, setIsProcessStacked] = useState(false);
  const scrollTRef = useRef(0);
  const scrollPRef = useRef(0);

  const [activePlan, setActivePlan] = useState(0);
  const [hoveredPlan, setHoveredPlan] = useState<number | null>(null);

  const [isPricingAccordion, setIsPricingAccordion] = useState(false);
  const [isPricingSlider, setIsPricingSlider] = useState(false);
  const [openPlan, setOpenPlan] = useState<number | null>(0);
  const [openProcessStep, setOpenProcessStep] = useState<number | null>(0);

  const maxPlanIndex = Math.max(0, pricingPlans.length - 1);
  const displayedActivePlan = Math.min(activePlan, maxPlanIndex);
  const displayedOpenPlan = isPricingAccordion
    ? Math.min(openPlan ?? 0, maxPlanIndex)
    : null;

  const swipeRef = useRef<{ x: number; y: number; id: number | null; fired: boolean }>({
    x: 0,
    y: 0,
    id: null,
    fired: false,
  });

  const [pricingSlideDir, setPricingSlideDir] = useState<1 | -1>(1);

  const swipePlan = (dir: 1 | -1) => {
    setPricingSlideDir(dir);
    setOpenPlan((p) => {
      const count = pricingPlans.length;
      if (!count) return 0;

      const cur = p ?? 0;
      return (cur + (dir === 1 ? 1 : -1) + count) % count;
    });
  };

  const [contactLock, setContactLock] = useState(false);
  const contactLockRef = useRef(false);
  const lockScrollYRef = useRef(0);
  const contactFormRef = useRef<HTMLFormElement | null>(null);

  const [contactFormStatus, setContactFormStatus] =
  useState<ContactFormStatus>("idle");

  const handleContactSubmit = async (
    event: React.FormEvent<HTMLFormElement>
  ) => {
    event.preventDefault();

    if (contactFormStatus === "submitting") return;

    const form = event.currentTarget;
    const formData = new FormData(form);

    setContactFormStatus("submitting");

    try {
      const response = await fetch("/api/contact", {
        method: "POST",
        body: formData,
      });

      const result = (await response.json().catch(() => null)) as
        | {
            success?: boolean;
            error?: string;
          }
        | null;

      if (!response.ok || result?.success !== true) {
        throw new Error(
          result?.error ||
            `Contact form request failed with status ${response.status}.`
        );
      }

      form.reset();
      setContactFormStatus("success");
    } catch (error) {
      console.error("Contact form submission failed.", error);
      setContactFormStatus("error");
    }
  };

  useEffect(() => {
    contactLockRef.current = contactLock;
  }, [contactLock]);

  const [reviewsPaused, setReviewsPaused] = useState(false);
  const [activeSecurityIndex, setActiveSecurityIndex] = useState(1);
  const securityDragRef = useRef<{
    startX: number;
    startY: number;
    pointerId: number | null;
    fired: boolean;
  }>({
    startX: 0,
    startY: 0,
    pointerId: null,
    fired: false,
  });

  useEffect(() => {
    const mq = window.matchMedia("(max-width: 900px)");
    const mqPortrait = window.matchMedia("(orientation: portrait)");

    const apply = () => {
      const smallW = mq.matches;
      const portrait = mqPortrait.matches;
      const smallH = window.innerHeight < 740;

      setIsPricingAccordion(smallW);
      setIsPricingSlider(smallW && portrait && smallH);
    };

    apply();
    mq.addEventListener?.("change", apply);
    mqPortrait.addEventListener?.("change", apply);
    window.addEventListener("resize", apply);

    return () => {
      mq.removeEventListener?.("change", apply);
      mqPortrait.removeEventListener?.("change", apply);
      window.removeEventListener("resize", apply);
    };
  }, []);

  useEffect(() => {
    if (isPricingAccordion) return;
    if (hoveredPlan !== null) return;
    if (pricingPlans.length <= 1) return;

    const id = window.setInterval(() => {
      setActivePlan((p) => (p + 1) % pricingPlans.length);
    }, 2200);

    return () => window.clearInterval(id);
  }, [hoveredPlan, isPricingAccordion, pricingPlans.length]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let disposed = false;
    let raf = 0;

    let model: THREE.Object3D | null = null;
    let modelMixer: THREE.AnimationMixer | null = null;
    let camMixer: THREE.AnimationMixer | null = null;
    let camClip: THREE.AnimationClip | null = null;
    let camAction: THREE.AnimationAction | null = null;
    const clock = new THREE.Clock();

    const autoCam = window.matchMedia("(hover: none), (pointer: coarse), (max-width: 900px)").matches;

    let autoP = 0;
    let autoDir: 1 | -1 = 1;
    let autoHold = 0;

    const AUTO_SPEED = 0.02;
    const AUTO_HOLD_SEC = 0.7;

    const defaultCamera = new THREE.PerspectiveCamera(45, 1, 0.1, 2000);
    defaultCamera.position.set(0, 1.2, 3.2);

    let activeCamera: THREE.Camera = defaultCamera;

    const LINEART_COLOR = "#DBDBDB";

    const outlineTargets: THREE.Object3D[] = [];

    const collectOutlineTargets = (root: THREE.Object3D) => {
      outlineTargets.length = 0;

      root.traverse((obj) => {
        const mesh = obj as THREE.Mesh;
        if (!mesh.isMesh || !mesh.geometry) return;
        outlineTargets.push(mesh);
      });
    };

    const normalMaterial = new THREE.MeshNormalMaterial();
    normalMaterial.blending = THREE.NoBlending;

    const depthMaterial = new THREE.MeshDepthMaterial({
      depthPacking: THREE.RGBADepthPacking,
    });
    depthMaterial.blending = THREE.NoBlending;

    const normalDepthBackground = new THREE.Color(0x000000);

    const internalLineartShader = {
      uniforms: {
        tDiffuse: { value: null },
        tNormal: { value: null },
        tDepth: { value: null },
        resolution: { value: new THREE.Vector2(1, 1) },
        lineColor: { value: new THREE.Color(LINEART_COLOR) },
        normalStrength: { value: 0.9 },
        depthStrength: { value: 0.95 },
        colorStrength: { value: 0.28 },
        normalThreshold: { value: 0.24 },
        depthThreshold: { value: 0.006 },
        colorThreshold: { value: 0.09 },
        opacity: { value: 0.72 },
      },
      vertexShader: `
        varying vec2 vUv;

        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D tDiffuse;
        uniform sampler2D tNormal;
        uniform sampler2D tDepth;
        uniform vec2 resolution;
        uniform vec3 lineColor;
        uniform float normalStrength;
        uniform float depthStrength;
        uniform float colorStrength;
        uniform float normalThreshold;
        uniform float depthThreshold;
        uniform float colorThreshold;
        uniform float opacity;

        #include <packing>

        varying vec2 vUv;

        float luma(vec3 color) {
          return dot(color, vec3(0.299, 0.587, 0.114));
        }

        vec3 decodeNormal(vec3 encodedNormal) {
          return encodedNormal * 2.0 - 1.0;
        }

        void main() {
          vec4 base = texture2D(tDiffuse, vUv);
          vec2 texel = 1.0 / resolution;

          vec3 normalCenter = decodeNormal(texture2D(tNormal, vUv).rgb);
          vec3 normalRight = decodeNormal(texture2D(tNormal, vUv + vec2(texel.x, 0.0)).rgb);
          vec3 normalLeft = decodeNormal(texture2D(tNormal, vUv - vec2(texel.x, 0.0)).rgb);
          vec3 normalUp = decodeNormal(texture2D(tNormal, vUv + vec2(0.0, texel.y)).rgb);
          vec3 normalDown = decodeNormal(texture2D(tNormal, vUv - vec2(0.0, texel.y)).rgb);

          float normalDiff = 0.0;
          normalDiff = max(normalDiff, length(normalCenter - normalRight));
          normalDiff = max(normalDiff, length(normalCenter - normalLeft));
          normalDiff = max(normalDiff, length(normalCenter - normalUp));
          normalDiff = max(normalDiff, length(normalCenter - normalDown));

          float normalEdge = smoothstep(normalThreshold, normalThreshold + 0.16, normalDiff) * normalStrength;

          float depthCenter = unpackRGBAToDepth(texture2D(tDepth, vUv));
          float depthRight = unpackRGBAToDepth(texture2D(tDepth, vUv + vec2(texel.x, 0.0)));
          float depthLeft = unpackRGBAToDepth(texture2D(tDepth, vUv - vec2(texel.x, 0.0)));
          float depthUp = unpackRGBAToDepth(texture2D(tDepth, vUv + vec2(0.0, texel.y)));
          float depthDown = unpackRGBAToDepth(texture2D(tDepth, vUv - vec2(0.0, texel.y)));

          float depthDiff = 0.0;
          depthDiff = max(depthDiff, abs(depthCenter - depthRight));
          depthDiff = max(depthDiff, abs(depthCenter - depthLeft));
          depthDiff = max(depthDiff, abs(depthCenter - depthUp));
          depthDiff = max(depthDiff, abs(depthCenter - depthDown));

          float depthEdge = smoothstep(depthThreshold, depthThreshold + 0.012, depthDiff) * depthStrength;

          float lRight = luma(texture2D(tDiffuse, vUv + vec2(texel.x, 0.0)).rgb);
          float lLeft = luma(texture2D(tDiffuse, vUv - vec2(texel.x, 0.0)).rgb);
          float lUp = luma(texture2D(tDiffuse, vUv + vec2(0.0, texel.y)).rgb);
          float lDown = luma(texture2D(tDiffuse, vUv - vec2(0.0, texel.y)).rgb);

          float colorDiff = max(abs(lRight - lLeft), abs(lUp - lDown));
          float colorEdge = smoothstep(colorThreshold, colorThreshold + 0.08, colorDiff) * colorStrength;

          float edge = clamp(max(max(normalEdge, depthEdge), colorEdge) * opacity, 0.0, 1.0);
          vec3 finalColor = mix(base.rgb, lineColor, edge);

          gl_FragColor = vec4(finalColor, base.a);
        }
      `,
    };

    const scene = new THREE.Scene();
    const ambient = new THREE.AmbientLight(0xffffff, 0.05);
    scene.add(ambient);

    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.05);
    hemiLight.position.set(0, 20, 0);
    scene.add(hemiLight);

    const dirFrontRight = new THREE.DirectionalLight(0xffffff, 0.04);
    dirFrontRight.position.set(10, 8, 10);
    scene.add(dirFrontRight);

    const dirFrontLeft = new THREE.DirectionalLight(0xffffff, 0.04);
    dirFrontLeft.position.set(-10, 8, 10);
    scene.add(dirFrontLeft);

    const dirBackRight = new THREE.DirectionalLight(0xffffff, 0.03);
    dirBackRight.position.set(10, 8, -10);
    scene.add(dirBackRight);

    const dirBackLeft = new THREE.DirectionalLight(0xffffff, 0.03);
    dirBackLeft.position.set(-10, 8, -10);
    scene.add(dirBackLeft);
    scene.background =
      typeof background === "string"
        ? new THREE.Color(background)
        : new THREE.Color(background);

    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      powerPreference: "high-performance",
    });

    renderer.outputColorSpace = THREE.SRGBColorSpace;

    const normalTarget = new THREE.WebGLRenderTarget(1, 1, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      depthBuffer: true,
      stencilBuffer: false,
    });
    normalTarget.texture.name = "SpectlineInternalLineartNormalTexture";

    const depthTarget = new THREE.WebGLRenderTarget(1, 1, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      depthBuffer: true,
      stencilBuffer: false,
    });
    depthTarget.texture.name = "SpectlineInternalLineartDepthTexture";

    const composer = new EffectComposer(renderer);
    const renderPass = new RenderPass(scene, activeCamera);
    composer.addPass(renderPass);

    const internalLineartPass = new ShaderPass(internalLineartShader);
    internalLineartPass.uniforms.tNormal.value = normalTarget.texture;
    internalLineartPass.uniforms.tDepth.value = depthTarget.texture;
    composer.addPass(internalLineartPass);

    const outlinePass = new OutlinePass(new THREE.Vector2(1, 1), scene, activeCamera);
    outlinePass.visibleEdgeColor.set(LINEART_COLOR);
    outlinePass.hiddenEdgeColor.set(LINEART_COLOR);
    outlinePass.edgeStrength = 1.35;
    outlinePass.edgeGlow = 0.0;
    outlinePass.edgeThickness = 0.85;
    outlinePass.pulsePeriod = 0;
    composer.addPass(outlinePass);

    const outputPass = new OutputPass();
    composer.addPass(outputPass);

    let rafSize = 0;

    const setSize = () => {
      rafSize = 0;

      const r = canvas.getBoundingClientRect();
      const w = Math.max(1, Math.round(r.width));
      const h = Math.max(1, Math.round(r.height));

      const dpr = Math.min(2, window.devicePixelRatio || 1);

      const renderW = Math.max(1, Math.round(w * dpr));
      const renderH = Math.max(1, Math.round(h * dpr));

      renderer.setPixelRatio(dpr);
      renderer.setSize(w, h, false);
      composer.setPixelRatio(dpr);
      composer.setSize(w, h);
      normalTarget.setSize(renderW, renderH);
      depthTarget.setSize(renderW, renderH);
      internalLineartPass.uniforms.resolution.value.set(renderW, renderH);
      outlinePass.resolution.set(w, h);

      if (activeCamera instanceof THREE.PerspectiveCamera) {
        activeCamera.aspect = w / h;
        activeCamera.updateProjectionMatrix();
      } else if (activeCamera instanceof THREE.OrthographicCamera) {
        activeCamera.updateProjectionMatrix();
      }
    };

    let lastViewportWidth = window.innerWidth;

    const onResize = () => {
      const nextWidth = window.innerWidth;

      const mobileViewport = window.matchMedia(
        "(max-width: 900px)"
      ).matches;

      const widthChanged =
        Math.abs(nextWidth - lastViewportWidth) > 2;

      if (mobileViewport && !widthChanged) {
        return;
      }

      lastViewportWidth = nextWidth;

      if (!rafSize) {
        rafSize = requestAnimationFrame(setSize);
      }
    };

    const onOrientationChange = () => {
      window.setTimeout(() => {
        lastViewportWidth = window.innerWidth;
        setSize();
      }, 250);
    };

    const ro = new ResizeObserver(onResize);
    ro.observe(canvas);

    window.addEventListener("resize", onResize);
    window.addEventListener(
      "orientationchange",
      onOrientationChange
    );

    setSize();

    const loader = new GLTFLoader();
    const dracoLoader = new DRACOLoader();

    dracoLoader.setDecoderPath(
      "https://www.gstatic.com/draco/versioned/decoders/1.5.6/"
    );

    loader.setDRACOLoader(dracoLoader);

    loader.load(
      glbUrl,
      (gltf) => {
        if (disposed) return;

        const root = gltf.scene;
        model = root;

        const glbCam =
          gltf.cameras?.[0] ??
          (root.getObjectByProperty("isCamera", true) as THREE.Camera | null);

        if (glbCam) {
          activeCamera = glbCam;

          if (!activeCamera.parent) scene.add(activeCamera);

          renderPass.camera = activeCamera;
          outlinePass.renderCamera = activeCamera;

          setSize();
        }

        collectOutlineTargets(root);
          outlinePass.selectedObjects = outlineTargets;

          scene.add(root);

          if (gltf.animations?.length) {
          const camName = activeCamera.name || undefined;

          modelMixer = new THREE.AnimationMixer(root);

          const playModel = (clip: THREE.AnimationClip) => {
            const a = modelMixer!.clipAction(clip);
            a.loop = THREE.LoopRepeat;
            a.play();
          };

          for (const clip of gltf.animations) {
            if (!camName) {
              playModel(clip);
              continue;
            }

            const camTracks = clip.tracks.filter((tr: THREE.KeyframeTrack) =>
              tr.name.startsWith(camName + ".")
            );
            const otherTracks = clip.tracks.filter((tr: THREE.KeyframeTrack) =>
              !tr.name.startsWith(camName + ".")
            );

            if (camTracks.length && !camClip) {
              camClip = new THREE.AnimationClip(clip.name + "__cam", clip.duration, camTracks);

              camMixer = new THREE.AnimationMixer(activeCamera);

              camAction = camMixer.clipAction(camClip);
              camAction.play();
              camAction.paused = true;
            }

            if (otherTracks.length) {
              const rest = new THREE.AnimationClip(clip.name + "__rest", clip.duration, otherTracks);
              playModel(rest);
            } else if (!camTracks.length) {
              playModel(clip);
            }
          }
        }

        setReady(true);
      },
      undefined,
      (err) => console.error("[AndroidBasic] load error", err)
    );

    const animate = () => {
      raf = requestAnimationFrame(animate);

      const dt = clock.getDelta();
      if (modelMixer) modelMixer.update(dt);

      if (camMixer && camClip && camAction) {
        const dur = camClip.duration;
        let p: number;

        if (autoCam) {
          if (autoHold > 0) {
            autoHold = Math.max(0, autoHold - dt);
          } else {
            autoP += autoDir * AUTO_SPEED * dt;

            if (autoP >= 1) {
              autoP = 1;
              autoDir = -1;
              autoHold = AUTO_HOLD_SEC;
            } else if (autoP <= 0) {
              autoP = 0;
              autoDir = 1;
              autoHold = AUTO_HOLD_SEC;
            }
          }
          p = autoP;
        } else {
          p = Math.min(1, Math.max(0, scrollPRef.current));
        }

        camAction.time = p * (dur - 1e-4);
        camMixer.update(0);
      }

      const previousOverrideMaterial = scene.overrideMaterial;
      const previousBackground = scene.background;

      scene.background = normalDepthBackground;

      scene.overrideMaterial = depthMaterial;
      renderer.setRenderTarget(depthTarget);
      renderer.clear();
      renderer.render(scene, activeCamera);

      scene.overrideMaterial = normalMaterial;
      renderer.setRenderTarget(normalTarget);
      renderer.clear();
      renderer.render(scene, activeCamera);

      renderer.setRenderTarget(null);
      scene.overrideMaterial = previousOverrideMaterial;
      scene.background = previousBackground;

      composer.render();
    };

    animate();

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);

      if (rafSize) cancelAnimationFrame(rafSize);
      ro.disconnect();

      window.removeEventListener("resize", onResize);
      window.removeEventListener(
        "orientationchange",
        onOrientationChange
      );

      if (modelMixer) {
        modelMixer.stopAllAction();
        try {
          modelMixer.uncacheRoot(modelMixer.getRoot());
        } catch {}
        modelMixer = null;
      }

      if (camMixer) {
        camMixer.stopAllAction();
        try {
          camMixer.uncacheRoot(camMixer.getRoot());
        } catch {}
        camMixer = null;
        camClip = null;
      }

      if (model) {
        model.traverse((object) => {
          if (!(object instanceof THREE.Mesh)) return;

          object.geometry.dispose();

          const materials = Array.isArray(object.material)
            ? object.material
            : [object.material];

          materials.forEach((material) => material.dispose());
        });
        scene.remove(model);
        model = null;
      }

      normalTarget.dispose();
      depthTarget.dispose();
      normalMaterial.dispose();
      depthMaterial.dispose();
      composer.dispose();
      renderer.dispose();
      setReady(false);
    };
  }, [glbUrl, background]);

  useLayoutEffect(() => {
    const mq = window.matchMedia("(max-width: 1500px)");
    const apply = () => {
      const m = mq.matches;
      setIsMobile(m);
      isMobileRef.current = m;
    };
    apply();
    mq.addEventListener?.("change", apply);
    return () => mq.removeEventListener?.("change", apply);
  }, []);

  useLayoutEffect(() => {
    const mq = window.matchMedia("(max-width: 900px)");

    const apply = () => {
      setIsMobileNav(mq.matches);
    };

    apply();
    mq.addEventListener?.("change", apply);

    return () => {
      mq.removeEventListener?.("change", apply);
    };
  }, []);

  useLayoutEffect(() => {
    const mq = window.matchMedia("(max-width: 1300px)");
    const apply = () => setIsProcessStacked(mq.matches);
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, []);

  useEffect(() => {
    let raf = 0;

    const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));

    const compute = () => {
      raf = 0;

      if (contactLockRef.current) return;

      const doc = document.documentElement;
      const maxScroll = Math.max(1, doc.scrollHeight - doc.clientHeight);

      const p = clamp(window.scrollY / maxScroll, 0, 1);
      scrollPRef.current = p;

      const t = p * SECTION_COUNT;

      if (t !== scrollTRef.current) {
        scrollTRef.current = t;
        setScrollT(t);
      }
    };

    const onScroll = () => {
      if (!raf) raf = requestAnimationFrame(compute);
    };

    compute();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);

    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);

  useEffect(() => {
    if (!contactLock) return;

    lockScrollYRef.current = window.scrollY;

    const body = document.body;
    body.style.position = "fixed";
    body.style.top = `-${lockScrollYRef.current}px`;
    body.style.left = "0";
    body.style.right = "0";
    body.style.width = "100%";

    return () => {
      const y = lockScrollYRef.current;
      body.style.position = "";
      body.style.top = "";
      body.style.left = "";
      body.style.right = "";
      body.style.width = "";
      window.scrollTo(0, y);
    };
  }, [contactLock]);

  const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
  const t = scrollT;

  const FADE = 0.25;
  const SECTION_LENGTH = 1;

  const sectionOpacity = (t: number, index: number) => {
    const start = index * SECTION_LENGTH;
    const localT = t - start;
    return clamp01(localT / FADE) * clamp01((SECTION_LENGTH - localT) / FADE);
  };

  const heroOpacity = sectionOpacity(t, 0);
  const processOpacity = sectionOpacity(t, 1);
  const pricingOpacity = sectionOpacity(t, 2);
  const demoOpacity = sectionOpacity(t, 3);
  const reviewsOpacity = sectionOpacity(t, 4);
  const contactOpacity = sectionOpacity(t, 5);
  const footerOpacity = clamp01((t - 6) / FADE);

  const invertFactor = clamp01(clamp01((t - 1.8) / 0.4) * clamp01((5.2 - t) / 0.4));

  const introControlOpacity = ready ? clamp01(1 - scrollT / 0.22) : 0;

  const showMobileHamburger =
    isMobileNav &&
    ready &&
    scrollT >= 0.22;

  const topControlOpacity = isMobileNav
    ? ready
      ? 1
      : 0
    : introControlOpacity;

  useEffect(() => {
    document.documentElement.style.setProperty(
      "--scrollbar-invert-factor",
      String(invertFactor)
    );

    return () => {
      document.documentElement.style.removeProperty("--scrollbar-invert-factor");
    };
  }, [invertFactor]);

  return (
    <div style={{ '--invert-factor': invertFactor } as React.CSSProperties}>
      {/* REAL scroll targets for #hash */}
      {Array.from({ length: SECTION_COUNT + 1 }).map((_, i) =>
        i < SECTION_COUNT ? (
          <div
            key={SECTION_IDS[i]}
            id={SECTION_IDS[i]}
            style={{
              height: "var(--spectline-viewport-height, 100svh)",
            }}
          />
        ) : (
          <div
            key="spacer"
            style={{
              height: "var(--spectline-viewport-height, 100svh)",
            }}
          />
        )
      )}

      <div
        onPointerDownCapture={onBackgroundPointerDownCapture}
        style={{
          position: "fixed",
          left: 0,
          top: 0,
          width: "100vw",
          height: "var(--spectline-viewport-height, 100svh)",
          opacity: hydrated ? 1 : 0,
          transition: "opacity 500ms ease",
        }}
      >
        <RadialMenu
          open={radial.open}
          x={radial.x}
          y={radial.y}
          items={radialItems}
          onClose={() => setRadial((s) => ({ ...s, open: false }))}
          onPick={(item) => {
            if (item.kind === "section") scrollToSection(item.id);
            if (item.kind === "route") router.push(item.href);
            setRadial((s) => ({ ...s, open: false }));
          }}
        />

        <div
          className="first_section__canvas_wrap"
          aria-hidden
          style={{ opacity: ready ? 1 : 0, transition: "opacity 600ms ease" }}
        >
          <canvas ref={canvasRef} className="first_section__canvas" />
        </div>

        <div
          className="scroll_hint_wrap"
          aria-hidden
          style={{ opacity: introControlOpacity }}
        >
          <div className="scroll_hint">{copy.scrollHint}</div>
        </div>

        <div
          className="absolute right-[18px] top-[18px] md:right-[clamp(20px,6vw,80px)] md:top-auto md:bottom-[clamp(22px,4vh,42px)] z-[5]"
          style={{
            opacity: topControlOpacity,
            pointerEvents: topControlOpacity > 0.05 ? "auto" : "none",
            transition: "opacity 400ms ease",
          }}
        >
          <button
            data-no-radial
            type="button"
            aria-label={
              showMobileHamburger
                ? lang === "sk"
                  ? "Otvoriť navigáciu"
                  : "Open navigation"
                : copy.language.ariaLabel
            }
            aria-expanded={
              showMobileHamburger
                ? radial.open
                : undefined
            }
            onClick={
              showMobileHamburger
                ? openMobileRadial
                : toggleLang
            }
            className={[
              "mobile-nav-trigger",
              "rounded-xl border border-[#19191A]/15",
              "bg-[#F4F4F4]/75 backdrop-blur-md",
              "px-4 py-2 text-sm font-semibold",
              "tracking-[0.16em] text-[#19191A]",
              "hover:border-[#19191A]/25",
              "hover:bg-[#E7E7E7]/85",
              "shadow-[0_8px_26px_rgba(0,0,0,0.08)]",
              showMobileHamburger
                ? "mobile-nav-trigger--menu"
                : "",
            ]
              .filter(Boolean)
              .join(" ")}
          >
            <span
              className="mobile-nav-trigger__language"
              aria-hidden={showMobileHamburger}
            >
              <span
                className={
                  lang === "sk"
                    ? "text-[#19191A]"
                    : "text-[#19191A]/45"
                }
              >
                SK
              </span>

              <span className="px-2 text-[#19191A]/35">
                /
              </span>

              <span
                className={
                  lang === "en"
                    ? "text-[#19191A]"
                    : "text-[#19191A]/45"
                }
              >
                EN
              </span>
            </span>

            <span
              className="mobile-nav-trigger__hamburger"
              aria-hidden={!showMobileHamburger}
            >
              <Menu className="h-6 w-6" strokeWidth={1.8} />
            </span>
          </button>
        </div>

        <section
          data-layer="home"
          className={[
            "first_section",
            ready ? "first_section--ready" : "",
            isMobile ? "first_section--mobile" : "",
          ].filter(Boolean).join(" ")}
          style={{
            position: "absolute",
            inset: 0,
            opacity: ready ? heroOpacity : 0,
            pointerEvents: heroOpacity > 0.05 ? "auto" : "none",
            transition: "opacity 600ms ease",
            zIndex: 2,
          }}
          aria-hidden={heroOpacity < 0.05}
        >
          {/* removed canvas from here */}
          <div className="first_section__grid">
            <div className="first_section__left">
              <h1 className="first_section__brand">{copy.hero.brand}</h1>

              <div className="first_section__panel">
                <p className="first_section__desc">
                  {copy.hero.desc}
                </p>

                <div className="first_section__actions">
                  <a
                    href="#pricing"
                    className="first_section__btn first_section__btn--primary"
                    onClick={(e) => {
                      e.preventDefault();
                      scrollToSection("pricing");
                    }}
                  >
                    {copy.hero.primaryCta}
                  </a>

                  <a
                    href="#process"
                    className="first_section__btn first_section__btn--ghost"
                    onClick={(e) => {
                      e.preventDefault();
                      scrollToSection("process");
                    }}
                  >
                    {copy.hero.secondaryCta}
                  </a>
                </div>
              </div>
            </div>

            <div className="first_section__right">
              <div className="first_section__card_wrap">
                <div className="first_section__card">
                  {copy.hero.card}
                </div>
              </div>
            </div>
          </div>
        </section>

        <section
          data-layer="process"
          style={{
            position: "absolute",
            inset: 0,
            opacity: processOpacity,
            transform: `translateY(${24 * (1 - scrollT)}px)`,
            pointerEvents: processOpacity ? "auto" : "none",
            transition: "opacity 120ms linear",
            zIndex: 1,
            display: "grid",
            placeItems: "center",
            padding: "clamp(20px, 6vw, 80px)",
          }}
          aria-hidden={t < 1 || t >= 2}
        >
          <div className="w-full max-w-[1200px]">
            <div className="text-center mb-8">
              <h2 className="text-[#19191A] text-3xl md:text-5xl tracking-[0.12em] uppercase opacity-95">
                {copy.process.title}
              </h2>
              <p className="text-[#19191A]/70 mt-3 max-w-[60ch] mx-auto">
                {copy.process.subtitle}
              </p>
            </div>

            {!isProcessStacked ? (
              <div className="flex items-stretch justify-center gap-6">
                {processSteps.map((s, i) => (
                  <div key={s.n} className="flex items-stretch gap-6">
                    <div className="w-[320px] rounded-2xl border border-white/10 bg-white/5 backdrop-blur-md p-5 transition duration-200 hover:scale-[1.02] hover:border-white/35 hover:bg-white/10">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full border border-white/15 bg-white/5 grid place-items-center text-[#19191A] font-semibold">
                          {s.n}
                        </div>
                        <div className="text-[#19191A] font-semibold">{s.title}</div>
                      </div>
                      <div className="text-[#19191A]/75 mt-3 leading-relaxed text-sm">
                        {s.text}
                      </div>
                    </div>

                    {i < processSteps.length - 1 ? (
                      <div className="w-16 flex items-center">
                        <div className="h-[2px] w-full bg-white/15" />
                      </div>
                    ) : null}
                  </div>
                ))}
              </div>
            ) : (
            <div className="w-full max-w-[520px] mx-auto space-y-3">
              {processSteps.map((s, i) => {
                const open = openProcessStep === i;

                return (
                  <div
                    key={s.n}
                    className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-md transition duration-200 hover:scale-[1.01] hover:border-white/25 hover:bg-white/10"
                  >
                    <button
                      type="button"
                      onClick={() => setOpenProcessStep(open ? null : i)}
                      className="w-full flex items-start justify-between gap-4 p-4 text-left"
                    >
                      <div className="flex gap-3 items-start min-w-0">
                        <div className="shrink-0 w-9 h-9 rounded-full border border-white/15 bg-white/5 grid place-items-center text-[#19191A] font-semibold">
                          {s.n}
                        </div>

                        <div className="min-w-0">
                          <div className="text-[#19191A] font-semibold text-base">
                            {s.title}
                          </div>

                          <div
                            className={[
                              "text-[#19191A]/75 mt-2 text-sm leading-relaxed overflow-hidden transition-all duration-200",
                              open ? "max-h-[180px]" : "max-h-[42px]",
                            ].join(" ")}
                          >
                            {s.text}
                          </div>
                        </div>
                      </div>

                      <span
                        className={[
                          "shrink-0 pt-1 text-[#19191A]/70 transition-transform",
                          open ? "rotate-180" : "",
                        ].join(" ")}
                      >
                        ▾
                      </span>
                    </button>
                  </div>
                );
              })}
            </div>
          )}
          </div>
        </section>

        <section
          data-layer="pricing"
          className={[
            "absolute inset-0 z-[3] grid px-[clamp(20px,6vw,80px)] py-[clamp(20px,6vw,80px)]",
            "place-items-center",
          ].join(" ")}
          style={{
            opacity: pricingOpacity,
            pointerEvents: pricingOpacity > 0.05 ? "auto" : "none",
            transition: "opacity 120ms linear",
            overflowY: isPricingAccordion ? "auto" : "visible",
          }}
          aria-hidden={pricingOpacity < 0.05}
        >
          <div className="w-full max-w-[1200px]">
            <div className="text-center mb-8">
              <h2 className="text-[#19191A] text-3xl md:text-5xl tracking-[0.12em] uppercase opacity-95">
                {copy.pricing.title}
              </h2>
              <p className="text-[#F4F4F4]/70 mt-3 max-w-[60ch] mx-auto">
                {copy.pricing.subtitle}
              </p>
            </div>

            {isPricingAccordion ? (
              isPricingSlider ? (
                (() => {
                  const pricingSliderIdx = displayedOpenPlan ?? 0;
                  const pricingSliderPlan = pricingPlans[pricingSliderIdx] ?? pricingPlans[0];

                  if (!pricingSliderPlan) return null;

                  return (
                    <div>
                      <div
                        key={pricingSliderIdx}
                        className={[
                          "rounded-2xl border border-[#F4F4F4]/15 bg-[#19191A]/70 backdrop-blur-md select-none",
                          pricingSlideDir === 1
                            ? "animate-[planInRight_260ms_ease-out]"
                            : "animate-[planInLeft_260ms_ease-out]",
                        ].join(" ")}
                        style={{ touchAction: "pan-y" }}
                        onPointerDown={(e) => {
                          if (e.pointerType === "mouse") return;
                          swipeRef.current = { x: e.clientX, y: e.clientY, id: e.pointerId, fired: false };
                          (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
                        }}
                        onPointerMove={(e) => {
                          const s = swipeRef.current;
                          if (s.id !== e.pointerId || s.fired) return;

                          const dx = e.clientX - s.x;
                          const dy = e.clientY - s.y;

                          if (Math.abs(dy) > Math.abs(dx)) return;

                          if (Math.abs(dx) > 46) {
                            s.fired = true;
                            swipePlan(dx < 0 ? 1 : -1);
                          }
                        }}
                        onPointerUp={(e) => {
                          if (swipeRef.current.id === e.pointerId) swipeRef.current.id = null;
                        }}
                        onPointerCancel={(e) => {
                          if (swipeRef.current.id === e.pointerId) swipeRef.current.id = null;
                        }}
                      >
                        <div className="p-4">
                          <div className="text-[#F4F4F4] font-semibold text-lg">{pricingSliderPlan.name}</div>
                          <div className="text-[#F4F4F4]/90 mt-1">{pricingSliderPlan.price}</div>

                          <div className="text-[#F4F4F4]/80 leading-relaxed mt-3">{pricingSliderPlan.desc}</div>

                          <ul className="mt-4 space-y-2 text-[#F4F4F4]/85">
                            {pricingSliderPlan.details.map((d) => (
                              <li key={d} className="flex gap-3 items-start">
                                <span className="opacity-70">•</span>
                                <span>{d}</span>
                              </li>
                            ))}
                          </ul>

                          <div className="pt-4">
                            <button
                              type="button"
                              onClick={() => scrollToSection("contact")}
                              className={[
                                "pricing-contact-btn", "inline-flex w-full items-center justify-center rounded-xl border px-4 py-3 font-semibold transition cursor-pointer",
                                "border-[#F4F4F4]/20 bg-[#19191A]/55 text-[#F4F4F4]",
                                "hover:border-[#F4F4F4]/35 hover:bg-[#19191A]/45",
                              ].join(" ")}
                            >
                              {copy.pricing.order}
                            </button>
                          </div>
                        </div>
                      </div>

                      <div className="mt-4 flex justify-center gap-2" aria-hidden>
                        {pricingPlans.map((_, idx) => (
                          <span
                            key={idx}
                            className={[
                              "h-2 rounded-full transition-all duration-200",
                              pricingSliderIdx === idx ? "w-8 bg-[#F4F4F4]/80" : "w-2 bg-[#F4F4F4]/30",
                            ].join(" ")}
                          />
                        ))}
                      </div>
                    </div>
                  );
                })()
              ) : (
                <div className="space-y-3">
                  {pricingPlans.map((p, i) => {
                    const open = displayedOpenPlan === i;

                    return (
                      <div
                        key={p.name}
                        className="rounded-2xl border border-[#F4F4F4]/15 bg-[#19191A]/70 backdrop-blur-md"
                      >
                        <button
                          type="button"
                          onClick={() => setOpenPlan(open ? null : i)}
                          className="w-full flex items-center justify-between gap-4 p-4 text-left"
                        >
                          <div>
                            <div className="text-[#F4F4F4] font-semibold text-lg">{p.name}</div>
                            <div className="text-[#F4F4F4]/90 mt-1">{p.price}</div>
                          </div>
                          <span
                            className={[
                              "text-[#F4F4F4]/70 transition-transform",
                              open ? "rotate-180" : "",
                            ].join(" ")}
                          >
                            ▾
                          </span>
                        </button>

                        {open ? (
                          <div className="px-4 pb-4">
                            <div className="text-[#F4F4F4]/80 leading-relaxed">{p.desc}</div>

                            <ul className="mt-4 space-y-2 text-[#F4F4F4]/85">
                              {p.details.map((d) => (
                                <li key={d} className="flex gap-3 items-start">
                                  <span className="opacity-70">•</span>
                                  <span>{d}</span>
                                </li>
                              ))}
                            </ul>

                            <div className="pt-4">
                              <button
                                type="button"
                                onClick={() => scrollToSection("contact")}
                                className={[
                                  "pricing-contact-btn",
                                  "inline-flex w-full items-center justify-center rounded-xl border px-4 py-3 font-semibold transition cursor-pointer",
                                  "border-[#F4F4F4]/20 bg-[#19191A]/55 text-[#F4F4F4]",
                                  "hover:border-[#F4F4F4]/35 hover:bg-[#19191A]/45",
                                ].join(" ")}
                              >
                                {copy.pricing.order}
                              </button>
                            </div>
                          </div>
                        ) : null}
                      </div>
                    );
                  })}
                </div>
              )
            ) : (
              <div
                className={[
                  "grid gap-6 items-stretch",
                  pricingPlans.length === 1
                    ? "md:grid-cols-1 max-w-[430px] mx-auto"
                    : pricingPlans.length === 2
                    ? "md:grid-cols-2 max-w-[900px] mx-auto"
                    : "md:grid-cols-3",
                ].join(" ")}
              >
                {pricingPlans.map((p, i) => {
                  const active = hoveredPlan !== null ? i === hoveredPlan : i === displayedActivePlan;

                  return (
                    <div
                      key={p.name}
                      onMouseEnter={() => {
                        setHoveredPlan(i);
                        setActivePlan(i);
                      }}
                      onMouseLeave={() => setHoveredPlan(null)}
                      className={[
                        "h-full flex flex-col rounded-2xl border p-5 transition duration-200 text-[#F4F4F4]",
                        "border-[#F4F4F4]/15 bg-[#19191A]/80 backdrop-blur-md",
                        "hover:scale-[1.02] hover:border-[#F4F4F4]/35 hover:bg-[#19191A]/60 hover:backdrop-blur-md",
                        active ? "scale-[1.03] border-[#F4F4F4]/40 bg-[#19191A]/55 backdrop-blur-md" : "",
                      ].join(" ")}
                    >
                      <div className="text-[#F4F4F4] font-semibold text-lg">{p.name}</div>
                      <div className="text-[#F4F4F4] mt-1 text-lg opacity-95">{p.price}</div>
                      <div className="text-[#F4F4F4]/80 mt-3 leading-relaxed">{p.desc}</div>

                      <ul className="mt-4 space-y-2 text-[#F4F4F4]/85">
                        {p.details.map((d) => (
                          <li key={d} className="flex gap-3 items-start">
                            <span className="opacity-70">•</span>
                            <span>{d}</span>
                          </li>
                        ))}
                      </ul>

                      <div className="mt-auto pt-4">
                        <button
                          type="button"
                          onClick={() => scrollToSection("contact")}
                          className={[
                            "pricing-contact-btn",
                            "inline-flex w-full items-center justify-center rounded-xl border px-4 py-3 font-semibold transition",
                            "border-[#F4F4F4]/20 bg-[#19191A]/70 backdrop-blur-md text-[#F4F4F4] cursor-pointer",
                            "hover:border-[#F4F4F4]/35 hover:bg-[#19191A]/55",
                            active ? "border-[#F4F4F4]/35 bg-[#19191A]/55" : "",
                          ].join(" ")}
                        >
                          {copy.pricing.order}
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </section>

        <section
          data-layer="demo"
          className="absolute inset-0 z-[3] grid px-[clamp(20px,6vw,80px)] py-[clamp(18px,4svh,52px)] place-items-center overflow-hidden max-[900px]:overflow-y-auto"
          style={{
            opacity: demoOpacity,
            pointerEvents: demoOpacity > 0.05 ? "auto" : "none",
            transition: "opacity 120ms linear",
          }}
          aria-hidden={demoOpacity < 0.05}
        >
          <div className="w-full max-w-[1600px] flex flex-col items-center">
            <div className="text-center mb-12">
              <h2 className="text-[#F4F4F4] text-3xl md:text-5xl tracking-[0.12em] uppercase opacity-95">
                {copy.security.title}
              </h2>
              <p className="text-[#F4F4F4]/70 mt-3 max-w-[70ch] mx-auto">
                {copy.security.subtitle}
              </p>
            </div>

            {(() => {
              const cards = [
                {
                  title: copy.security.mediaTitle,
                  desc: copy.security.mediaText,
                  imgSrc: "/images/logo.png",
                },
                {
                  title: copy.security.panelTitle,
                  desc: copy.security.items[0],
                  imgSrc: "/images/logo.png",
                },
                {
                  title: copy.security.cta,
                  desc: copy.security.items[1],
                  imgSrc: "/images/logo.png",
                },
              ];

              const nextSlide = () => {
                setActiveSecurityIndex((prev) => (prev + 1) % cards.length);
              };

              const prevSlide = () => {
                setActiveSecurityIndex((prev) => (prev - 1 + cards.length) % cards.length);
              };

              return (
                <div className="w-full flex flex-col items-center">
                  <div
                    data-no-radial
                    className="relative w-full h-[460px] md:h-[720px] flex items-center justify-center overflow-hidden perspective-1000 select-none cursor-grab active:cursor-grabbing"
                    style={{ touchAction: "pan-y" }}
                    onPointerDown={(e) => {
                      securityDragRef.current = {
                        startX: e.clientX,
                        startY: e.clientY,
                        pointerId: e.pointerId,
                        fired: false,
                      };

                      e.currentTarget.setPointerCapture(e.pointerId);
                    }}
                    onPointerMove={(e) => {
                      const drag = securityDragRef.current;

                      if (drag.pointerId !== e.pointerId || drag.fired) return;

                      const dx = e.clientX - drag.startX;
                      const dy = e.clientY - drag.startY;

                      if (Math.abs(dy) > Math.abs(dx)) return;

                      if (Math.abs(dx) > 55) {
                        drag.fired = true;

                        if (dx < 0) {
                          nextSlide();
                        } else {
                          prevSlide();
                        }
                      }
                    }}
                    onPointerUp={(e) => {
                      if (securityDragRef.current.pointerId === e.pointerId) {
                        securityDragRef.current.pointerId = null;
                      }
                    }}
                    onPointerCancel={(e) => {
                      if (securityDragRef.current.pointerId === e.pointerId) {
                        securityDragRef.current.pointerId = null;
                      }
                    }}
                  >
                    {cards.map((card, index) => {
                      let position = "center";

                      if (index === (activeSecurityIndex - 1 + cards.length) % cards.length) {
                        position = "left";
                      }

                      if (index === (activeSecurityIndex + 1) % cards.length) {
                        position = "right";
                      }

                      const isCenter = position === "center";

                      return (
                        <div
                          key={index}
                          onClick={() => {
                            if (securityDragRef.current.fired) return;
                            setActiveSecurityIndex(index);
                          }}
                          className={`absolute w-[85vw] md:w-[1000px] rounded-2xl border bg-[#19191A]/95 backdrop-blur-md overflow-hidden cursor-pointer select-none transition-all duration-500 ease-out shadow-none ${
                            isCenter
                              ? "z-30 border-[#F4F4F4]/40 scale-100 opacity-100 translate-x-0"
                              : position === "left"
                              ? "z-10 border-[#F4F4F4]/10 scale-75 opacity-25 -translate-x-[65%] md:-translate-x-[80%]"
                              : "z-10 border-[#F4F4F4]/10 scale-75 opacity-25 translate-x-[65%] md:translate-x-[80%]"
                          }`}
                        >
                          <div className="relative aspect-square md:aspect-video w-full bg-black/50 flex items-center justify-center border-b border-[#F4F4F4]/15">
                            <Image
                              src={card.imgSrc}
                              alt={card.title}
                              fill
                              sizes="(max-width: 768px) 85vw, 1000px"
                              className="object-contain pointer-events-none"
                              draggable={false}
                            />
                          </div>

                          <div className="p-4 md:p-5 min-h-0 flex flex-col justify-center bg-black/30">
                            <h3 className="text-[#F4F4F4] text-lg md:text-xl font-semibold mb-1 text-center">
                              {card.title}
                            </h3>

                            <p className="text-[#F4F4F4]/65 text-xs md:text-sm text-center leading-relaxed line-clamp-2">
                              {card.desc}
                            </p>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })()}
          </div>
        </section>

        <section
          data-layer="reviews"
          className="absolute inset-0 z-[3] grid place-items-center px-[clamp(20px,6vw,80px)] py-[clamp(20px,6vw,80px)]"
          style={{
            opacity: reviewsOpacity,
            pointerEvents: reviewsOpacity > 0.05 ? "auto" : "none",
            transition: "opacity 120ms linear",
          }}
          aria-hidden={reviewsOpacity < 0.05}
        >
          <div className="mx-auto max-w-[1200px] text-center">
            <h2 className="text-[#F4F4F4] text-3xl md:text-5xl tracking-[0.12em] uppercase opacity-95">
              {copy.reviews.title}
            </h2>
            <p className="text-[#F4F4F4]/70 mt-3 max-w-[70ch] mx-auto">
              {copy.reviews.subtitle}
            </p>
          </div>

          <div className="mt-10 w-full overflow-hidden py-6 relative">
            <div
              className={[
                "flex flex-nowrap w-max gap-6 animate-[reviews-marquee_42s_linear_infinite]",
                "hover:[animation-play-state:paused]",
                reviewsPaused ? "[animation-play-state:paused]" : "",
              ].join(" ")}
            >
              {[...reviews, ...reviews].map((r, idx) => (
                <div key={`${idx}-${r.name}`} className="shrink-0 w-[min(340px,84vw)] md:w-[min(420px,86vw)]">
                  <div
                    data-no-radial
                    onClick={() => setReviewsPaused((p) => !p)}
                    className="h-full min-h-[240px] md:min-h-[320px] rounded-2xl border border-[#F4F4F4]/15 bg-[#19191A]/70 backdrop-blur-md p-5 md:p-8 transition-all duration-300 hover:-translate-y-1 hover:shadow-xl flex flex-col text-left"
                  >
                    <div className="flex items-center gap-1 text-[#c9a227] text-xl">
                      <span>★</span><span>★</span><span>★</span><span>★</span><span>★</span>
                    </div>

                    <p className="mt-4 text-lg text-[#F4F4F4]/80 leading-relaxed flex-1">
                      “{r.text}”
                    </p>

                    <div className="mt-6 pt-5 border-t border-[#F4F4F4]/15">
                      <div className="font-semibold text-[#F4F4F4]">{r.name}</div>
                      <div className="text-sm text-[#F4F4F4]/60">{r.role}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          <style jsx global>{`
            @keyframes reviews-marquee {
              from { transform: translateX(0); }
              to   { transform: translateX(-50%); }
            }
          `}</style>
        </section>

        <section
          data-layer="contact"
          className="absolute inset-0 z-[3] grid px-[clamp(20px,6vw,80px)] py-[clamp(20px,6vw,80px)] place-items-center"
          style={{
            opacity: contactOpacity,
            pointerEvents: contactOpacity > 0.05 ? "auto" : "none",
            transition: "opacity 120ms linear",
          }}
          aria-hidden={contactOpacity < 0.05}
        >
          <div className="relative w-full max-w-[1200px] mx-auto">
            <div className="relative w-full mx-auto max-w-[900px] grid gap-8 place-items-center text-center">
              <div className="w-full">
                <h2 className="text-[#F4F4F4] text-3xl md:text-5xl tracking-[0.12em] uppercase opacity-95">
                  {copy.contact.title}
                </h2>
                <p className="mt-4 text-lg text-[#19191A]/70 leading-relaxed mx-auto max-w-[60ch] max-[640px]:hidden">
                  {copy.contact.desc}
                </p>

                <ul className="mt-6 space-y-2 text-[#19191A]/70 mx-auto w-fit text-left max-[740px]:hidden">
                  <li>{copy.contact.emailLine}</li>
                  <li>{copy.contact.supportLine}</li>
                </ul>
              </div>

              <form
                ref={contactFormRef}
                onSubmit={handleContactSubmit}
                aria-busy={contactFormStatus === "submitting"}
                onChangeCapture={() => {
                  if (
                    contactFormStatus === "success" ||
                    contactFormStatus === "error"
                  ) {
                    setContactFormStatus("idle");
                  }
                }}
                onFocusCapture={() => {
                  if (window.matchMedia("(hover: hover) and (pointer: fine)").matches) return;

                  scrollTRef.current = CONTACT_T;
                  scrollPRef.current = CONTACT_T / SECTION_COUNT;
                  setScrollT(CONTACT_T);
                  setContactLock(true);
                }}
                onBlurCapture={() => {
                  setTimeout(() => {
                    const root = contactFormRef.current;
                    const a = document.activeElement as HTMLElement | null;
                    if (!root) return setContactLock(false);
                    if (a && root.contains(a)) return;
                    setContactLock(false);
                  }, 0);
                }}
                className="w-full space-y-4 rounded-2xl border border-[#19191A]/15 bg-[#F4F4F4]/75 p-6 sm:p-8 backdrop-blur-md text-left text-[#19191A] shadow-[0_18px_60px_rgba(0,0,0,0.10)]"
              >
                <div className="grid gap-4 md:grid-cols-2">
                  <div>
                    <label htmlFor="contact-name" className="block text-sm mb-1 text-[#19191A]/75">
                      {copy.contact.fields.name.label}
                    </label>
                    <input
                      id="contact-name"
                      name="name"
                      type="text"
                      required
                      className="w-full rounded-xl bg-[#E7E7E7]/85 border border-[#19191A]/15 px-4 py-3 text-[#19191A] placeholder-[#19191A]/40 focus:outline-none focus:ring-2 focus:ring-[#19191A]/20"
                      placeholder={copy.contact.fields.name.placeholder}
                    />
                  </div>

                  <div>
                    <label htmlFor="contact-system" className="block text-sm mb-1 text-[#19191A]/75">
                      {copy.contact.fields.system.label}
                    </label>
                    <input
                      id="contact-system"
                      name="system"
                      type="text"
                      className="w-full rounded-xl bg-[#E7E7E7]/85 border border-[#19191A]/15 px-4 py-3 text-[#19191A] placeholder-[#19191A]/40 focus:outline-none focus:ring-2 focus:ring-[#19191A]/20"
                      placeholder={copy.contact.fields.system.placeholder}
                    />
                  </div>
                </div>

                <div>
                  <label htmlFor="contact-email" className="block text-sm mb-1 text-[#19191A]/75">
                    {copy.contact.fields.email.label}
                  </label>
                  <input
                    id="contact-email"
                    name="email"
                    type="email"
                    required
                    className="w-full rounded-xl bg-[#E7E7E7]/85 border border-[#19191A]/15 px-4 py-3 text-[#19191A] placeholder-[#19191A]/40 focus:outline-none focus:ring-2 focus:ring-[#19191A]/20"
                    placeholder={copy.contact.fields.email.placeholder}
                  />
                </div>

                <div>
                  <label htmlFor="contact-message" className="block text-sm mb-1 text-[#19191A]/75">
                    {copy.contact.fields.message.label}
                  </label>
                  <textarea
                    id="contact-message"
                    name="message"
                    rows={4}
                    required
                    className="w-full rounded-xl bg-[#E7E7E7]/85 border border-[#19191A]/15 px-4 py-3 text-[#19191A] placeholder-[#19191A]/40 focus:outline-none focus:ring-2 focus:ring-[#19191A]/20"
                    placeholder={copy.contact.fields.message.placeholder}
                  />
                </div>

                <button
                  type="submit"
                  disabled={contactFormStatus === "submitting"}
                  className={[
                    "w-full rounded-2xl bg-[#E7E7E7]/90",
                    "border border-[#19191A]/15 px-8 py-3",
                    "text-lg font-semibold text-[#19191A]",
                    "transition-all duration-300",
                    "hover:-translate-y-0.5",
                    "hover:bg-[rgba(59,130,246,0.34)]",
                    "hover:border-[rgba(147,197,253,0.38)]",
                    "hover:text-[#F4F4F4]",
                    "hover:shadow-[inset_0_1px_0_rgba(255,255,255,0.16),0_8px_26px_rgba(59,130,246,0.12)]",
                    "disabled:cursor-not-allowed",
                    "disabled:opacity-60",
                    "disabled:hover:translate-y-0",
                  ].join(" ")}
                >
                  {contactFormStatus === "submitting"
                    ? contactStatusCopy?.submitting ?? copy.contact.submit
                    : copy.contact.submit}
                </button>
                <div
                  aria-live="polite"
                  aria-atomic="true"
                  className="min-h-6 text-center text-sm"
                >
                  {contactFormStatus === "success" &&
                  contactStatusCopy?.success ? (
                    <p role="status" className="text-emerald-700">
                      {contactStatusCopy.success}
                    </p>
                  ) : null}

                  {contactFormStatus === "error" &&
                  contactStatusCopy?.error ? (
                    <p role="alert" className="text-red-700">
                      {contactStatusCopy.error}
                    </p>
                  ) : null}
                </div>
              </form>
            </div>
          </div>
        </section>

        <footer
          data-layer="footer"
          className="absolute inset-0 z-[3] grid px-[clamp(20px,6vw,80px)] py-[clamp(20px,6vw,80px)]"
          style={{
            opacity: footerOpacity,
            pointerEvents: footerOpacity > 0.05 ? "auto" : "none",
            transition: "opacity 120ms linear",
          }}
          aria-hidden={footerOpacity < 0.05}
        >
          <div className="w-full max-w-[1200px] mx-auto self-end">
            <div className="rounded-2xl border border-[#19191A]/15 bg-[#F4F4F4]/75 backdrop-blur-md p-8 text-[#19191A] shadow-[0_18px_60px_rgba(0,0,0,0.10)]">
              <div className="grid gap-10 min-[900px]:grid-cols-3 text-center min-[900px]:text-left">
                <div>
                  <h3 className="text-lg font-semibold text-[#19191A]">{copy.footer.infoTitle}</h3>
                  <ul className="mt-4 space-y-2 text-lg text-[#19191A]/70">
                    <li>
                      <a
                        href={documentHref("terms")}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="transition-colors duration-200 hover:text-[#19191A]"
                      >
                        {copy.footer.infoLinks.terms}
                      </a>
                    </li>

                    <li>
                      <a
                        href={documentHref("gdpr")}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="transition-colors duration-200 hover:text-[#19191A]"
                      >
                        {copy.footer.infoLinks.privacy}
                      </a>
                    </li>

                    <li>
                      <a
                        href={documentHref("complaint")}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="transition-colors duration-200 hover:text-[#19191A]"
                      >
                        {copy.footer.infoLinks.complaint}
                      </a>
                    </li>

                    <li>
                      <a
                        href={documentHref("invoice")}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="transition-colors duration-200 hover:text-[#19191A]"
                      >
                        {copy.footer.infoLinks.invoice}
                      </a>
                    </li>
                  </ul>
                </div>

                <div>
                  <h3 className="text-lg font-semibold text-[#19191A]">{copy.footer.socialTitle}</h3>
                    <ul className="mt-4 space-y-2 text-lg text-[#19191A]/70">
                    <li>
                      <a
                        href="https://www.instagram.com/spectline/"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="transition-colors duration-200 hover:text-[#19191A]"
                      >
                        {copy.footer.socialLinks.instagram}
                      </a>
                    </li>
                    <li>
                      <a
                        href="https://www.youtube.com/@Spectline"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="transition-colors duration-200 hover:text-[#19191A]"
                      >
                        {copy.footer.socialLinks.youtube}
                      </a>
                    </li>
                    <li>
                      <a
                        href="https://www.tiktok.com/@spectline"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="transition-colors duration-200 hover:text-[#19191A]"
                      >
                        {copy.footer.socialLinks.tiktok}
                      </a>
                    </li>
                  </ul>
                </div>

                <div>
                  <h3 className="text-lg font-semibold text-[#19191A]">{copy.footer.newsletterTitle}</h3>
                  <form
                    className="mt-4 flex flex-col sm:flex-row gap-3 justify-center md:justify-start"
                    action="/api/newsletter"
                    method="post"
                  >
                    <label htmlFor="nl-email" className="sr-only">{copy.footer.newsletterLabel}</label>
                    <input
                      id="nl-email"
                      name="email"
                      type="email"
                      required
                      placeholder={copy.footer.newsletterPlaceholder}
                      className="w-full rounded-xl bg-[#E7E7E7]/85 border border-[#19191A]/15 px-4 py-3 text-[#19191A] placeholder-[#19191A]/40 focus:outline-none focus:ring-2 focus:ring-[#19191A]/20"
                    />
                    <button
                      type="submit"
                      className={[
                        "shrink-0 rounded-xl border px-5 py-3 font-semibold transition",
                        "border-[#19191A]/15 bg-[#E7E7E7]/90 text-[#19191A]",
                        "hover:-translate-y-0.5 hover:bg-[rgba(34,197,94,0.34)] hover:border-[rgba(134,239,172,0.38)] hover:text-[#F4F4F4] hover:shadow-[inset_0_1px_0_rgba(255,255,255,0.16),0_8px_26px_rgba(34,197,94,0.12)]",
                      ].join(" ")}
                    >
                      {copy.footer.newsletterButton}
                    </button>
                  </form>
                </div>
              </div>
            </div>

            <div className="mt-6 text-center text-sm text-[#19191A]/55">
              © {new Date().getFullYear()} Spectline
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}