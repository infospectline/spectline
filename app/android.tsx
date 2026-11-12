"use client";

import Image from "next/image";
import { useRouter } from "next/navigation";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  ChevronLeft,
  ChevronRight,
  X,
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
import sk from "@/lib/json/sk.json";
import en from "@/lib/json/en.json";

type AndroidBasicProps = {
  glbUrl?: string;
  background?: number | string;
  launcherFinished?: boolean;
};

type Lang = "sk" | "en";
type SiteCopy = typeof sk;
type NavLabelKey = keyof SiteCopy["nav"];

const COPY_BY_LANG: Record<Lang, SiteCopy> = {
  sk,
  en,
};

const SECTION_IDS = ["home", "process", "services", "portfolio", "reviews", "contact", "footer"] as const;
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

type NewsletterNotificationStatus =
  | "idle"
  | "pending"
  | "confirmed"
  | "unsubscribed"
  | "invalid"
  | "approval_approved"
  | "approval_rejected"
  | "approval_invalid"
  | "approval_already_processed"
  | "approval_error";

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
  { key: "services", kind: "section", id: "services", labelKey: "services", icon: Monitor },
  { key: "portfolio", kind: "section", id: "portfolio", labelKey: "portfolio", icon: Images },
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
  launcherFinished = true,
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
  const servicesPlans = copy.services.plans;
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

    const sectionOffset = id === "reviews" ? -0.18 : 0;

    window.scrollTo({
      top: (idx + sweet + sectionOffset) * vh,
      behavior: "smooth",
    });
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
  const invertFactorRef = useRef(0);

  const [activePlan, setActivePlan] = useState(0);
  const [hoveredPlan, setHoveredPlan] = useState<number | null>(null);

  const [isservicesAccordion, setIsservicesAccordion] = useState(false);
  const [isservicesSlider, setIsservicesSlider] = useState(false);
  const [openPlan, setOpenPlan] = useState<number | null>(0);
  const [openProcessStep, setOpenProcessStep] = useState<number | null>(0);

  const maxPlanIndex = Math.max(0, servicesPlans.length - 1);
  const displayedActivePlan = Math.min(activePlan, maxPlanIndex);
  const displayedOpenPlan = isservicesAccordion
    ? Math.min(openPlan ?? 0, maxPlanIndex)
    : null;

  const swipeRef = useRef<{ x: number; y: number; id: number | null; fired: boolean }>({
    x: 0,
    y: 0,
    id: null,
    fired: false,
  });

  const [servicesSlideDir, setservicesSlideDir] = useState<1 | -1>(1);

  const swipePlan = (dir: 1 | -1) => {
    setservicesSlideDir(dir);
    setOpenPlan((p) => {
      const count = servicesPlans.length;
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

  const [
    newsletterNotificationStatus,
    setNewsletterNotificationStatus,
  ] = useState<NewsletterNotificationStatus>("idle");

  const [
    queuedNewsletterNotificationStatus,
    setQueuedNewsletterNotificationStatus,
  ] = useState<NewsletterNotificationStatus>("idle");

  const [newsletterSubmitting, setNewsletterSubmitting] =
    useState(false);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);

    const approvalStatus = params.get("approval");
    const newsletterStatus = params.get("newsletter");

    let nextStatus: NewsletterNotificationStatus = "idle";

    if (approvalStatus === "approved") {
      nextStatus = "approval_approved";
    } else if (approvalStatus === "rejected") {
      nextStatus = "approval_rejected";
    } else if (approvalStatus === "invalid") {
      nextStatus = "approval_invalid";
    } else if (approvalStatus === "already_processed") {
      nextStatus = "approval_already_processed";
    } else if (approvalStatus === "error") {
      nextStatus = "approval_error";
    } else if (newsletterStatus === "confirmed") {
      nextStatus = "confirmed";
    } else if (newsletterStatus === "unsubscribed") {
      nextStatus = "unsubscribed";
    } else if (
      newsletterStatus === "invalid_confirmation" ||
      newsletterStatus === "invalid_or_expired" ||
      newsletterStatus === "invalid_unsubscribe" ||
      newsletterStatus === "invalid_or_used_unsubscribe"
    ) {
      nextStatus = "invalid";
    }

    if (nextStatus === "idle") {
      return;
    }

    setQueuedNewsletterNotificationStatus(nextStatus);

    params.delete("newsletter");
    params.delete("approval");

    const cleanUrl = `${window.location.pathname}${
      params.toString() ? `?${params.toString()}` : ""
    }${window.location.hash}`;

    window.history.replaceState(null, "", cleanUrl);
  }, []);

  useEffect(() => {
    if (
      !launcherFinished ||
      queuedNewsletterNotificationStatus === "idle"
    ) {
      return;
    }

    setNewsletterNotificationStatus(
      queuedNewsletterNotificationStatus
    );

    setQueuedNewsletterNotificationStatus("idle");
  }, [
    launcherFinished,
    queuedNewsletterNotificationStatus,
  ]);

  useEffect(() => {
    if (newsletterNotificationStatus === "idle") {
      return;
    }

    const timeoutId = window.setTimeout(() => {
      setNewsletterNotificationStatus("idle");
    }, 5000);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [newsletterNotificationStatus]);

  useEffect(() => {
    if (
      contactFormStatus !== "success" &&
      contactFormStatus !== "error"
    ) {
      return;
    }

    const timeoutId = window.setTimeout(() => {
      setContactFormStatus("idle");
    }, 3000);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [contactFormStatus]);

  const handleNewsletterSubmit = async (
    event: React.FormEvent<HTMLFormElement>
  ) => {
    event.preventDefault();

    if (newsletterSubmitting) return;

    const form = event.currentTarget;
    const formData = new FormData(form);

    const email = String(
      formData.get("email") ?? ""
    ).trim();

    if (!email) return;
    
    setNewsletterNotificationStatus("idle");

    setNewsletterSubmitting(true);

    try {
      const response = await fetch("/api/newsletter", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          email,
          language: lang,
        }),
      });

      const result = (await response.json().catch(() => null)) as
        | {
            success?: boolean;
            status?: string;
            error?: string;
          }
        | null;

      if (!response.ok || result?.success !== true) {
        throw new Error(
          result?.error ||
            `Newsletter request failed with status ${response.status}.`
        );
      }

      form.reset();

      if (result?.status === "already_active") {
        setNewsletterNotificationStatus("confirmed");
      } else {
        setNewsletterNotificationStatus("pending");
      }
    } catch (error) {
      console.error(
        "Newsletter subscription request failed.",
        error
      );
    } finally {
      setNewsletterSubmitting(false);
    }
  };

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

  const [activeportfolioIndex, setActiveportfolioIndex] = useState(0);

  const [isPortfolioFullscreen, setIsPortfolioFullscreen] = useState(false);
  const portfolioDragRef = useRef<{
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
    const imageCount = copy.portfolio.shots.length;

    if (!isPortfolioFullscreen || imageCount === 0) return;

    const previousOverflow = document.body.style.overflow;

    document.body.style.overflow = "hidden";

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsPortfolioFullscreen(false);
      }

      if (event.key === "ArrowRight") {
        event.preventDefault();

        setActiveportfolioIndex((current) =>
          (current + 1) % imageCount
        );
      }

      if (event.key === "ArrowLeft") {
        event.preventDefault();

        setActiveportfolioIndex((current) =>
          (current - 1 + imageCount) % imageCount
        );
      }
    };

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      document.body.style.overflow = previousOverflow;

      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [
    isPortfolioFullscreen,
    copy.portfolio.shots.length,
  ]);

  useEffect(() => {
    const mq = window.matchMedia("(max-width: 900px)");
    const mqPortrait = window.matchMedia("(orientation: portrait)");

    const apply = () => {
      const smallW = mq.matches;
      const portrait = mqPortrait.matches;
      const smallH = window.innerHeight < 740;

      setIsservicesAccordion(smallW);
      setIsservicesSlider(smallW && portrait && smallH);
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
    if (isservicesAccordion) return;
    if (hoveredPlan !== null) return;
    if (servicesPlans.length <= 1) return;

    const id = window.setInterval(() => {
      setActivePlan((p) => (p + 1) % servicesPlans.length);
    }, 2200);

    return () => window.clearInterval(id);
  }, [hoveredPlan, isservicesAccordion, servicesPlans.length]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let disposed = false;
    let raf = 0;
    let rafSize = 0;

    let model: THREE.Object3D | null = null;
    let modelGroup: THREE.Group | null = null;
    let modelMixer: THREE.AnimationMixer | null = null;
    let currentAction: THREE.AnimationAction | null = null;
    let currentActionName = "";

    const actionsByName = new Map<string, THREE.AnimationAction>();
    const ACTION_FADE_SECONDS = 0.75;

    // --------------------------------------------------
    // PARTICLE TRANSITION CONTROLLER
    // --------------------------------------------------

    type ParticleTransitionPhase =
      | "scatter"
      | "hold"
      | "reform";

    type ParticleTransition = {
      phase: ParticleTransitionPhase;
      elapsed: number;
      targetAction: string;
    };

    let particleTransition: ParticleTransition | null = null;
    let queuedAction: string | null = null;

    const PARTICLE_SCATTER_SECONDS = 0.85;
    const PARTICLE_HOLD_SECONDS = 0.12;
    const PARTICLE_REFORM_SECONDS = 1.05;

    // PARTICLE FOG CHARACTER

    // 72 % particles form the dense silhouette.
    const PARTICLE_FOG_CORE_SHARE = 0.72;

    // Dense particle cloud near the original body.
    const PARTICLE_FOG_CORE_MIN = 0.08;
    const PARTICLE_FOG_CORE_MAX = 0.28;

    // Sparse particles forming the outer fog.
    const PARTICLE_FOG_OUTER_MIN = 0.30;
    const PARTICLE_FOG_OUTER_MAX = 0.82;

    const particleBatches: {
      mesh: THREE.Mesh;
      triangleIndices: Uint32Array;
      barycentric: Float32Array;
      scatterOffsets: Float32Array;
      positionAttribute: THREE.BufferAttribute;
      points: THREE.Points;
    }[] = [];

    const skeletons = new Set<THREE.Skeleton>();
    const clock = new THREE.Clock();

    const lightPageParticleColor = new THREE.Color("#19191A");
    const darkPageParticleColor = new THREE.Color("#F4F4F4");
    const targetParticleColor = lightPageParticleColor.clone();

    const tempA = new THREE.Vector3();
    const tempB = new THREE.Vector3();
    const tempC = new THREE.Vector3();
    const tempPosition = new THREE.Vector3();
    const edgeAB = new THREE.Vector3();
    const edgeAC = new THREE.Vector3();
    const areaNormal = new THREE.Vector3();

    let modelSize = new THREE.Vector3(1, 1, 1);
    let hasModelBounds = false;

    type AmbientField = {
      points: THREE.Points;
      geometry: THREE.BufferGeometry;
      material: THREE.ShaderMaterial;
      positions: Float32Array;
      opacities: Float32Array;
      ages: Float32Array;
      lifeSpeeds: Float32Array;
      driftX: Float32Array;
      driftY: Float32Array;
      peakOpacities: Float32Array;
      respawn: (index: number) => void;
    };

    let ambientField: AmbientField | null = null;

    // --------------------------------------------------
    // RENDERER AND CAMERA
    // --------------------------------------------------

    const scene = new THREE.Scene();
    scene.background = null;

    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
      powerPreference: "high-performance",
      premultipliedAlpha: true,
    });

    renderer.setClearColor(background, 0);
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    const defaultCamera = new THREE.PerspectiveCamera(
      40,
      1,
      0.01,
      2000
    );

    const fitCamera = (width: number, height: number) => {
      const aspect = width / Math.max(1, height);
      defaultCamera.aspect = aspect;

      if (hasModelBounds) {
        const verticalFov = THREE.MathUtils.degToRad(
          defaultCamera.fov
        );
        const horizontalFov = 2 * Math.atan(
          Math.tan(verticalFov / 2) * aspect
        );

        const distanceY =
          modelSize.y * 0.5 / Math.tan(verticalFov / 2);
        const distanceX =
          modelSize.x * 0.5 / Math.tan(horizontalFov / 2);

        // --------------------------------------------------
        // RESPONSIVE CAMERA ZOOM
        // --------------------------------------------------

        const isMobileViewport = window.matchMedia(
          "(max-width: 900px), (pointer: coarse)"
        ).matches;

        const mobileZoom = isMobileViewport ? 0.50 : 1;

        const distance =
          (Math.max(distanceX, distanceY) + modelSize.z * 0.5) *
          1.18 *
          mobileZoom;

        defaultCamera.position.set(
          0,
          modelSize.y * 0.015,
          distance
        );
        defaultCamera.lookAt(0, 0, 0);
        defaultCamera.near = Math.max(
          0.01,
          distance - modelSize.length() * 2
        );
        defaultCamera.far =
          distance + modelSize.length() * 8 + 50;
      }

      defaultCamera.updateProjectionMatrix();
    };

    // --------------------------------------------------
    // PARTICLE TEXTURE
    // --------------------------------------------------

    const textureSize = 64;
    const textureData = new Uint8Array(
      textureSize * textureSize * 4
    );

    for (let y = 0; y < textureSize; y++) {
      for (let x = 0; x < textureSize; x++) {
        const dx = (x - 31.5) / 31.5;
        const dy = (y - 31.5) / 31.5;
        const distance = Math.sqrt(dx * dx + dy * dy);
        const alpha = Math.pow(Math.max(0, 1 - distance), 2);
        const offset = (y * textureSize + x) * 4;

        textureData[offset] = 255;
        textureData[offset + 1] = 255;
        textureData[offset + 2] = 255;
        textureData[offset + 3] = Math.round(alpha * 255);
      }
    }

    const particleTexture = new THREE.DataTexture(
      textureData,
      textureSize,
      textureSize,
      THREE.RGBAFormat
    );

    particleTexture.magFilter = THREE.LinearFilter;
    particleTexture.minFilter = THREE.LinearFilter;
    particleTexture.generateMipmaps = false;
    particleTexture.needsUpdate = true;

    // --------------------------------------------------
    // ANDROID PARTICLE MATERIAL
    // --------------------------------------------------

    const particleMaterial = new THREE.PointsMaterial({
      color: lightPageParticleColor,
      size: 0.016,
      sizeAttenuation: true,
      map: particleTexture,
      transparent: true,
      alphaTest: 0.025,
      opacity: 0.98,
      blending: THREE.NormalBlending,
      depthTest: true,
      depthWrite: true,
      toneMapped: false,
    });

    // --------------------------------------------------
    // ANIMATION CONTROLLER
    // --------------------------------------------------

    const getThemeAction = (sectionPosition: number): string => {
      if (sectionPosition < 2.0) return "walk";
      if (sectionPosition < 4.8) return "stand";

      return "idle";
    };

    const startAction = (
      name: string,
      repeat: boolean
    ): boolean => {
      const nextAction = actionsByName.get(name.toLowerCase());

      if (!nextAction) {
        console.warn(`[Android] Animation not found: ${name}`);
        return false;
      }

      if (currentAction === nextAction) return true;

      nextAction.reset();
      nextAction.enabled = true;
      nextAction.setEffectiveTimeScale(1);
      nextAction.setEffectiveWeight(1);
      nextAction.setLoop(
        repeat ? THREE.LoopRepeat : THREE.LoopOnce,
        repeat ? Infinity : 1
      );
      nextAction.clampWhenFinished = !repeat;
      nextAction.play();

      if (currentAction) {
        currentAction.crossFadeTo(
          nextAction,
          ACTION_FADE_SECONDS,
          false
        );
      } else {
        nextAction.fadeIn(0.35);
      }

      currentAction = nextAction;
      currentActionName = name.toLowerCase();

      return true;
    };
    
    // --------------------------------------------------
    // PARTICLE DISPERSION
    // --------------------------------------------------

    const beginParticleTransition = (targetAction: string) => {
      const nextName = targetAction.toLowerCase();

      if (!modelMixer || !actionsByName.has(nextName)) return;
      if (nextName === currentActionName) return;

      const direction = new THREE.Vector3();

      for (const batch of particleBatches) {
        const positions =
          batch.positionAttribute.array as Float32Array;

        const offsets = batch.scatterOffsets;

        const particleCount = batch.triangleIndices.length / 3;

        for (let i = 0; i < particleCount; i++) {
          const offset = i * 3;

          const px = positions[offset];
          const py = positions[offset + 1];
          const pz = positions[offset + 2];

                    
          direction.set(px, py, pz);

          if (direction.lengthSq() < 0.0001) {
            direction.set(
              Math.random() - 0.5,
              Math.random() - 0.5,
              Math.random() - 0.5
            );
          }

          direction.normalize();

          // Controlled directional turbulence.
          // Keeps the overall human silhouette readable.
          direction.x += (Math.random() - 0.5) * 0.45;
          direction.y += (Math.random() - 0.5) * 0.32 + 0.12;
          direction.z += (Math.random() - 0.5) * 0.45;

          direction.normalize();

          // Choose between dense core and outer fog.
          const isOuterFogParticle =
            Math.random() > PARTICLE_FOG_CORE_SHARE;

          const minSpread = isOuterFogParticle
            ? PARTICLE_FOG_OUTER_MIN
            : PARTICLE_FOG_CORE_MIN;

          const maxSpread = isOuterFogParticle
            ? PARTICLE_FOG_OUTER_MAX
            : PARTICLE_FOG_CORE_MAX;

          // Calculate distance relative to model height.
          const distance =
            modelSize.y *
            (minSpread + Math.random() * (maxSpread - minSpread));

          // Store final particle displacement.
          offsets[offset] = direction.x * distance;
          offsets[offset + 1] = direction.y * distance;
          offsets[offset + 2] = direction.z * distance;

        }
      }

      particleTransition = {
        phase: "scatter",
        elapsed: 0,
        targetAction: nextName,
      };

      queuedAction = null;

    };

    // --------------------------------------------------
    // PARTICLE TRANSITION UPDATE
    // --------------------------------------------------

    const updateParticleTransition = (
      dt: number,
      desiredAction: string
    ) => {
      if (!modelMixer) return;

      if (!particleTransition) {
        if (desiredAction !== currentActionName) {
          beginParticleTransition(desiredAction);
        }

        return;
      }

      const transition = particleTransition;

      if (transition.phase === "scatter") {
        transition.targetAction = desiredAction;
      } else {
        queuedAction =
          desiredAction !== currentActionName
            ? desiredAction
            : null;
      }

      transition.elapsed += dt;

      if (
        transition.phase === "scatter" &&
        transition.elapsed >= PARTICLE_SCATTER_SECONDS
      ) {
        const nextAction = transition.targetAction;

        if (nextAction !== currentActionName) {
          startAction(nextAction, true);
        }

        transition.phase = "hold";
        transition.elapsed = 0;

        return;
      }

      if (transition.phase === "hold") {
        if (queuedAction && queuedAction !== currentActionName) {
          const nextAction = queuedAction;

          queuedAction = null;

          startAction(nextAction, true);
          transition.targetAction = nextAction;
        }

        if (transition.elapsed >= PARTICLE_HOLD_SECONDS) {
          transition.phase = "reform";
          transition.elapsed = 0;
        }

        return;
      }

      if (
        transition.phase === "reform" &&
        transition.elapsed >= PARTICLE_REFORM_SECONDS
      ) {
        particleTransition = null;

        const nextAction = queuedAction;
        queuedAction = null;

        if (nextAction && nextAction !== currentActionName) {
          beginParticleTransition(nextAction);
        }
      }
    };

    // --------------------------------------------------
    // PARTICLE TRANSITION PROGRESS
    // --------------------------------------------------

    const getParticleScatterAmount = (): number => {
      if (!particleTransition) return 0;

      const transition = particleTransition;

      if (transition.phase === "hold") {
        return 1;
      }

      const duration =
        transition.phase === "scatter"
          ? PARTICLE_SCATTER_SECONDS
          : PARTICLE_REFORM_SECONDS;

      const progress = THREE.MathUtils.clamp(
        transition.elapsed / duration,
        0,
        1
      );

      if (transition.phase === "scatter") {
        return 1 - Math.pow(1 - progress, 3);
      }

      const smoothProgress =
        progress * progress * (3 - 2 * progress);

      return 1 - smoothProgress;
    };


    // --------------------------------------------------
    // AMBIENT PARTICLE FIELD
    // --------------------------------------------------

    const createAmbientField = () => {
      if (ambientField || !hasModelBounds) return;

      const isMobile = window.matchMedia(
        "(max-width: 900px), (pointer: coarse)"
      ).matches;
      const count = isMobile ? 240 : 650;

      const geometry = new THREE.BufferGeometry();
      const positions = new Float32Array(count * 3);
      const opacities = new Float32Array(count);
      const sizes = new Float32Array(count);
      const ages = new Float32Array(count);
      const lifeSpeeds = new Float32Array(count);
      const driftX = new Float32Array(count);
      const driftY = new Float32Array(count);
      const peakOpacities = new Float32Array(count);

      const respawn = (index: number) => {
        const offset = index * 3;

        positions[offset] =
          (Math.random() - 0.5) * modelSize.y * 3.2;
        positions[offset + 1] =
          (Math.random() - 0.5) * modelSize.y * 1.8;
        positions[offset + 2] =
          -modelSize.z * 0.5 - 0.7 - Math.random() * modelSize.y * 1.1;

        ages[index] = Math.random();
        lifeSpeeds[index] = 0.035 + Math.random() * 0.065;
        driftX[index] = (Math.random() - 0.5) * 0.018;
        driftY[index] = (Math.random() - 0.5) * 0.018;
        peakOpacities[index] = 0.30 + Math.random() * 0.25;
        sizes[index] = 0.020 + Math.random() * 0.030;

        const fade = Math.sin(Math.PI * ages[index]);
        opacities[index] = fade * fade * peakOpacities[index];
      };

      for (let i = 0; i < count; i++) respawn(i);

      const positionAttribute = new THREE.BufferAttribute(positions, 3);
      const opacityAttribute = new THREE.BufferAttribute(opacities, 1);
      const sizeAttribute = new THREE.BufferAttribute(sizes, 1);

      positionAttribute.setUsage(THREE.DynamicDrawUsage);
      opacityAttribute.setUsage(THREE.DynamicDrawUsage);

      geometry.setAttribute("position", positionAttribute);
      geometry.setAttribute("aOpacity", opacityAttribute);
      geometry.setAttribute("aSize", sizeAttribute);

      const material = new THREE.ShaderMaterial({
        uniforms: {
          uColor: { value: particleMaterial.color.clone() },
          uPixelRatio: { value: renderer.getPixelRatio() },
          uViewportHeight: { value: Math.max(1, canvas.clientHeight) },
        },
        vertexShader: `
          uniform float uPixelRatio;
          uniform float uViewportHeight;
          attribute float aOpacity;
          attribute float aSize;
          varying float vOpacity;

          void main() {
            vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
            gl_Position = projectionMatrix * mvPosition;
            float perspective = (uViewportHeight * 0.5) / max(0.1, -mvPosition.z);
            gl_PointSize = max(1.0, aSize * uPixelRatio * perspective);
            vOpacity = aOpacity;
          }
        `,
        fragmentShader: `
          uniform vec3 uColor;
          varying float vOpacity;

          void main() {
            vec2 centered = gl_PointCoord - vec2(0.5);
            float distanceFromCenter = length(centered);
            float circle = 1.0 - smoothstep(0.12, 0.5, distanceFromCenter);
            float alpha = circle * vOpacity;
            if (alpha < 0.005) discard;
            gl_FragColor = vec4(uColor, alpha);
            #include <colorspace_fragment>
          }
        `,
        transparent: true,
        depthTest: true,
        depthWrite: false,
        blending: THREE.NormalBlending,
        toneMapped: false,
      });

      const points = new THREE.Points(geometry, material);
      points.frustumCulled = false;
      points.renderOrder = 2;
      scene.add(points);

      ambientField = {
        points,
        geometry,
        material,
        positions,
        opacities,
        ages,
        lifeSpeeds,
        driftX,
        driftY,
        peakOpacities,
        respawn,
      };
    };

    // --------------------------------------------------
    // RESIZE HANDLING
    // --------------------------------------------------

    const setSize = () => {
      rafSize = 0;

      const rect = canvas.getBoundingClientRect();
      const width = Math.max(1, Math.round(rect.width));
      const height = Math.max(1, Math.round(rect.height));
      const dpr = Math.min(2, window.devicePixelRatio || 1);

      renderer.setPixelRatio(dpr);
      renderer.setSize(width, height, false);
      fitCamera(width, height);

      if (ambientField) {
        ambientField.material.uniforms.uPixelRatio.value = dpr;
        ambientField.material.uniforms.uViewportHeight.value = height;
      }
    };

    const requestResize = () => {
      if (!rafSize) rafSize = requestAnimationFrame(setSize);
    };

    const resizeObserver = new ResizeObserver(requestResize);
    resizeObserver.observe(canvas);

    window.addEventListener("resize", requestResize);
    window.addEventListener("orientationchange", requestResize);
    setSize();

    // --------------------------------------------------
    // GLB LOADING AND SURFACE SAMPLING
    // --------------------------------------------------

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
        modelGroup = new THREE.Group();
        scene.add(modelGroup);
        modelGroup.add(root);
        modelGroup.updateMatrixWorld(true);

        const bounds = new THREE.Box3().setFromObject(root);
        const center = bounds.getCenter(new THREE.Vector3());
        modelSize = bounds.getSize(new THREE.Vector3());
        hasModelBounds = true;

        modelGroup.position.set(-center.x, -center.y, -center.z);
        modelGroup.updateMatrixWorld(true);

        const meshes: THREE.Mesh[] = [];
        root.traverse((object) => {
          const mesh = object as THREE.Mesh;
          if (!mesh.isMesh || !mesh.geometry.getAttribute("position")) return;

          meshes.push(mesh);
          const skinnedMesh = mesh as THREE.SkinnedMesh;
          if (skinnedMesh.isSkinnedMesh) {
            skeletons.add(skinnedMesh.skeleton);
          }
        });

        for (const skeleton of skeletons) skeleton.update();

        const surfaceData = meshes
          .map((mesh) => {
            const geometry = mesh.geometry;
            const position = geometry.getAttribute("position");
            const index = geometry.getIndex();
            const elementCount = index ? index.count : position.count;
            const triangleCount = Math.floor(elementCount / 3);
            const cumulativeAreas = new Float64Array(triangleCount);
            let totalArea = 0;

            for (let triangle = 0; triangle < triangleCount; triangle++) {
              const offset = triangle * 3;
              const i0 = index ? index.getX(offset) : offset;
              const i1 = index ? index.getX(offset + 1) : offset + 1;
              const i2 = index ? index.getX(offset + 2) : offset + 2;

              mesh.getVertexPosition(i0, tempA);
              mesh.getVertexPosition(i1, tempB);
              mesh.getVertexPosition(i2, tempC);

              tempA.applyMatrix4(mesh.matrixWorld);
              tempB.applyMatrix4(mesh.matrixWorld);
              tempC.applyMatrix4(mesh.matrixWorld);

              edgeAB.subVectors(tempB, tempA);
              edgeAC.subVectors(tempC, tempA);
              const area = areaNormal.crossVectors(edgeAB, edgeAC).length() * 0.5;

              totalArea += area;
              cumulativeAreas[triangle] = totalArea;
            }

            return {
              mesh,
              index,
              triangleCount,
              cumulativeAreas,
              totalArea,
            };
          })
          .filter((item) => item.totalArea > 0 && item.triangleCount > 0);

        const totalSurfaceArea = surfaceData.reduce(
          (sum, item) => sum + item.totalArea,
          0
        );

        const isMobile = window.matchMedia(
          "(max-width: 900px), (pointer: coarse)"
        ).matches;
        const particleBudget = isMobile ? 9000 : 24000;

        if (totalSurfaceArea > 0) {
          for (const surface of surfaceData) {
            const {
              mesh,
              index,
              triangleCount,
              cumulativeAreas,
              totalArea,
            } = surface;

            const particleCount = Math.max(
              1,
              Math.round(particleBudget * (totalArea / totalSurfaceArea))
            );

            const triangleIndices = new Uint32Array(particleCount * 3);
            const barycentric = new Float32Array(particleCount * 3);
            const positions = new Float32Array(particleCount * 3);
            const scatterOffsets = new Float32Array(particleCount * 3);
            const positionAttribute = new THREE.BufferAttribute(positions, 3);
            positionAttribute.setUsage(THREE.DynamicDrawUsage);

            const particleGeometry = new THREE.BufferGeometry();
            particleGeometry.setAttribute("position", positionAttribute);

            for (let i = 0; i < particleCount; i++) {
              const targetArea = Math.random() * totalArea;
              let low = 0;
              let high = triangleCount - 1;

              while (low < high) {
                const middle = Math.floor((low + high) / 2);
                if (cumulativeAreas[middle] < targetArea) {
                  low = middle + 1;
                } else {
                  high = middle;
                }
              }

              const triangleOffset = low * 3;
              const i0 = index ? index.getX(triangleOffset) : triangleOffset;
              const i1 = index ? index.getX(triangleOffset + 1) : triangleOffset + 1;
              const i2 = index ? index.getX(triangleOffset + 2) : triangleOffset + 2;
              const offset = i * 3;

              triangleIndices[offset] = i0;
              triangleIndices[offset + 1] = i1;
              triangleIndices[offset + 2] = i2;

              const rootRandom = Math.sqrt(Math.random());
              const secondRandom = Math.random();
              barycentric[offset] = 1 - rootRandom;
              barycentric[offset + 1] = rootRandom * (1 - secondRandom);
              barycentric[offset + 2] = rootRandom * secondRandom;
            }

            const points = new THREE.Points(
              particleGeometry,
              particleMaterial
            );
            points.frustumCulled = false;
            points.renderOrder = 1;
            scene.add(points);

            particleBatches.push({
              mesh,
              triangleIndices,
              barycentric,
              scatterOffsets,
              positionAttribute,
              points,
            });

            mesh.visible = false;
          }
        } else {
          console.error("[Android] The GLB contains no usable triangle surfaces.");
        }

        // --------------------------------------------------
        // BLENDER ACTIONS
        // --------------------------------------------------

        if (gltf.animations.length > 0) {
          modelMixer = new THREE.AnimationMixer(root);

          for (const clip of gltf.animations) {
            actionsByName.set(
              clip.name.toLowerCase(),
              modelMixer.clipAction(clip)
            );
          }

          const initialAction = getThemeAction(
            scrollTRef.current
          );

          if (!startAction(initialAction, true)) {
            if (actionsByName.has("idle")) {
              startAction("idle", true);
            }
          }
        } else {
          console.warn("[Android] No Blender animations were found in the GLB.");
        }

        createAmbientField();
        setSize();
        setReady(true);

        },
        undefined,
        (error) => {
          if (!disposed) {
            console.error("[Android] GLB loading failed:", error);
            setReady(false);
          }
        }
      );

    // --------------------------------------------------
    // RENDER LOOP
    // --------------------------------------------------

    const animate = () => {
      raf = requestAnimationFrame(animate);

      const dt = Math.min(clock.getDelta(), 0.05);

      // --------------------------------------------------
      // ANIMATION TRANSITION CONTROL
      // --------------------------------------------------

      const currentSectionPosition = THREE.MathUtils.clamp(
        scrollTRef.current,
        0,
        SECTION_COUNT
      );

      const desiredAction = getThemeAction(
        currentSectionPosition
      );

      updateParticleTransition(dt, desiredAction);

      if (modelMixer) {
        modelMixer.update(dt);
      }

      // --------------------------------------------------
      // MODEL AND SKELETON UPDATE
      // --------------------------------------------------

      if (modelGroup) {
        modelGroup.updateMatrixWorld(true);
      }

      for (const skeleton of skeletons) {
        skeleton.update();
      }

      const scatterAmount = getParticleScatterAmount();

      if (modelGroup) modelGroup.updateMatrixWorld(true);
      for (const skeleton of skeletons) skeleton.update();

      for (const batch of particleBatches) {
        const positions = batch.positionAttribute.array as Float32Array;
        const triangleIndices = batch.triangleIndices;
        const barycentric = batch.barycentric;
        const particleCount = triangleIndices.length / 3;

        for (let i = 0; i < particleCount; i++) {
          const offset = i * 3;
          const i0 = triangleIndices[offset];
          const i1 = triangleIndices[offset + 1];
          const i2 = triangleIndices[offset + 2];

          const w0 = barycentric[offset];
          const w1 = barycentric[offset + 1];
          const w2 = barycentric[offset + 2];

          batch.mesh.getVertexPosition(i0, tempA);
          batch.mesh.getVertexPosition(i1, tempB);
          batch.mesh.getVertexPosition(i2, tempC);

          tempPosition
            .copy(tempA)
            .multiplyScalar(w0)
            .addScaledVector(tempB, w1)
            .addScaledVector(tempC, w2)
            .applyMatrix4(batch.mesh.matrixWorld);

          positions[offset] =
            tempPosition.x +
            batch.scatterOffsets[offset] * scatterAmount;

          positions[offset + 1] =
            tempPosition.y +
            batch.scatterOffsets[offset + 1] * scatterAmount;

          positions[offset + 2] =
            tempPosition.z +
            batch.scatterOffsets[offset + 2] * scatterAmount;
        }

        batch.positionAttribute.needsUpdate = true;
      }

      // --------------------------------------------------
      // THEME SYNCHRONIZATION
      // --------------------------------------------------

      const themeProgress = THREE.MathUtils.clamp(
        invertFactorRef.current,
        0,
        1
      );

      targetParticleColor
        .copy(lightPageParticleColor)
        .lerp(darkPageParticleColor, themeProgress);

      particleMaterial.color.lerp(
        targetParticleColor,
        1 - Math.exp(-dt * 6)
      );

      // Particles become larger and softer during dispersion.
      particleMaterial.size = THREE.MathUtils.lerp(
        0.016,
        0.024,
        scatterAmount
      );

      // Reduce opacity to create a fog-like appearance.
      particleMaterial.opacity = THREE.MathUtils.lerp(
        0.98,
        0.76,
        scatterAmount
      );

      // --------------------------------------------------
      // AMBIENT PARTICLE ANIMATION
      // --------------------------------------------------

      if (ambientField) {
        const field = ambientField;

        for (let i = 0; i < field.ages.length; i++) {
          field.ages[i] += field.lifeSpeeds[i] * dt;

          if (field.ages[i] >= 1) {
            field.respawn(i);
            continue;
          }

          const offset = i * 3;
          field.positions[offset] += field.driftX[i] * dt;
          field.positions[offset + 1] += field.driftY[i] * dt;

          const fade = Math.sin(Math.PI * field.ages[i]);
          field.opacities[i] =
            fade * fade * field.peakOpacities[i];
        }

        const positions = field.geometry.getAttribute(
          "position"
        ) as THREE.BufferAttribute;
        const opacities = field.geometry.getAttribute(
          "aOpacity"
        ) as THREE.BufferAttribute;

        positions.needsUpdate = true;
        opacities.needsUpdate = true;
        field.material.uniforms.uColor.value.copy(
          particleMaterial.color
        );
      }

      renderer.render(scene, defaultCamera);
    };

    animate();

    // --------------------------------------------------
    // CLEANUP
    // --------------------------------------------------

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);

      if (rafSize) cancelAnimationFrame(rafSize);
      resizeObserver.disconnect();

      window.removeEventListener("resize", requestResize);
      window.removeEventListener("orientationchange", requestResize);

      if (modelMixer) {
        modelMixer.stopAllAction();
        modelMixer.uncacheRoot(modelMixer.getRoot());
        modelMixer = null;
      }

      particleTransition = null;
      queuedAction = null;

      if (ambientField) {
        scene.remove(ambientField.points);
        ambientField.geometry.dispose();
        ambientField.material.dispose();
        ambientField = null;
      }

      for (const batch of particleBatches) {
        scene.remove(batch.points);
        batch.points.geometry.dispose();
      }
      particleBatches.length = 0;

      if (model) {
        model.traverse((object) => {
          if (!(object instanceof THREE.Mesh)) return;

          object.geometry.dispose();

          const materials = Array.isArray(object.material)
            ? object.material
            : [object.material];

          materials.forEach((material) => material.dispose());
        });

        model = null;
      }

      if (modelGroup) {
        scene.remove(modelGroup);
        modelGroup = null;
      }

      particleMaterial.dispose();
      particleTexture.dispose();
      dracoLoader.dispose();
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
  const servicesOpacity = sectionOpacity(t, 2);
  const portfolioOpacity = sectionOpacity(t, 3);
  const reviewsOpacity = sectionOpacity(t, 4);
  const contactOpacity = sectionOpacity(t, 5);
  const footerOpacity = clamp01((t - 6) / FADE);

  const backgroundFadeIn = clamp01((t - 1.8) / 0.4);
  const backgroundFadeOut = clamp01((5 - t) / 0.4);

  const invertFactor = clamp01(
    backgroundFadeIn * backgroundFadeOut
  );

  useLayoutEffect(() => {
    invertFactorRef.current = invertFactor;
  }, [invertFactor]);

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

        {newsletterNotificationStatus !== "idle" ? (
        <div
          role={
            newsletterNotificationStatus === "invalid"
              ? "alert"
              : "status"
          }
          aria-live={
            newsletterNotificationStatus === "invalid"
              ? "assertive"
              : "polite"
          }
          aria-atomic="true"
          className={[
            "fixed inset-x-0 top-6 z-[10000]",
            "mx-auto w-[calc(100%-32px)] max-w-[520px]",
            "rounded-2xl border border-[#19191A]/20",
            "bg-[#C7C7C7]/95 px-6 py-4",
            "text-center text-sm font-medium text-[#19191A]",
            "shadow-[0_18px_50px_rgba(0,0,0,0.18)]",
            "backdrop-blur-md",
            "animate-[contactPopupIn_220ms_ease-out]",
          ].join(" ")}
        >
          {copy.footer.newsletterStatus[
            newsletterNotificationStatus
          ]}
        </div>
      ) : null}

        <div
          className="first_section__canvas_wrap"
          aria-hidden
          style={{ opacity: ready ? 1 : 0, transition: "opacity 600ms ease" }}
        >
          <canvas ref={canvasRef} className="first_section__canvas" style={{ filter: "none" }} />
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
          <div className="first_section__grid">
            <div className="first_section__left">
              <h1 className="first_section__brand">{copy.hero.brand}</h1>

              <div className="first_section__panel bg-white/5 border border-white/10 backdrop-blur-md rounded-2xl p-6">
                <p className="first_section__desc">
                  {copy.hero.desc}
                </p>

                <div className="first_section__actions">
                  <a
                    href="#services"
                    className="first_section__btn first_section__btn--primary"
                    onClick={(e) => {
                      e.preventDefault();
                      scrollToSection("services");
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
          data-layer="services"
          className={[
            "absolute inset-0 z-[3] grid px-[clamp(20px,6vw,80px)] py-[clamp(20px,6vw,80px)]",
            "place-items-center",
          ].join(" ")}
          style={{
            opacity: servicesOpacity,
            pointerEvents: servicesOpacity > 0.05 ? "auto" : "none",
            transition: "opacity 120ms linear",
            overflowY: isservicesAccordion ? "auto" : "visible",
          }}
          aria-hidden={servicesOpacity < 0.05}
        >
          <div className="w-full max-w-[1200px]">
            <div className="text-center mb-8">
              <h2 className="text-[#19191A] text-3xl md:text-5xl tracking-[0.12em] uppercase opacity-95">
                {copy.services.title}
              </h2>
              <p className="text-[#F4F4F4]/70 mt-3 max-w-[60ch] mx-auto">
                {copy.services.subtitle}
              </p>
            </div>

            {isservicesAccordion ? (
              isservicesSlider ? (
                (() => {
                  const servicesSliderIdx = displayedOpenPlan ?? 0;
                  const servicesSliderPlan = servicesPlans[servicesSliderIdx] ?? servicesPlans[0];

                  if (!servicesSliderPlan) return null;

                  return (
                    <div>
                      <div
                        key={servicesSliderIdx}
                        className={[
                          "rounded-2xl border border-[#F4F4F4]/15 bg-[#19191A]/70 backdrop-blur-md select-none",
                          servicesSlideDir === 1
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
                          <div className="text-[#F4F4F4] font-semibold text-lg">{servicesSliderPlan.name}</div>
                          <div className="text-[#F4F4F4]/90 mt-1">{servicesSliderPlan.price}</div>

                          <div className="text-[#F4F4F4]/80 leading-relaxed mt-3">{servicesSliderPlan.desc}</div>

                          <ul className="mt-4 space-y-2 text-[#F4F4F4]/85">
                            {servicesSliderPlan.details.map((d) => (
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
                                "services-contact-btn", "inline-flex w-full items-center justify-center rounded-xl border px-4 py-3 font-semibold transition cursor-pointer",
                                "border-[#F4F4F4]/20 bg-[#19191A]/55 text-[#F4F4F4]",
                                "hover:border-[#F4F4F4]/35 hover:bg-[#19191A]/45",
                              ].join(" ")}
                            >
                              {copy.services.order}
                            </button>
                          </div>
                        </div>
                      </div>

                      <div className="mt-4 flex justify-center gap-2" aria-hidden>
                        {servicesPlans.map((_, idx) => (
                          <span
                            key={idx}
                            className={[
                              "h-2 rounded-full transition-all duration-200",
                              servicesSliderIdx === idx ? "w-8 bg-[#F4F4F4]/80" : "w-2 bg-[#F4F4F4]/30",
                            ].join(" ")}
                          />
                        ))}
                      </div>
                    </div>
                  );
                })()
              ) : (
                <div className="space-y-3">
                  {servicesPlans.map((p, i) => {
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
                                  "services-contact-btn",
                                  "inline-flex w-full items-center justify-center rounded-xl border px-4 py-3 font-semibold transition cursor-pointer",
                                  "border-[#F4F4F4]/20 bg-[#19191A]/55 text-[#F4F4F4]",
                                  "hover:border-[#F4F4F4]/35 hover:bg-[#19191A]/45",
                                ].join(" ")}
                              >
                                {copy.services.order}
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
                  servicesPlans.length === 1
                    ? "md:grid-cols-1 max-w-[430px] mx-auto"
                    : servicesPlans.length === 2
                    ? "md:grid-cols-2 max-w-[900px] mx-auto"
                    : "md:grid-cols-3",
                ].join(" ")}
              >
                {servicesPlans.map((p, i) => {
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
                          onClick={() => scrollToSection("portfolio")}
                          className={[
                            "services-contact-btn",
                            "inline-flex w-full items-center justify-center rounded-xl border px-4 py-3 font-semibold transition",
                            "border-[#F4F4F4]/20 bg-[#19191A]/70 backdrop-blur-md text-[#F4F4F4] cursor-pointer",
                            "hover:border-[#F4F4F4]/35 hover:bg-[#19191A]/55",
                            active ? "border-[#F4F4F4]/35 bg-[#19191A]/55" : "",
                          ].join(" ")}
                        >
                          {copy.services.order}
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
          data-layer="portfolio"
          className="absolute inset-0 z-[3] grid px-[clamp(20px,6vw,80px)] py-[clamp(18px,4svh,52px)] place-items-center overflow-hidden max-[900px]:overflow-y-auto"
          style={{
            opacity: portfolioOpacity,
            pointerEvents: portfolioOpacity > 0.05 ? "auto" : "none",
            transition: "opacity 120ms linear",
          }}
          aria-hidden={portfolioOpacity < 0.05}
        >
          <div className="w-full max-w-[1600px] flex flex-col items-center">
            <div className="text-center mb-12">
              <h2 className="text-[#F4F4F4] text-3xl md:text-5xl tracking-[0.12em] uppercase opacity-95">
                {copy.portfolio.title}
              </h2>
              <p className="text-[#F4F4F4]/70 mt-3 max-w-[70ch] mx-auto">
                {copy.portfolio.subtitle}
              </p>
            </div>

            {(() => {
              const cards = copy.portfolio.shots.map((shot, index) => ({
                title: shot.title,
                desc: shot.desc,
                imgSrc: `/images/chart_${index + 1}.webp`,
              }));

              const nextSlide = () => {
                setActiveportfolioIndex((prev) => (prev + 1) % cards.length);
              };

              const prevSlide = () => {
                setActiveportfolioIndex((prev) => (prev - 1 + cards.length) % cards.length);
              };

              return (
                <div className="w-full flex flex-col items-center">
                  <div
                    data-no-radial
                    className="relative w-full h-[460px] md:h-[720px] flex items-center justify-center overflow-hidden perspective-1000 select-none cursor-grab active:cursor-grabbing"
                    style={{ touchAction: "pan-y" }}
                    onPointerDown={(e) => {
                      portfolioDragRef.current = {
                        startX: e.clientX,
                        startY: e.clientY,
                        pointerId: e.pointerId,
                        fired: false,
                      };

                      const target = e.target;

                      const card =
                        target instanceof HTMLElement
                          ? target.closest<HTMLElement>("[data-portfolio-card]")
                          : null;

                      (card ?? e.currentTarget).setPointerCapture(e.pointerId);
                    }}
                    onPointerMove={(e) => {
                      const drag = portfolioDragRef.current;

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
                      if (portfolioDragRef.current.pointerId === e.pointerId) {
                        portfolioDragRef.current.pointerId = null;
                      }
                    }}
                    onPointerCancel={(e) => {
                      if (portfolioDragRef.current.pointerId === e.pointerId) {
                        portfolioDragRef.current.pointerId = null;
                      }
                    }}
                  >
                    {cards.map((card, index) => {
                      const total = cards.length;
                      const diff = (index - activeportfolioIndex + total) % total;

                      let position = "hidden";
                      if (diff === 0) position = "center";
                      else if (diff === 1 || diff === -(total - 1)) position = "right";
                      else if (diff === total - 1 || diff === -1) position = "left";

                      const isCenter = position === "center";
                      const isLeft = position === "left";
                      const isRight = position === "right";

                      const isPortfolioActive = portfolioOpacity > 0.05;

                      return (
                        <div
                          key={index}
                          data-portfolio-card
                          onClick={() => {
                            if (portfolioDragRef.current.fired || !isPortfolioActive) return;

                            setActiveportfolioIndex(index);

                            if (isCenter) {
                              setIsPortfolioFullscreen(true);
                            }
                          }}
                          className={`absolute w-[85vw] md:w-[1000px] rounded-2xl border bg-[#19191A]/95 backdrop-blur-md overflow-hidden select-none transition-all duration-500 ease-out shadow-2xl ${
                            !isPortfolioActive
                              ? "pointer-events-none opacity-0 z-0" 
                              : isCenter
                              ? "z-30 border-[#F4F4F4]/40 scale-100 opacity-100 translate-x-0 pointer-events-auto cursor-pointer"
                              : isLeft
                              ? "z-10 border-[#F4F4F4]/10 scale-75 opacity-20 -translate-x-[85%] md:-translate-x-[105%] pointer-events-auto cursor-pointer"
                              : isRight
                              ? "z-10 border-[#F4F4F4]/10 scale-75 opacity-20 translate-x-[85%] md:translate-x-[105%] pointer-events-auto cursor-pointer"
                              : "z-0 scale-50 opacity-0 pointer-events-none translate-x-0"
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

                  {isPortfolioFullscreen &&
                  cards[activeportfolioIndex] &&
                  createPortal(
                    <div
                      className="portfolio-lightbox"
                      role="dialog"
                      aria-modal="true"
                      aria-label={cards[activeportfolioIndex].title}
                      onClick={() => setIsPortfolioFullscreen(false)}
                    >

                      <button
                        type="button"
                        className="portfolio-lightbox__close"
                        aria-label="Zavrieť galériu"
                        onClick={(event) => {
                          event.stopPropagation();
                          setIsPortfolioFullscreen(false);
                        }}
                      >
                        <X size={26} strokeWidth={1.7} />
                      </button>


                      <button
                        type="button"
                        className="portfolio-lightbox__nav portfolio-lightbox__nav--prev"
                        aria-label="Predchádzajúci obrázok"
                        onClick={(event) => {
                          event.stopPropagation();
                          prevSlide();
                        }}
                      >
                        <ChevronLeft size={34} strokeWidth={1.6} />
                      </button>


                      <div
                        className="portfolio-lightbox__content"
                        onClick={(event) => event.stopPropagation()}
                      >
                        <div className="portfolio-lightbox__image-wrap">

                          <Image
                            src={cards[activeportfolioIndex].imgSrc}
                            alt={cards[activeportfolioIndex].title}
                            fill
                            priority
                            sizes="100vw"
                            className="portfolio-lightbox__image"
                            draggable={false}
                          />

                        </div>

                        <div className="portfolio-lightbox__caption">
                          <h3>
                            {cards[activeportfolioIndex].title}
                          </h3>

                          <p>
                            {cards[activeportfolioIndex].desc}
                          </p>
                        </div>
                      </div>


                      <button
                        type="button"
                        className="portfolio-lightbox__nav portfolio-lightbox__nav--next"
                        aria-label="Nasledujúci obrázok"
                        onClick={(event) => {
                          event.stopPropagation();
                          nextSlide();
                        }}
                      >
                        <ChevronRight size={34} strokeWidth={1.6} />
                      </button>

                    </div>,
                    document.body
                  )}
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
            @keyframes contactPopupIn {
              from {
                opacity: 0;
                transform: translateY(-12px) scale(0.97);
              }

              to {
                opacity: 1;
                transform: translateY(0) scale(1);
              }
            }
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
              </form>
              {contactFormStatus === "success" &&
              contactStatusCopy?.success ? (
                <div
                  role="status"
                  aria-live="polite"
                  aria-atomic="true"
                  className={[
                    "fixed inset-x-0 top-6 z-[10000]",
                    "mx-auto w-[calc(100%-32px)] max-w-[520px]",
                    "rounded-2xl border border-[#19191A]/20",
                    "bg-[#C7C7C7]/95 px-6 py-4",
                    "text-center text-sm font-medium text-[#19191A]",
                    "shadow-[0_18px_50px_rgba(0,0,0,0.18)]",
                    "backdrop-blur-md",
                    "animate-[contactPopupIn_220ms_ease-out]",
                  ].join(" ")}
                >
                  {contactStatusCopy.success}
                </div>
              ) : null}

              {contactFormStatus === "error" &&
              contactStatusCopy?.error ? (
                <div
                  role="alert"
                  aria-live="assertive"
                  aria-atomic="true"
                  className={[
                    "fixed inset-x-0 top-6 z-[10000]",
                    "mx-auto w-[calc(100%-32px)] max-w-[520px]",
                    "rounded-2xl border border-[#19191A]/30",
                    "bg-[#C7C7C7]/95 px-6 py-4",
                    "text-center text-sm font-medium text-[#19191A]",
                    "shadow-[0_18px_50px_rgba(0,0,0,0.18)]",
                    "backdrop-blur-md",
                    "animate-[contactPopupIn_220ms_ease-out]",
                  ].join(" ")}
                >
                  {contactStatusCopy.error}
                </div>
              ) : null}
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
                    onSubmit={handleNewsletterSubmit}
                    aria-busy={newsletterSubmitting}
                    className="mt-4 flex flex-col sm:flex-row gap-3 justify-center md:justify-start"
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
                      disabled={newsletterSubmitting}
                      className={[
                        "shrink-0 rounded-xl border px-5 py-3 font-semibold transition",
                        "border-[#19191A]/15 bg-[#E7E7E7]/90 text-[#19191A]",
                        "hover:-translate-y-0.5 hover:bg-[rgba(34,197,94,0.34)] hover:border-[rgba(134,239,172,0.38)] hover:text-[#F4F4F4] hover:shadow-[inset_0_1px_0_rgba(255,255,255,0.16),0_8px_26px_rgba(34,197,94,0.12)]",
                        "disabled:cursor-not-allowed disabled:opacity-60 disabled:hover:translate-y-0",
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