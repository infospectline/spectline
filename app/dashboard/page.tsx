"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Bell,
  LogOut,
  Menu,
  Settings,
  UserRound,
  X,
} from "lucide-react";
import { createAuthClient } from "better-auth/react";
import sk from "@/lib/json/sk.json";
import en from "@/lib/json/en.json";

const authClient = createAuthClient();

type Lang = "sk" | "en";

type DashboardCopy = {
  logoAlt: string;
  loading: string;
  userFallback: string;

  header: {
    notificationsAriaLabel: string;
    menuAriaLabel: string;
  };

  notifications: {
    title: string;
    empty: string;
    closeAriaLabel: string;
  };

  sidebar: {
    title: string;
    closeAriaLabel: string;
    account: string;
    settings: string;
    logout: string;
    loggingOut: string;
  };
};

type LocalizedCopy = {
  dashboard?: DashboardCopy;
};

const COPY_BY_LANG: Record<Lang, LocalizedCopy> = {
  sk: sk as unknown as LocalizedCopy,
  en: en as unknown as LocalizedCopy,
};

export default function DashboardPage() {
  const router = useRouter();
  const { data: session, isPending } = authClient.useSession();

  const [lang, setLang] = useState<Lang>("sk");
  const [hydrated, setHydrated] = useState(false);

  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const [isLoggingOut, setIsLoggingOut] = useState(false);

  const copy = COPY_BY_LANG[lang].dashboard;

  useEffect(() => {
    const savedLanguage = window.localStorage.getItem("spectline-lang");

    if (savedLanguage === "sk" || savedLanguage === "en") {
      setLang(savedLanguage);
    }

    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;

    document.documentElement.lang = lang;
  }, [hydrated, lang]);

  useEffect(() => {
    if (!isPending && !session) {
      router.replace("/login");
    }
  }, [isPending, session, router]);

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key !== "Escape") return;

      setSidebarOpen(false);
      setNotificationsOpen(false);
    }

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  useEffect(() => {
    if (!sidebarOpen) return;

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, [sidebarOpen]);

  async function handleLogout() {
    if (isLoggingOut) return;

    setIsLoggingOut(true);

    try {
      await authClient.signOut();

      router.replace("/login");
      router.refresh();
    } finally {
      setIsLoggingOut(false);
    }
  }

  if (!hydrated || !copy) {
    return <div className="min-h-screen bg-[#C7C7C7]" />;
  }

  if (isPending) {
    return (
      <main
        className="
          min-h-screen
          bg-[#C7C7C7]
          text-[#19191A]
          grid place-items-center
          px-6
        "
      >
        <div className="flex flex-col items-center gap-5">
          <div
            className="
              h-14 w-14
              rounded-full
              border-4 border-[#19191A]/10
              border-t-[#19191A]
              animate-spin
            "
            aria-hidden="true"
          />

          <p className="text-sm font-medium text-[#19191A]/60">
            {copy.loading}
          </p>
        </div>
      </main>
    );
  }

  if (!session) return null;

  const userName = session.user.name?.trim() || copy.userFallback;

  const blueHoverClass = `
    hover:-translate-y-0.5
    hover:bg-[rgba(59,130,246,0.34)]
    hover:border-[rgba(147,197,253,0.38)]
    hover:text-[rgba(244,244,244,0.96)]
    hover:shadow-[inset_0_1px_0_rgba(255,255,255,0.18),0_8px_26px_rgba(59,130,246,0.12)]
  `;

  const redHoverClass = `
    hover:-translate-y-0.5
    hover:bg-[rgba(239,68,68,0.32)]
    hover:border-[rgba(252,165,165,0.38)]
    hover:text-[rgba(244,244,244,0.96)]
    hover:shadow-[inset_0_1px_0_rgba(255,255,255,0.16),0_8px_26px_rgba(239,68,68,0.12)]
  `;

  const iconButtonClass = `
    inline-flex h-11 w-11
    items-center justify-center
    rounded-xl
    bg-[#E7E7E7]/90
    border border-[#19191A]/15
    text-[#19191A]/75
    shadow-sm
    cursor-pointer
    transition-all duration-300
    focus:outline-none
    focus-visible:ring-2
    focus-visible:ring-[rgba(59,130,246,0.28)]
    focus-visible:ring-offset-2
    focus-visible:ring-offset-[#C7C7C7]
  `;

  const sidebarButtonClass = `
    w-full
    flex items-center gap-3
    rounded-2xl
    bg-[#E7E7E7]/90
    border border-[#19191A]/15
    px-5 py-4
    text-left
    text-base font-semibold
    text-[#19191A]
    shadow-sm
    cursor-pointer
    transition-all duration-300
    focus:outline-none
    focus-visible:ring-2
    focus-visible:ring-offset-2
    focus-visible:ring-offset-[#E7E7E7]
  `;

  return (
    <div
      className="
        relative
        min-h-screen
        overflow-hidden
        bg-[#C7C7C7]
        text-[#19191A]
      "
    >
      <div
        aria-hidden="true"
        className="
          pointer-events-none
          absolute inset-0 z-0
          grid place-items-center
          overflow-hidden
        "
      >
        <img
          src="/images/logo.png"
          alt=""
          className="
            w-[min(900px,88vw)]
            max-h-[78vh]
            object-contain
            brightness-0
            opacity-[0.07]
            select-none
          "
        />
      </div>

      <header
        className="
          relative z-20
          border-b border-[#19191A]/10
          bg-[#E7E7E7]/55
          shadow-[0_8px_28px_rgba(25,25,26,0.06)]
          backdrop-blur-xl
        "
      >
        <div
          className="
            mx-auto
            flex min-h-[88px]
            max-w-[1600px]
            items-center justify-between
            gap-5
            px-5 sm:px-8
          "
        >
          <div className="flex min-w-0 items-center gap-4">
            <img
              src="/images/logo.png"
              alt={copy.logoAlt}
              className="
                h-12 w-12
                shrink-0
                object-contain
                brightness-0
                opacity-80
              "
            />

            <div className="min-w-0">
              <p
                className="
                  truncate
                  text-base font-semibold
                  tracking-[0.04em]
                  text-[#19191A]
                  sm:text-lg
                "
              >
                {userName}
              </p>
            </div>
          </div>

          <div className="relative flex shrink-0 items-center gap-2">
            <button
              type="button"
              aria-label={copy.header.notificationsAriaLabel}
              aria-expanded={notificationsOpen}
              aria-controls="dashboard-notifications"
              onClick={() => {
                setNotificationsOpen((current) => !current);
                setSidebarOpen(false);
              }}
              className={`${iconButtonClass} ${blueHoverClass}`}
            >
              <Bell className="h-5 w-5" aria-hidden="true" />
            </button>

            <button
              type="button"
              aria-label={copy.header.menuAriaLabel}
              aria-expanded={sidebarOpen}
              aria-controls="dashboard-sidebar"
              onClick={() => {
                setSidebarOpen(true);
                setNotificationsOpen(false);
              }}
              className={`${iconButtonClass} ${blueHoverClass}`}
            >
              <Menu className="h-6 w-6" aria-hidden="true" />
            </button>

            {notificationsOpen && (
              <div
                id="dashboard-notifications"
                role="dialog"
                aria-modal="false"
                className="
                  absolute right-0 top-[calc(100%+14px)]
                  w-[min(340px,calc(100vw-40px))]
                  overflow-hidden
                  rounded-2xl
                  border border-[#19191A]/12
                  bg-[#E7E7E7]/95
                  shadow-[0_20px_60px_rgba(25,25,26,0.18)]
                  backdrop-blur-xl
                "
              >
                <div
                  className="
                    flex items-center justify-between
                    border-b border-[#19191A]/10
                    px-5 py-4
                  "
                >
                  <h2 className="font-semibold text-[#19191A]">
                    {copy.notifications.title}
                  </h2>

                  <button
                    type="button"
                    aria-label={copy.notifications.closeAriaLabel}
                    onClick={() => setNotificationsOpen(false)}
                    className={`
                      inline-flex h-9 w-9
                      items-center justify-center
                      rounded-xl
                      border border-[#19191A]/12
                      bg-[#F4F4F4]/70
                      text-[#19191A]/65
                      cursor-pointer
                      transition-all duration-300
                      focus:outline-none
                      focus-visible:ring-2
                      focus-visible:ring-[rgba(59,130,246,0.28)]
                      ${blueHoverClass}
                    `}
                  >
                    <X className="h-4 w-4" aria-hidden="true" />
                  </button>
                </div>

                <div className="px-5 py-6">
                  <p
                    className="
                      rounded-xl
                      border border-[#19191A]/10
                      bg-[#F4F4F4]/55
                      px-4 py-5
                      text-center
                      text-sm text-[#19191A]/55
                    "
                  >
                    {copy.notifications.empty}
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      <main
        className="
          relative z-10
          min-h-[calc(100vh-89px)]
        "
      />

      <button
        type="button"
        tabIndex={sidebarOpen ? 0 : -1}
        aria-hidden={!sidebarOpen}
        aria-label={copy.sidebar.closeAriaLabel}
        onClick={() => setSidebarOpen(false)}
        className={`
          fixed inset-0 z-40
          bg-[#19191A]/28
          backdrop-blur-[2px]
          transition-opacity duration-300
          ${
            sidebarOpen
              ? "pointer-events-auto opacity-100"
              : "pointer-events-none opacity-0"
          }
        `}
      />

      <aside
        id="dashboard-sidebar"
        aria-hidden={!sidebarOpen}
        className={`
          fixed bottom-0 right-0 top-0 z-50
          flex w-full max-w-sm
          flex-col
          border-l border-[#19191A]/10
          bg-[#E7E7E7]/95
          shadow-[-22px_0_70px_rgba(25,25,26,0.18)]
          backdrop-blur-xl
          transition-transform duration-300 ease-out
          ${
            sidebarOpen
              ? "translate-x-0"
              : "pointer-events-none translate-x-full"
          }
        `}
      >
        <div
          className="
            flex min-h-[88px]
            items-center justify-between
            border-b border-[#19191A]/10
            px-6
          "
        >
          <h2
            className="
              text-xl font-semibold
              tracking-[0.04em]
              text-[#19191A]
            "
          >
            {copy.sidebar.title}
          </h2>

          <button
            type="button"
            aria-label={copy.sidebar.closeAriaLabel}
            onClick={() => setSidebarOpen(false)}
            className={`${iconButtonClass} ${blueHoverClass}`}
          >
            <X className="h-5 w-5" aria-hidden="true" />
          </button>
        </div>

        <nav
          className="
            flex flex-1
            flex-col
            gap-3
            overflow-y-auto
            p-6
          "
        >
          <Link
            href="/dashboard/account"
            onClick={() => setSidebarOpen(false)}
            className={`
              ${sidebarButtonClass}
              ${blueHoverClass}
              focus-visible:ring-[rgba(59,130,246,0.28)]
            `}
          >
            <UserRound className="h-5 w-5 shrink-0" aria-hidden="true" />

            <span>{copy.sidebar.account}</span>
          </Link>

          <Link
            href="/dashboard/settings"
            onClick={() => setSidebarOpen(false)}
            className={`
              ${sidebarButtonClass}
              ${blueHoverClass}
              focus-visible:ring-[rgba(59,130,246,0.28)]
            `}
          >
            <Settings className="h-5 w-5 shrink-0" aria-hidden="true" />

            <span>{copy.sidebar.settings}</span>
          </Link>

          <div className="mt-auto pt-6">
            <button
              type="button"
              onClick={handleLogout}
              disabled={isLoggingOut}
              className={`
                ${sidebarButtonClass}
                ${redHoverClass}
                focus-visible:ring-[rgba(239,68,68,0.25)]
                disabled:cursor-not-allowed
                disabled:opacity-55
                disabled:hover:translate-y-0
                disabled:hover:bg-[#E7E7E7]/90
                disabled:hover:border-[#19191A]/15
                disabled:hover:text-[#19191A]
                disabled:hover:shadow-sm
              `}
            >
              <LogOut className="h-5 w-5 shrink-0" aria-hidden="true" />

              <span>
                {isLoggingOut
                  ? copy.sidebar.loggingOut
                  : copy.sidebar.logout}
              </span>
            </button>
          </div>
        </nav>
      </aside>
    </div>
  );
}