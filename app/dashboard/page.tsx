"use client";

import {
  useEffect,
  useState,
  type FormEvent,
} from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ArrowRight,
  Bell,
  LoaderCircle,
  LogOut,
  Menu,
  Moon,
  Save,
  Settings,
  Sun,
  UserRound,
  X,
} from "lucide-react";
import { createAuthClient } from "better-auth/react";
import { inferAdditionalFields } from "better-auth/client/plugins";
import type { auth } from "@/lib/auth";
import sk from "@/lib/json/sk.json";
import en from "@/lib/json/en.json";

const authClient = createAuthClient({
  plugins: [inferAdditionalFields<typeof auth>()],
});

type Lang = "sk" | "en";
type DashboardTheme = "light" | "dark";

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

type SignupCopy = {
  fields: {
    firstName: string;
    lastName: string;
    address: string;
    city: string;
    phone: string;
    email: string;
    company: string;
    website: string;
  };
};

type AccountFormState = {
  firstName: string;
  lastName: string;
  address: string;
  city: string;
  phone: string;
  email: string;
  company: string;
  website: string;
};

type LocalizedCopy = {
  dashboard?: DashboardCopy;
  signup?: SignupCopy;
};

const COPY_BY_LANG: Record<Lang, LocalizedCopy> = {
  sk: sk as unknown as LocalizedCopy,
  en: en as unknown as LocalizedCopy,
};

export default function DashboardPage() {
  const router = useRouter();
  const {
    data: session,
    isPending,
    refetch,
  } = authClient.useSession();

  const [lang, setLang] = useState<Lang>("sk");
  const [hydrated, setHydrated] = useState(false);

  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const [isLoggingOut, setIsLoggingOut] = useState(false);

  const [accountOpen, setAccountOpen] = useState(false);
  const [isSavingAccount, setIsSavingAccount] = useState(false);

  const [settingsOpen, setSettingsOpen] = useState(false);

  const [theme, setTheme] =
    useState<DashboardTheme>("light");

  const [accountForm, setAccountForm] =
    useState<AccountFormState>({
      firstName: "",
      lastName: "",
      address: "",
      city: "",
      phone: "",
      email: "",
      company: "",
      website: "",
    });

  const copy = COPY_BY_LANG[lang].dashboard;
  const signupCopy = COPY_BY_LANG[lang].signup;

  const isDark = theme === "dark";

  // Centralized dashboard palette.
  // Light mode keeps the original dashboard colors.
  // Dark mode mirrors the dark palette used on the main Android scene.
  const pageThemeClass = isDark
    ? "bg-[#0A0A0A] text-[#F4F4F4]"
    : "bg-[#C7C7C7] text-[#19191A]";

  const headerThemeClass = isDark
    ? `
        border-[#F4F4F4]/10
        bg-[#19191A]/65
        shadow-[0_8px_28px_rgba(0,0,0,0.28)]
      `
    : `
        border-[#19191A]/10
        bg-[#E7E7E7]/55
        shadow-[0_8px_28px_rgba(25,25,26,0.06)]
      `;

  const panelThemeClass = isDark
    ? `
        border-[#F4F4F4]/10
        bg-[#19191A]/95
      `
    : `
        border-[#19191A]/10
        bg-[#E7E7E7]/95
      `;

  const accountPanelThemeClass = isDark
    ? `
        border-[#F4F4F4]/10
        bg-[#19191A]/92
      `
    : `
        border-[#19191A]/10
        bg-[#E7E7E7]/92
      `;

  const iconTileThemeClass = isDark
    ? `
        border-[#F4F4F4]/12
        bg-[#252526]/65
        text-[#F4F4F4]/75
      `
    : `
        border-[#19191A]/12
        bg-[#F4F4F4]/65
        text-[#19191A]/75
      `;

  const softButtonThemeClass = isDark
    ? `
        border-[#F4F4F4]/12
        bg-[#252526]/70
        text-[#F4F4F4]/65
      `
    : `
        border-[#19191A]/12
        bg-[#F4F4F4]/70
        text-[#19191A]/65
      `;

  const softMutedThemeClass = isDark
    ? `
        border-[#F4F4F4]/10
        bg-[#252526]/55
        text-[#F4F4F4]/55
      `
    : `
        border-[#19191A]/10
        bg-[#F4F4F4]/55
        text-[#19191A]/55
      `;

  const dividerThemeClass = isDark
    ? "border-[#F4F4F4]/10"
    : "border-[#19191A]/10";

  const textThemeClass = isDark
    ? "text-[#F4F4F4]"
    : "text-[#19191A]";

  const mutedTextThemeClass = isDark
    ? "text-[#F4F4F4]/60"
    : "text-[#19191A]/60";

  const labelTextThemeClass = isDark
    ? "text-[#F4F4F4]/65"
    : "text-[#19191A]/65";

  const logoThemeClass = isDark
    ? "brightness-0 invert"
    : "brightness-0";

  const backdropThemeClass = isDark
    ? "bg-black/40"
    : "bg-[#19191A]/28";

  const themeToggleTrackClass = isDark
    ? "border-[#F4F4F4]/15 bg-[#252526]/90"
    : "border-[#19191A]/15 bg-[#F4F4F4]/70";

  const themeToggleThumbClass = isDark
    ? "translate-x-14 bg-[#19191A] text-[#F4F4F4]"
    : "translate-x-0 bg-[#E7E7E7] text-[#19191A]";

  useEffect(() => {
    const savedLanguage =
      window.localStorage.getItem("spectline-lang");

    if (
      savedLanguage === "sk" ||
      savedLanguage === "en"
    ) {
      setLang(savedLanguage);
    }

    const savedTheme = window.localStorage.getItem(
      "spectline-dashboard-theme"
    );

    if (
      savedTheme === "light" ||
      savedTheme === "dark"
    ) {
      setTheme(savedTheme);
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
    if (!session?.user) return;

    setAccountForm({
      firstName: session.user.firstName ?? "",
      lastName: session.user.lastName ?? "",
      address: session.user.address ?? "",
      city: session.user.city ?? "",
      phone: session.user.phone ?? "",
      email: session.user.email ?? "",
      company: session.user.company ?? "",
      website: session.user.website ?? "",
    });
  }, [
    session?.user.firstName,
    session?.user.lastName,
    session?.user.address,
    session?.user.city,
    session?.user.phone,
    session?.user.email,
    session?.user.company,
    session?.user.website,
  ]);

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key !== "Escape") return;

      setSidebarOpen(false);
      setNotificationsOpen(false);
      setAccountOpen(false);
      setSettingsOpen(false);
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

  function handleAccountToggle() {
    setNotificationsOpen(false);
    setSettingsOpen(false);

    const isMobile = window.matchMedia(
      "(max-width: 900px)"
    ).matches;

    if (isMobile) {
      setAccountOpen(true);
      return;
    }

    setAccountOpen((current) => !current);
  }

  function handleMobileAccountBack() {
    setAccountOpen(false);
    setSidebarOpen(true);
  }

  function handleMobileAccountClose() {
    setAccountOpen(false);
    setSidebarOpen(false);
  }

  function handleSettingsToggle() {
    setNotificationsOpen(false);
    setAccountOpen(false);

    const isMobile = window.matchMedia(
      "(max-width: 900px)"
    ).matches;

    if (isMobile) {
      setSettingsOpen(true);
      return;
    }

    setSettingsOpen((current) => !current);
  }

  function handleMobileSettingsBack() {
    setSettingsOpen(false);
    setSidebarOpen(true);
  }

  function handleMobileSettingsClose() {
    setSettingsOpen(false);
    setSidebarOpen(false);
  }

  function handleThemeToggle() {
    setTheme((current) => {
      const next =
        current === "light"
          ? "dark"
          : "light";

      window.localStorage.setItem(
        "spectline-dashboard-theme",
        next
      );

      return next;
    });
  }

  async function handleAccountSave(
    event: FormEvent<HTMLFormElement>
  ) {
    event.preventDefault();

    if (isSavingAccount || !session) return;

    setIsSavingAccount(true);

    try {
      const firstName = accountForm.firstName.trim();
      const lastName = accountForm.lastName.trim();

      const fullName =
        `${firstName} ${lastName}`.trim();

      const { error: updateError } =
        await authClient.updateUser({
          name: fullName || session.user.name,

          firstName,
          lastName,
          address: accountForm.address.trim(),
          city: accountForm.city.trim(),
          phone: accountForm.phone.trim(),
          company: accountForm.company.trim(),
          website: accountForm.website.trim(),
        });

      if (updateError) {
        console.error(
          "Account update failed.",
          updateError
        );

        return;
      }

      const nextEmail = accountForm.email.trim();

      if (
        nextEmail &&
        nextEmail !== session.user.email
      ) {
        const { error: emailError } =
          await authClient.changeEmail({
            newEmail: nextEmail,
            callbackURL: "/dashboard",
          });

        if (emailError) {
          console.error(
            "Email update failed.",
            emailError
          );

          return;
        }
      }

      await refetch();
    } finally {
      setIsSavingAccount(false);
    }
  }

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

  if (!hydrated || !copy || !signupCopy) {
    return <div className="min-h-screen bg-[#C7C7C7]" />;
  }

  if (isPending) {
    return (
      <main
        className={`
          min-h-screen
          grid place-items-center
          px-6
          transition-colors duration-300
          ${pageThemeClass}
        `}
      >
        <div className="flex flex-col items-center gap-5">
          <div
            className={`
              h-14 w-14
              rounded-full
              border-4
              animate-spin
              ${
                isDark
                  ? "border-[#F4F4F4]/10 border-t-[#F4F4F4]"
                  : "border-[#19191A]/10 border-t-[#19191A]"
              }
            `}
            aria-hidden="true"
          />

          <p className={`text-sm font-medium ${mutedTextThemeClass}`}>
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
    shadow-sm
    cursor-pointer
    transition-all duration-300
    focus:outline-none
    focus-visible:ring-2
    focus-visible:ring-[rgba(59,130,246,0.28)]
    focus-visible:ring-offset-2
      ${
        isDark
          ? `
            bg-[#252526]/90
            border border-[#F4F4F4]/15
            text-[#F4F4F4]/75
            focus-visible:ring-offset-[#0A0A0A]
          `
          : `
            bg-[#E7E7E7]/90
            border border-[#19191A]/15
            text-[#19191A]/75
            focus-visible:ring-offset-[#C7C7C7]
          `
      }
  `;

  const sidebarButtonClass = `
    w-full
    flex items-center gap-3
    rounded-2xl
    px-5 py-4
    text-left
    text-base font-semibold
    shadow-sm
    cursor-pointer
    transition-all duration-300
    focus:outline-none
    focus-visible:ring-2
    focus-visible:ring-offset-2
      ${
        isDark
          ? `
            bg-[#252526]/90
            border border-[#F4F4F4]/15
            text-[#F4F4F4]
            focus-visible:ring-offset-[#19191A]
          `
          : `
            bg-[#E7E7E7]/90
            border border-[#19191A]/15
            text-[#19191A]
            focus-visible:ring-offset-[#E7E7E7]
          `
      }
  `;

  const accountInputClass = `
    w-full
    rounded-xl
    px-4 py-3
    shadow-sm
    outline-none
    transition-all duration-300

    ${
      isDark
        ? `
          border border-[#F4F4F4]/15
          bg-[#252526]/80
          text-[#F4F4F4]
          focus:border-[#F4F4F4]/30
          focus:ring-2
          focus:ring-[#F4F4F4]/10
        `
        : `
          border border-[#19191A]/15
          bg-[#F4F4F4]/60
          text-[#19191A]
          focus:border-[#19191A]/30
          focus:ring-2
          focus:ring-[#19191A]/10
        `
    }
  `;

  return (
    <div
      className={`
        relative
        min-h-screen
        overflow-hidden
        transition-colors duration-300
        ${pageThemeClass}
      `}
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
          className={`
            w-[min(900px,88vw)]
            max-h-[78vh]
            object-contain
            select-none
            ${logoThemeClass}

            ${
              isDark
                ? "opacity-[0.05]"
                : "opacity-[0.07]"
            }
          `}
        />
      </div>

      <header
        className={`
          relative z-20
          border-b
          backdrop-blur-xl
          transition-colors duration-300
          ${headerThemeClass}
        `}
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
              className={`
                h-12 w-12
                shrink-0
                object-contain
                opacity-80
                ${logoThemeClass}
              `}
            />

            <div className="min-w-0">
              <p
                className={`
                  truncate
                  text-base font-semibold
                  tracking-[0.04em]
                  sm:text-lg
                  ${textThemeClass}
                `}
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
                setAccountOpen(false);
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
                className={`
                  absolute right-0 top-[calc(100%+14px)]
                  w-[min(340px,calc(100vw-40px))]
                  overflow-hidden
                  rounded-2xl
                  border
                  shadow-[0_20px_60px_rgba(25,25,26,0.18)]
                  backdrop-blur-xl
                  transition-colors duration-300
                  ${panelThemeClass}
                `}
              >
                <div
                  className={`
                    flex items-center justify-between
                    border-b
                    px-5 py-4
                    ${dividerThemeClass}
                  `}
                >
                  <h2 className={`font-semibold ${textThemeClass}`}>
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
                      border
                      cursor-pointer
                      transition-all duration-300
                      focus:outline-none
                      focus-visible:ring-2
                      focus-visible:ring-[rgba(59,130,246,0.28)]
                      ${softButtonThemeClass}
                      ${blueHoverClass}
                    `}
                  >
                    <X className="h-4 w-4" aria-hidden="true" />
                  </button>
                </div>

                <div className="px-5 py-6">
                  <p
                    className={`
                      rounded-xl
                      border
                      px-4 py-5
                      text-center
                      text-sm
                      ${softMutedThemeClass}
                    `}
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

      {accountOpen && (
      <section
        id="dashboard-account-panel"
        className={`
          fixed
          left-[calc((100vw-24rem)/2)]
          top-1/2
          -translate-x-1/2
          -translate-y-1/2
          z-[45]
          h-[calc(100vh-160px)]
          w-[min(720px,calc(100vw-440px))]
          overflow-hidden
          rounded-3xl
          border
          shadow-[0_22px_70px_rgba(25,25,26,0.16)]
          backdrop-blur-xl
          transition-colors duration-300
          ${accountPanelThemeClass}

          max-[900px]:inset-0
          max-[900px]:z-[60]
          max-[900px]:h-full
          max-[900px]:w-full
          max-[900px]:translate-x-0
          max-[900px]:translate-y-0
          max-[900px]:rounded-none
          max-[900px]:border-0
        `}
      >
        <div className="flex h-full flex-col">
          <div
            className={`
              flex
              min-h-[88px]
              shrink-0
              items-center
              justify-between
              gap-4
              border-b
              px-6
              ${dividerThemeClass}
            `}
          >
            <div className="flex min-w-0 items-center gap-3">
              <div
                className={`
                  grid h-11 w-11
                  shrink-0
                  place-items-center
                  rounded-xl
                  border
                  ${iconTileThemeClass}
                `}
              >
                <UserRound
                  className="h-5 w-5"
                  aria-hidden="true"
                />
              </div>

              <h2
                className={`
                  truncate
                  text-xl font-semibold
                  tracking-[0.04em]
                  ${textThemeClass}
                `}
              >
                {copy.sidebar.account}
              </h2>
            </div>

            <div
              className="
                hidden
                shrink-0
                items-center
                gap-2
                max-[900px]:flex
              "
            >
              <button
                type="button"
                aria-label={copy.sidebar.title}
                onClick={handleMobileAccountBack}
                className={`${iconButtonClass} ${blueHoverClass}`}
              >
                <ArrowRight
                  className="h-5 w-5"
                  aria-hidden="true"
                />
              </button>

              <button
                type="button"
                aria-label={copy.sidebar.closeAriaLabel}
                onClick={handleMobileAccountClose}
                className={`${iconButtonClass} ${blueHoverClass}`}
              >
                <X
                  className="h-5 w-5"
                  aria-hidden="true"
                />
              </button>
            </div>
          </div>

          <form
            onSubmit={handleAccountSave}
            className="
              flex-1
              overflow-y-auto
              p-6
              sm:p-8
            "
          >
            <div
              className="
                grid
                grid-cols-1
                gap-5
                sm:grid-cols-2
              "
            >
              <label className="block">
                <span
                  className={`
                    mb-2 block
                    text-sm font-medium
                    ${labelTextThemeClass}
                  `}
                >
                  {signupCopy.fields.firstName}
                </span>

                <input
                  type="text"
                  value={accountForm.firstName}
                  autoComplete="given-name"
                  onChange={(event) =>
                    setAccountForm((current) => ({
                      ...current,
                      firstName: event.target.value,
                    }))
                  }
                  className={accountInputClass}
                />
              </label>

              <label className="block">
                <span
                  className={`
                    mb-2 block
                    text-sm font-medium
                    ${labelTextThemeClass}
                  `}
                >
                  {signupCopy.fields.lastName}
                </span>

                <input
                  type="text"
                  value={accountForm.lastName}
                  autoComplete="family-name"
                  onChange={(event) =>
                    setAccountForm((current) => ({
                      ...current,
                      lastName: event.target.value,
                    }))
                  }
                  className={accountInputClass}
                />
              </label>

              <label className="block">
                <span
                  className={`
                    mb-2 block
                    text-sm font-medium
                    ${labelTextThemeClass}
                  `}
                >
                  {signupCopy.fields.address}
                </span>

                <input
                  type="text"
                  value={accountForm.address}
                  autoComplete="street-address"
                  onChange={(event) =>
                    setAccountForm((current) => ({
                      ...current,
                      address: event.target.value,
                    }))
                  }
                  className={accountInputClass}
                />
              </label>

              <label className="block">
                <span
                  className={`
                    mb-2 block
                    text-sm font-medium
                    ${labelTextThemeClass}
                  `}
                >
                  {signupCopy.fields.city}
                </span>

                <input
                  type="text"
                  value={accountForm.city}
                  autoComplete="address-level2"
                  onChange={(event) =>
                    setAccountForm((current) => ({
                      ...current,
                      city: event.target.value,
                    }))
                  }
                  className={accountInputClass}
                />
              </label>

              <label className="block">
                <span
                  className={`
                    mb-2 block
                    text-sm font-medium
                    ${labelTextThemeClass}
                  `}
                >
                  {signupCopy.fields.phone}
                </span>

                <input
                  type="tel"
                  value={accountForm.phone}
                  autoComplete="tel"
                  onChange={(event) =>
                    setAccountForm((current) => ({
                      ...current,
                      phone: event.target.value,
                    }))
                  }
                  className={accountInputClass}
                />
              </label>

              <label className="block">
                <span
                  className={`
                    mb-2 block
                    text-sm font-medium
                    ${labelTextThemeClass}
                  `}
                >
                  {signupCopy.fields.email}
                </span>

                <input
                  type="email"
                  value={accountForm.email}
                  autoComplete="email"
                  onChange={(event) =>
                    setAccountForm((current) => ({
                      ...current,
                      email: event.target.value,
                    }))
                  }
                  className={accountInputClass}
                />
              </label>

              <label className="block">
                <span
                  className={`
                    mb-2 block
                    text-sm font-medium
                    ${labelTextThemeClass}
                  `}
                >
                  {signupCopy.fields.company}
                </span>

                <input
                  type="text"
                  value={accountForm.company}
                  autoComplete="organization"
                  onChange={(event) =>
                    setAccountForm((current) => ({
                      ...current,
                      company: event.target.value,
                    }))
                  }
                  className={accountInputClass}
                />
              </label>

              <label className="block">
                <span
                  className={`
                    mb-2 block
                    text-sm font-medium
                    ${labelTextThemeClass}
                  `}
                >
                  {signupCopy.fields.website}
                </span>

                <input
                  type="text"
                  value={accountForm.website}
                  autoComplete="url"
                  onChange={(event) =>
                    setAccountForm((current) => ({
                      ...current,
                      website: event.target.value,
                    }))
                  }
                  className={accountInputClass}
                />
              </label>
            </div>

            <div className="mt-7 flex justify-end">
              <button
                type="submit"
                disabled={isSavingAccount}
                aria-label={copy.sidebar.account}
                className={`
                  ${iconButtonClass}
                  ${blueHoverClass}
                  disabled:cursor-not-allowed
                  disabled:opacity-50
                  disabled:hover:translate-y-0
                `}
              >
                {isSavingAccount ? (
                  <LoaderCircle
                    className="h-5 w-5 animate-spin"
                    aria-hidden="true"
                  />
                ) : (
                  <Save
                    className="h-5 w-5"
                    aria-hidden="true"
                  />
                )}
              </button>
            </div>
          </form>
        </div>
      </section>
    )}

    {settingsOpen && (
      <section
        id="dashboard-settings-panel"
        className={`
          fixed
          left-[calc((100vw-24rem)/2)]
          top-1/2
          -translate-x-1/2
          -translate-y-1/2
          z-[45]
          h-[calc(100vh-160px)]
          w-[min(720px,calc(100vw-440px))]
          overflow-hidden
          rounded-3xl
          border
          ${panelThemeClass}
          shadow-[0_22px_70px_rgba(25,25,26,0.16)]
          backdrop-blur-xl
          transition-colors duration-300

          max-[900px]:inset-0
          max-[900px]:z-[60]
          max-[900px]:h-full
          max-[900px]:w-full
          max-[900px]:translate-x-0
          max-[900px]:translate-y-0
          max-[900px]:rounded-none
          max-[900px]:border-0
        `}
      >
        <div className="flex h-full flex-col">
          <div
            className={`
              flex
              min-h-[88px]
              shrink-0
              items-center
              justify-between
              gap-4
              border-b
              px-6
              ${dividerThemeClass}
            `}
          >
            <div className="flex min-w-0 items-center gap-3">
              <div
                className={`
                  grid h-11 w-11
                  shrink-0
                  place-items-center
                  rounded-xl
                  border
                  ${iconTileThemeClass}
                `}
              >
                <Settings
                  className="h-5 w-5"
                  aria-hidden="true"
                />
              </div>

              <h2
                className={`
                  truncate
                  text-xl font-semibold
                  tracking-[0.04em]
                  ${textThemeClass}
                `}
              >
                {copy.sidebar.settings}
              </h2>
            </div>

            <div
              className="
                hidden
                shrink-0
                items-center
                gap-2
                max-[900px]:flex
              "
            >
              <button
                type="button"
                aria-label={copy.sidebar.title}
                onClick={handleMobileSettingsBack}
                className={`${iconButtonClass} ${blueHoverClass}`}
              >
                <ArrowRight
                  className="h-5 w-5"
                  aria-hidden="true"
                />
              </button>

              <button
                type="button"
                aria-label={copy.sidebar.closeAriaLabel}
                onClick={handleMobileSettingsClose}
                className={`${iconButtonClass} ${blueHoverClass}`}
              >
                <X
                  className="h-5 w-5"
                  aria-hidden="true"
                />
              </button>
            </div>
          </div>

          <div
            className="
              flex flex-1
              items-center
              justify-center
              p-6
            "
          >
            <button
              type="button"
              onClick={handleThemeToggle}
              aria-label={copy.sidebar.settings}
              aria-pressed={isDark}
              className={`
                relative
                inline-flex
                h-14 w-28
                items-center
                rounded-full
                border
                p-1
                shadow-sm
                cursor-pointer
                transition-all duration-300

                ${themeToggleTrackClass}
                ${blueHoverClass}

                focus:outline-none
                focus-visible:ring-2
                focus-visible:ring-[rgba(59,130,246,0.28)]
              `}
            >
              <span
                className={`
                  grid
                  h-11 w-11
                  place-items-center
                  rounded-full
                  shadow-sm
                  transition-all duration-300

                  ${themeToggleThumbClass}
                `}
              >
                {isDark ? (
                  <Moon
                    className="h-5 w-5"
                    aria-hidden="true"
                  />
                ) : (
                  <Sun
                    className="h-5 w-5"
                    aria-hidden="true"
                  />
                )}
              </span>
            </button>
          </div>
        </div>
      </section>
    )}

      <button
        type="button"
        tabIndex={sidebarOpen ? 0 : -1}
        aria-hidden={!sidebarOpen}
        aria-label={copy.sidebar.closeAriaLabel}
        onClick={() => {
          setSidebarOpen(false);
          setAccountOpen(false);
          setSettingsOpen(false);
        }}
        className={`
          fixed inset-0 z-40
          ${backdropThemeClass}
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
          border-l
          shadow-[-22px_0_70px_rgba(25,25,26,0.18)]
          backdrop-blur-xl
          transition-colors duration-300
          transition-transform duration-300 ease-out
          ${panelThemeClass}
          ${
            sidebarOpen
              ? "translate-x-0"
              : "pointer-events-none translate-x-full"
          }
        `}
      >
        <div
          className={`
            flex min-h-[88px]
            items-center justify-between
            border-b
            px-6
            ${dividerThemeClass}
          `}
        >
          <h2
            className={`
              text-xl font-semibold
              tracking-[0.04em]
              ${textThemeClass}
            `}
          >
            {copy.sidebar.title}
          </h2>

          <button
            type="button"
            aria-label={copy.sidebar.closeAriaLabel}
            onClick={() => {
              setSidebarOpen(false);
              setAccountOpen(false);
              setSettingsOpen(false);
            }}
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
          <button
            type="button"
            aria-pressed={accountOpen}
            onClick={handleAccountToggle}
            className={`
              ${sidebarButtonClass}
              ${blueHoverClass}
              focus-visible:ring-[rgba(59,130,246,0.28)]
              ${
                accountOpen
                  ? `
                    bg-[rgba(59,130,246,0.18)]
                    border-[rgba(147,197,253,0.32)]
                  `
                  : ""
              }
            `}
          >
            <UserRound
              className="h-5 w-5 shrink-0"
              aria-hidden="true"
            />

            <span>{copy.sidebar.account}</span>
          </button>

          <button
            type="button"
            aria-pressed={settingsOpen}
            onClick={handleSettingsToggle}
            className={`
              ${sidebarButtonClass}
              ${blueHoverClass}
              focus-visible:ring-[rgba(59,130,246,0.28)]

              ${
                settingsOpen
                  ? `
                    bg-[rgba(59,130,246,0.18)]
                    border-[rgba(147,197,253,0.32)]
                  `
                  : ""
              }
            `}
          >
            <Settings
              className="h-5 w-5 shrink-0"
              aria-hidden="true"
            />

            <span>{copy.sidebar.settings}</span>
          </button>

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
                ${
                  isDark
                    ? `
                      disabled:hover:bg-[#252526]/90
                      disabled:hover:border-[#F4F4F4]/15
                      disabled:hover:text-[#F4F4F4]
                    `
                    : `
                      disabled:hover:bg-[#E7E7E7]/90
                      disabled:hover:border-[#19191A]/15
                      disabled:hover:text-[#19191A]
                    `
                }
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