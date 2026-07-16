"use client";

import { useEffect, useState } from "react";
import type { FormEvent, ReactNode } from "react";
import Link from "next/link";
import { createAuthClient } from "better-auth/react";
import sk from "@/lib/json/sk.json";
import en from "@/lib/json/en.json";

const authClient = createAuthClient();

type Lang = "sk" | "en";

type SignupCopy = {
  homeAriaLabel: string;
  logoAlt: string;
  title: string;
  sections: {
    client: string;
    company: string;
    password: string;
  };
  fields: {
    firstName: string;
    lastName: string;
    address: string;
    city: string;
    phone: string;
    email: string;
    company: string;
    website: string;
    password: string;
    confirmPassword: string;
  };
  validation: {
    nameRequired: string;
    emailRequired: string;
    addressRequired: string;
    passwordLength: string;
    passwordMismatch: string;
    signupFailed: string;
  };
  submit: string;
  existingAccount: string;
  login: string;
};

type LocalizedCopy = {
  signup?: SignupCopy;
};

const COPY_BY_LANG: Record<Lang, LocalizedCopy> = {
  sk: sk as LocalizedCopy,
  en: en as LocalizedCopy,
};

function Section({
  title,
  open,
  onToggle,
  children,
}: {
  title: string;
  open: boolean;
  onToggle: () => void;
  children: ReactNode;
}) {
  return (
    <div className="mt-4">
      <button
        type="button"
        onClick={onToggle}
        className="
          w-full flex items-center justify-between
          px-1 py-2
          text-[#19191A]/75 hover:text-[#19191A]
          transition-colors duration-200
          focus:outline-none focus-visible:ring-2
          focus-visible:ring-[#19191A]/20
          rounded-lg
        "
        aria-expanded={open}
      >
        <span className="font-semibold">{title}</span>

        <span
          className={`
            text-xl leading-none text-[#19191A]/50
            transition-transform duration-200
            ${open ? "rotate-180" : ""}
          `}
          aria-hidden="true"
        >
          ▾
        </span>
      </button>

      {open && <div className="mt-4">{children}</div>}
    </div>
  );
}

export default function SignupPage() {
  const [lang, setLang] = useState<Lang>("sk");
  const [hydrated, setHydrated] = useState(false);

  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [company, setCompany] = useState("");
  const [phone, setPhone] = useState("");
  const [addressLine, setAddressLine] = useState("");
  const [city, setCity] = useState("");
  const [email, setEmail] = useState("");
  const [website, setWebsite] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState("");

  const [sections, setSections] = useState({
    client: true,
    company: false,
    password: false,
  });

  useEffect(() => {
    const frameId = window.requestAnimationFrame(() => {
      const savedLanguage = window.localStorage.getItem("spectline-lang");

      if (savedLanguage === "sk" || savedLanguage === "en") {
        setLang(savedLanguage);
      }

      setHydrated(true);
    });

    return () => window.cancelAnimationFrame(frameId);
  }, []);

  const copy = COPY_BY_LANG[lang].signup;

  const inputClass = `
    w-full px-4 py-3 rounded-xl
    bg-white/55
    border border-[#19191A]/15
    text-[#19191A]
    placeholder:text-[#19191A]/40
    shadow-sm
    transition-all duration-300
    focus:outline-none
    focus:ring-2 focus:ring-[#19191A]/15
    focus:border-[#19191A]/30
  `;

  async function handleSignup(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!copy) return;

    setError("");

    if (!firstName.trim() || !lastName.trim()) {
      setError(copy.validation.nameRequired);
      return;
    }

    if (!email.trim()) {
      setError(copy.validation.emailRequired);
      return;
    }

    if (!addressLine.trim() || !city.trim()) {
      setError(copy.validation.addressRequired);
      return;
    }

    if (password.length < 8) {
      setError(copy.validation.passwordLength);
      return;
    }

    if (password !== confirmPassword) {
      setError(copy.validation.passwordMismatch);
      return;
    }

    const { error: signupError } = await authClient.signUp.email({
      name: `${firstName.trim()} ${lastName.trim()}`,
      email: email.trim(),
      password,
      callbackURL: "/dashboard",
    });

    if (signupError) {
      setError(copy.validation.signupFailed);
      return;
    }

    window.location.href = "/dashboard";
  }

  if (!hydrated || !copy) {
    return <div className="min-h-screen bg-[#C7C7C7]" />;
  }

  return (
    <main className="min-h-screen bg-[#C7C7C7] flex items-start justify-center px-6 py-[5vh]">
      <div
        className="
          w-full max-w-md max-h-[90vh]
          overflow-y-auto no-scrollbar
          bg-white/45
          border border-[#19191A]/10
          rounded-2xl px-8 py-6
          shadow-[0_18px_60px_rgba(25,25,26,0.10)]
          backdrop-blur-md
        "
      >
        <div className="mb-4 flex justify-start">
          <Link
            href="/"
            aria-label={copy.homeAriaLabel}
            className="
              inline-flex items-center justify-center
              p-1
              text-[#19191A]/65
              hover:text-[#19191A]
              active:text-[#19191A]
              focus:outline-none
              focus-visible:ring-2
              focus-visible:ring-[#19191A]/20
              rounded-lg
              transition-colors duration-200
              cursor-pointer
            "
          >
            <svg
              viewBox="0 0 24 24"
              className="w-9 h-9"
              aria-hidden="true"
            >
              <path
                d="M3 12L12 4l9 8"
                stroke="currentColor"
                strokeWidth="2"
                fill="none"
                strokeLinecap="round"
                strokeLinejoin="round"
              />

              <path
                d="M5 11v9a1 1 0 0 0 1 1h4v-6h4v6h4a1 1 0 0 0 1-1v-9"
                stroke="currentColor"
                strokeWidth="2"
                fill="none"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </Link>
        </div>

        <div className="flex justify-center mb-2">
          <img
            src="/images/logo_pdf.png"
            alt={copy.logoAlt}
            className="h-[200px] w-[200px] object-contain opacity-90"
          />
        </div>

        <h1
          className="
            text-[#19191A]
            text-2xl md:text-3xl
            tracking-[0.12em]
            uppercase
            opacity-95
            text-center
            mb-6
          "
        >
          {copy.title}
        </h1>

        <form onSubmit={handleSignup}>
          <Section
            title={copy.sections.client}
            open={sections.client}
            onToggle={() =>
              setSections((current) => ({
                ...current,
                client: !current.client,
              }))
            }
          >
            <input
              type="text"
              value={firstName}
              placeholder={copy.fields.firstName}
              autoComplete="given-name"
              onChange={(event) => setFirstName(event.target.value)}
              className={inputClass}
            />

            <input
              type="text"
              value={lastName}
              placeholder={copy.fields.lastName}
              autoComplete="family-name"
              onChange={(event) => setLastName(event.target.value)}
              className={`${inputClass} mt-4`}
            />

            <div className="mt-4 flex flex-col sm:flex-row items-center gap-3">
              <input
                type="text"
                value={addressLine}
                placeholder={copy.fields.address}
                autoComplete="street-address"
                onChange={(event) => setAddressLine(event.target.value)}
                className={inputClass}
              />

              <span
                className="
                  hidden sm:inline
                  text-[#19191A]/40
                  font-semibold
                  select-none
                "
                aria-hidden="true"
              >
                /
              </span>

              <input
                type="text"
                value={city}
                placeholder={copy.fields.city}
                autoComplete="address-level2"
                onChange={(event) => setCity(event.target.value)}
                className={inputClass}
              />
            </div>

            <input
              type="tel"
              value={phone}
              placeholder={copy.fields.phone}
              autoComplete="tel"
              onChange={(event) => setPhone(event.target.value)}
              className={`${inputClass} mt-4`}
            />

            <input
              type="email"
              value={email}
              placeholder={copy.fields.email}
              autoComplete="email"
              onChange={(event) => setEmail(event.target.value)}
              className={`${inputClass} mt-4`}
            />
          </Section>

          <Section
            title={copy.sections.company}
            open={sections.company}
            onToggle={() =>
              setSections((current) => ({
                ...current,
                company: !current.company,
              }))
            }
          >
            <input
              type="text"
              value={company}
              placeholder={copy.fields.company}
              autoComplete="organization"
              onChange={(event) => setCompany(event.target.value)}
              className={inputClass}
            />

            <input
              type="url"
              value={website}
              placeholder={copy.fields.website}
              autoComplete="url"
              onChange={(event) => setWebsite(event.target.value)}
              className={`${inputClass} mt-4`}
            />
          </Section>

          <Section
            title={copy.sections.password}
            open={sections.password}
            onToggle={() =>
              setSections((current) => ({
                ...current,
                password: !current.password,
              }))
            }
          >
            <input
              type="password"
              value={password}
              placeholder={copy.fields.password}
              autoComplete="new-password"
              onChange={(event) => setPassword(event.target.value)}
              className={inputClass}
            />

            <input
              type="password"
              value={confirmPassword}
              placeholder={copy.fields.confirmPassword}
              autoComplete="new-password"
              onChange={(event) => setConfirmPassword(event.target.value)}
              className={`${inputClass} mt-4`}
            />
          </Section>

          {error && (
            <p
              role="alert"
              className="
                mt-4 rounded-xl
                border border-red-700/15
                bg-red-500/10
                px-4 py-3
                text-sm text-red-800
              "
            >
              {error}
            </p>
          )}

          <button
            type="submit"
            className="
              w-full mt-6 px-8 py-3 rounded-2xl
              bg-[#E7E7E7]/90
              border border-[#19191A]/15
              text-lg font-semibold text-[#19191A]
              cursor-pointer
              transition-all duration-300
              hover:-translate-y-0.5
              hover:bg-[rgba(239,68,68,0.32)]
              hover:border-[rgba(252,165,165,0.38)]
              hover:text-[rgba(244,244,244,0.96)]
              hover:shadow-[inset_0_1px_0_rgba(255,255,255,0.16),0_8px_26px_rgba(239,68,68,0.12)]
              focus:outline-none
              focus-visible:ring-2
              focus-visible:ring-[rgba(239,68,68,0.25)]
              focus-visible:ring-offset-2
              focus-visible:ring-offset-[#F4F4F4]
            "
          >
            {copy.submit}
          </button>
        </form>

        <p className="text-center text-[#19191A]/60 mt-6">
          {copy.existingAccount}{" "}
          <Link
            href="/login"
            className="
              inline-block
              text-[#19191A]/85
              font-medium
              transform
              transition-all duration-300
              hover:text-[#19191A]
              hover:-translate-y-0.5
              focus:outline-none
              focus-visible:ring-2
              focus-visible:ring-[#19191A]/20
              rounded
            "
          >
            {copy.login}
          </Link>
        </p>
      </div>
    </main>
  );
}