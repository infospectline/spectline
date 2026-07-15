"use client";

import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import Link from "next/link";
import { createAuthClient } from "better-auth/react";
import sk from "@/lib/json/sk.json";
import en from "@/lib/json/en.json";

const authClient = createAuthClient();

type Lang = "sk" | "en";

type LoginCopy = {
  homeAriaLabel: string;
  logoAlt: string;
  title: string;

  fields: {
    email: string;
    password: string;
  };

  validation: {
    fieldsRequired: string;
    loginFailed: string;
  };

  submit: string;
  noAccount: string;
  signup: string;
};

type LocalizedCopy = {
  login?: LoginCopy;
};

const COPY_BY_LANG: Record<Lang, LocalizedCopy> = {
  sk: sk as unknown as LocalizedCopy,
  en: en as unknown as LocalizedCopy,
};

export default function LoginPage() {
  const [lang, setLang] = useState<Lang>("sk");
  const [hydrated, setHydrated] = useState(false);

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

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

  const copy = COPY_BY_LANG[lang].login;

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

  async function handleLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!copy || isSubmitting) return;

    setError("");

    if (!email.trim() || !password) {
      setError(copy.validation.fieldsRequired);
      return;
    }

    setIsSubmitting(true);

    try {
      const { error: loginError } = await authClient.signIn.email({
        email: email.trim(),
        password,
        rememberMe: true,
      });

      if (loginError) {
        setError(copy.validation.loginFailed);
        return;
      }

      window.location.href = "/dashboard";
    } catch {
      setError(copy.validation.loginFailed);
    } finally {
      setIsSubmitting(false);
    }
  }

  if (!hydrated || !copy) {
    return <div className="min-h-screen bg-[#C7C7C7]" />;
  }

  return (
    <main
      className="
        min-h-screen
        bg-[#C7C7C7]
        flex items-center justify-center
        px-6 py-8
      "
    >
      <div
        className="
          w-full max-w-md
          bg-white/45
          border border-[#19191A]/10
          rounded-2xl
          px-8 py-6
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
              rounded-lg
              transition-colors duration-200
              cursor-pointer
              focus:outline-none
              focus-visible:ring-2
              focus-visible:ring-[#19191A]/20
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
            className="
              h-[200px] w-[200px]
              object-contain
              brightness-0
              opacity-80
            "
          />
        </div>

        <h1
          className="
            mb-6
            text-center
            text-2xl md:text-3xl
            tracking-[0.12em]
            uppercase
            text-[#19191A]
            opacity-95
          "
        >
          {copy.title}
        </h1>

        <form onSubmit={handleLogin}>
          <input
            type="email"
            value={email}
            placeholder={copy.fields.email}
            autoComplete="email"
            onChange={(event) => setEmail(event.target.value)}
            className={inputClass}
          />

          <input
            type="password"
            value={password}
            placeholder={copy.fields.password}
            autoComplete="current-password"
            onChange={(event) => setPassword(event.target.value)}
            className={`${inputClass} mt-4`}
          />

          {error && (
            <p
              role="alert"
              aria-live="polite"
              className="
                mt-4
                rounded-xl
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
            disabled={isSubmitting}
            aria-busy={isSubmitting}
            className="
              w-full mt-6 px-8 py-3
              rounded-2xl
              bg-[#E7E7E7]/90
              border border-[#19191A]/15
              text-lg font-semibold text-[#19191A]
              cursor-pointer
              shadow-sm
              transition-all duration-300

              hover:-translate-y-0.5
              hover:bg-[rgba(34,197,94,0.34)]
              hover:border-[rgba(134,239,172,0.38)]
              hover:text-[rgba(244,244,244,0.96)]
              hover:shadow-[inset_0_1px_0_rgba(255,255,255,0.18),0_8px_26px_rgba(34,197,94,0.12)]

              focus:outline-none
              focus-visible:ring-2
              focus-visible:ring-[rgba(34,197,94,0.25)]
              focus-visible:ring-offset-2
              focus-visible:ring-offset-[#F4F4F4]

              disabled:cursor-not-allowed
              disabled:opacity-60
              disabled:hover:translate-y-0
              disabled:hover:bg-[#E7E7E7]/90
              disabled:hover:border-[#19191A]/15
              disabled:hover:text-[#19191A]
              disabled:hover:shadow-sm
            "
          >
            {copy.submit}
          </button>
        </form>

        <p className="mt-6 text-center text-[#19191A]/60">
          {copy.noAccount}{" "}
          <Link
            href="/signup"
            className="
              inline-block
              rounded
              font-medium
              text-[#19191A]/85
              transform
              transition-all duration-300
              hover:-translate-y-0.5
              hover:text-[#19191A]
              focus:outline-none
              focus-visible:ring-2
              focus-visible:ring-[#19191A]/20
            "
          >
            {copy.signup}
          </Link>
        </p>
      </div>
    </main>
  );
}