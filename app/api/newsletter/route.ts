import { createHash, randomBytes } from "crypto";
import { NextResponse } from "next/server";
import { Resend } from "resend";

import { prisma } from "@/lib/prisma";
import sk from "@/lib/json/sk.json";
import en from "@/lib/json/en.json";

export const runtime = "nodejs";

type NewsletterConfirmationEmailCopy = {
  subject: string;
  heading: string;
  text: string;
  button: string;
  expiration: string;
};

const resend = new Resend(
  process.env.RESEND_NEWSLETTER_API_KEY
);

const CONFIRMATION_TOKEN_TTL_MS = 24 * 60 * 60 * 1000;

function normalizeEmail(value: string) {
  return value.trim().toLowerCase();
}

function isValidEmail(value: string) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
}

function createToken() {
  return randomBytes(32).toString("hex");
}

function hashToken(token: string) {
  return createHash("sha256").update(token).digest("hex");
}

export async function POST(request: Request) {
  try {
    const newsletterApiKey =
      process.env.RESEND_NEWSLETTER_API_KEY;
    const siteUrl = process.env.BETTER_AUTH_URL;

    if (!newsletterApiKey || !siteUrl) {
      console.error(
        "Missing required newsletter environment variables.",
        {
          hasNewsletterApiKey: Boolean(newsletterApiKey),
          hasSiteUrl: Boolean(siteUrl),
        }
      );

      return NextResponse.json(
        {
          success: false,
          error: "The server is not configured correctly.",
        },
        { status: 500 }
      );
    }

    const body = (await request.json()) as {
      email?: unknown;
      language?: unknown;
    };

    const email =
      typeof body.email === "string"
        ? normalizeEmail(body.email)
        : "";

    const language =
      body.language === "en" ? "en" : "sk";

    if (
      !email ||
      email.length > 254 ||
      !isValidEmail(email)
    ) {
      return NextResponse.json(
        {
          success: false,
          error: "Invalid email address.",
        },
        { status: 400 }
      );
    }

    const confirmationToken = createToken();
    const confirmationTokenHash = hashToken(
      confirmationToken
    );

    const confirmationExpiresAt = new Date(
      Date.now() + CONFIRMATION_TOKEN_TTL_MS
    );

    const existingSubscriber =
      await prisma.newsletterSubscriber.findUnique({
        where: {
          email,
        },
      });

    if (existingSubscriber?.status === "ACTIVE") {
      return NextResponse.json({
        success: true,
        status: "already_active",
      });
    }

    const subscriber =
      await prisma.newsletterSubscriber.upsert({
        where: {
          email,
        },
        create: {
          email,
          language,
          status: "PENDING",
          confirmationTokenHash,
          confirmationExpiresAt,
          confirmedAt: null,
          unsubscribedAt: null,
        },
        update: {
          language,
          status: "PENDING",
          confirmationTokenHash,
          confirmationExpiresAt,
          confirmedAt: null,
          unsubscribedAt: null,
        },
      });

    const confirmationUrl = new URL(
      "/api/newsletter/confirm",
      siteUrl
    );

    confirmationUrl.searchParams.set(
      "token",
      confirmationToken
    );

   const selectedCopy = language === "en" ? en : sk;

  const confirmationCopy = (
    selectedCopy as typeof selectedCopy & {
      newsletterEmail?: {
        confirmation?: NewsletterConfirmationEmailCopy;
      };
    }
  ).newsletterEmail?.confirmation;

  if (!confirmationCopy) {
    console.error(
      "Missing newsletter confirmation email translations.",
      {
        language,
      }
    );

    return NextResponse.json(
      {
        success: false,
        error: "Newsletter email translations are missing.",
      },
      { status: 500 }
    );
  }

  const { data: emailData, error: emailError } =
    await resend.emails.send({
      from: "Spectline Newsletter <newsletter@spectline.com>",
      to: [email],
      subject: confirmationCopy.subject,
      text: [
        confirmationCopy.heading,
        "",
        confirmationCopy.text,
        "",
        confirmationUrl.toString(),
        "",
        confirmationCopy.expiration,
      ].join("\n"),
      html: `
        <!doctype html>
        <html lang="${language}">
          <head>
            <meta charset="utf-8" />
            <meta
              name="viewport"
              content="width=device-width, initial-scale=1"
            />
            <meta name="color-scheme" content="light" />
            <meta
              name="supported-color-schemes"
              content="light"
            />
            <title>
              ${escapeHtml(confirmationCopy.subject)}
            </title>
          </head>

          <body
            style="
              margin: 0;
              padding: 0;
              background-color: #C7C7C7;
              color: #19191A;
              font-family: Arial, Helvetica, sans-serif;
              -webkit-text-size-adjust: 100%;
              text-size-adjust: 100%;
            "
          >
            <table
              role="presentation"
              width="100%"
              cellspacing="0"
              cellpadding="0"
              border="0"
              style="
                width: 100%;
                background-color: #C7C7C7;
                border-collapse: collapse;
              "
            >
              <tr>
                <td
                  align="center"
                  style="
                    padding: 40px 16px;
                  "
                >
                  <table
                    role="presentation"
                    width="100%"
                    cellspacing="0"
                    cellpadding="0"
                    border="0"
                    style="
                      width: 100%;
                      max-width: 600px;
                      border-collapse: separate;
                      background-color: #E7E7E7;
                      border: 1px solid rgba(25, 25, 26, 0.15);
                      border-radius: 24px;
                      box-shadow: 0 18px 60px rgba(25, 25, 26, 0.12);
                      overflow: hidden;
                    "
                  >
                    <tr>
                      <td
                        style="
                          padding: 28px 32px 20px;
                          border-bottom: 1px solid rgba(25, 25, 26, 0.10);
                        "
                      >
                        <table
                          role="presentation"
                          width="100%"
                          cellspacing="0"
                          cellpadding="0"
                          border="0"
                          style="
                            width: 100%;
                            border-collapse: collapse;
                          "
                        >
                          <tr>
                            <td
                              style="
                                font-size: 13px;
                                line-height: 1.4;
                                font-weight: 700;
                                letter-spacing: 0.16em;
                                text-transform: uppercase;
                                color: rgba(25, 25, 26, 0.68);
                              "
                            >
                              SPECTLINE
                            </td>
                          </tr>
                        </table>
                      </td>
                    </tr>

                    <tr>
                      <td
                        style="
                          padding: 38px 32px 34px;
                          text-align: center;
                        "
                      >
                        <h1
                          style="
                            margin: 0;
                            font-size: 30px;
                            line-height: 1.2;
                            font-weight: 700;
                            letter-spacing: -0.02em;
                            color: #19191A;
                            text-align: center;
                          "
                        >
                          ${escapeHtml(
                            confirmationCopy.heading
                          )}
                        </h1>

                        <p
                          style="
                            margin: 20px 0 0;
                            font-size: 16px;
                            line-height: 1.7;
                            color: rgba(25, 25, 26, 0.72);
                            text-align: center;
                          "
                        >
                          ${escapeHtml(
                            confirmationCopy.text
                          )}
                        </p>

                        <table
                          role="presentation"
                          width="100%"
                          cellspacing="0"
                          cellpadding="0"
                          border="0"
                          style="
                            width: 100%;
                            margin: 30px 0 0;
                            border-collapse: separate;
                          "
                        >
                          <tr>
                            <td
                              align="center"
                              width="100%"
                              bgcolor="#E7E7E7"
                              style="
                                width: 100%;
                                border-radius: 14px;
                                background-color: #E7E7E7;
                                border: 1px solid rgba(25, 25, 26, 0.18);
                                box-shadow: 0 6px 18px rgba(25, 25, 26, 0.08);
                              "
                            >
                              <a
                                href="${confirmationUrl.toString()}"
                                style="
                                  display: block;
                                  width: 100%;
                                  box-sizing: border-box;
                                  padding: 16px 24px;
                                  border-radius: 14px;
                                  background-color: #E7E7E7;
                                  color: #19191A;
                                  font-size: 16px;
                                  line-height: 1.2;
                                  font-weight: 700;
                                  text-align: center;
                                  text-decoration: none;
                                "
                              >
                                ${escapeHtml(
                                  confirmationCopy.button
                                )}
                              </a>
                            </td>
                          </tr>
                        </table>

                        <div
                          style="
                            margin-top: 32px;
                            padding: 18px 20px;
                            border: 1px solid rgba(25, 25, 26, 0.10);
                            border-radius: 16px;
                            background-color: rgba(244, 244, 244, 0.58);
                            text-align: center;
                          "
                        >
                          <p
                            style="
                              margin: 0;
                              font-size: 13px;
                              line-height: 1.65;
                              color: rgba(25, 25, 26, 0.60);
                              text-align: center;
                            "
                          >
                            ${escapeHtml(
                              confirmationCopy.expiration
                            )}
                          </p>
                        </div>
                      </td>
                    </tr>

                    <tr>
                      <td
                        style="
                          padding: 18px 32px 24px;
                          border-top: 1px solid rgba(25, 25, 26, 0.10);
                          font-size: 12px;
                          line-height: 1.6;
                          text-align: center;
                          color: rgba(25, 25, 26, 0.45);
                        "
                      >
                      </td>
                    </tr>
                  </table>
                </td>
              </tr>
            </table>
          </body>
        </html>
      `,
    });

  if (emailError) {
    console.error(
      "Resend failed to send the newsletter confirmation email.",
      {
        error: emailError,
        subscriberId: subscriber.id,
      }
    );

    return NextResponse.json(
      {
        success: false,
        error: "The confirmation email could not be sent.",
      },
      { status: 500 }
    );
  }

  console.info(
    "Newsletter confirmation email sent successfully.",
    {
      subscriberId: subscriber.id,
      emailId: emailData?.id,
      status: subscriber.status,
      language: subscriber.language,
    }
  );

  return NextResponse.json({
    success: true,
    status: "pending",
  });
  } catch (error) {
    console.error(
      "Unexpected newsletter subscription error.",
      error
    );

    return NextResponse.json(
      {
        success: false,
        error: "An unexpected error occurred.",
      },
      { status: 500 }
    );
  }
}

function escapeHtml(value: string) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}