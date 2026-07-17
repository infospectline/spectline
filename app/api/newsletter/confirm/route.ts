import { createHash, randomBytes } from "crypto";
import { NextResponse } from "next/server";
import { Resend } from "resend";

import { prisma } from "@/lib/prisma";
import sk from "@/lib/json/sk.json";
import en from "@/lib/json/en.json";

export const runtime = "nodejs";

type NewsletterWelcomeEmailCopy = {
  subject: string;
  heading: string;
  text: string;
  unsubscribeText: string;
  unsubscribeButton: string;
};

const resend = new Resend(
  process.env.RESEND_NEWSLETTER_API_KEY
);

function hashToken(token: string) {
  return createHash("sha256").update(token).digest("hex");
}

function createToken() {
  return randomBytes(32).toString("hex");
}

function escapeHtml(value: string) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

export async function GET(request: Request) {
  try {
    const siteUrl = process.env.BETTER_AUTH_URL;

    if (!siteUrl) {
      console.error(
        "Missing BETTER_AUTH_URL for newsletter confirmation."
      );

      return NextResponse.json(
        {
          success: false,
          error: "The server is not configured correctly.",
        },
        { status: 500 }
      );
    }

    const requestUrl = new URL(request.url);
    const token = requestUrl.searchParams.get("token")?.trim();

    if (!token) {
      const redirectUrl = new URL("/", siteUrl);
      redirectUrl.searchParams.set(
        "newsletter",
        "invalid_confirmation"
      );

      return NextResponse.redirect(redirectUrl);
    }

    const confirmationTokenHash = hashToken(token);

    const subscriber =
      await prisma.newsletterSubscriber.findFirst({
        where: {
          confirmationTokenHash,
          status: "PENDING",
          confirmationExpiresAt: {
            gt: new Date(),
          },
        },
      });

    if (!subscriber) {
      const redirectUrl = new URL("/", siteUrl);
      redirectUrl.searchParams.set(
        "newsletter",
        "invalid_or_expired"
      );

      return NextResponse.redirect(redirectUrl);
    }

    const unsubscribeToken = createToken();
    const unsubscribeTokenHash = hashToken(
      unsubscribeToken
    );

    const confirmedSubscriber =
      await prisma.newsletterSubscriber.update({
        where: {
          id: subscriber.id,
        },
        data: {
          status: "ACTIVE",
          confirmedAt: new Date(),
          unsubscribedAt: null,
          confirmationTokenHash: null,
          confirmationExpiresAt: null,
          unsubscribeTokenHash,
        },
      });

    const unsubscribeUrl = new URL(
      "/api/newsletter/unsubscribe",
      siteUrl
    );

    unsubscribeUrl.searchParams.set(
      "token",
      unsubscribeToken
    );

    const newsletterSegmentId =
      process.env.RESEND_NEWSLETTER_SEGMENT_ID;

    if (!newsletterSegmentId) {
      console.error(
        "Missing RESEND_NEWSLETTER_SEGMENT_ID.",
        {
          subscriberId: confirmedSubscriber.id,
        }
      );

      return NextResponse.json(
        {
          success: false,
          error: "The newsletter segment is not configured.",
        },
        { status: 500 }
      );
    }

    const {
      data: contactData,
      error: contactError,
    } = await resend.contacts.create({
      email: confirmedSubscriber.email,
      unsubscribed: false,
    });

    if (contactError) {
      const contactAlreadyExists =
        contactError.message
          ?.toLowerCase()
          .includes("already exists") ?? false;

      if (!contactAlreadyExists) {
        console.error(
          "Resend failed to create the newsletter contact.",
          {
            error: contactError,
            subscriberId: confirmedSubscriber.id,
          }
        );

        return NextResponse.json(
          {
            success: false,
            error: "The newsletter contact could not be created.",
          },
          { status: 500 }
        );
      }
    }

    const {
      error: segmentError,
    } = await resend.contacts.segments.add({
      email: confirmedSubscriber.email,
      segmentId: newsletterSegmentId,
    });

    if (segmentError) {
      console.error(
        "Resend failed to add the contact to the newsletter segment.",
        {
          error: segmentError,
          subscriberId: confirmedSubscriber.id,
          contactId: contactData?.id,
        }
      );

      return NextResponse.json(
        {
          success: false,
          error: "The contact could not be added to the newsletter segment.",
        },
        { status: 500 }
      );
    }

    console.info(
      "Newsletter contact added to the Resend segment.",
      {
        subscriberId: confirmedSubscriber.id,
        contactId: contactData?.id,
        segmentId: newsletterSegmentId,
      }
    );

    const selectedCopy =
      confirmedSubscriber.language === "en" ? en : sk;

    const welcomeCopy = (
      selectedCopy as typeof selectedCopy & {
        newsletterEmail?: {
          welcome?: NewsletterWelcomeEmailCopy;
        };
      }
    ).newsletterEmail?.welcome;

    if (!welcomeCopy) {
      console.error(
        "Missing newsletter welcome email translations.",
        {
          language: confirmedSubscriber.language,
          subscriberId: confirmedSubscriber.id,
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
        to: [confirmedSubscriber.email],
        subject: welcomeCopy.subject,
        text: [
          welcomeCopy.heading,
          "",
          welcomeCopy.text,
          "",
          welcomeCopy.unsubscribeText,
          "",
          unsubscribeUrl.toString(),
        ].join("\n"),
        html: `
          <!doctype html>
          <html lang="${confirmedSubscriber.language}">
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
                ${escapeHtml(welcomeCopy.subject)}
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
                              welcomeCopy.heading
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
                              welcomeCopy.text
                            )}
                          </p>

                          <div
                            style="
                              margin-top: 30px;
                              padding: 20px;
                              border: 1px solid rgba(25, 25, 26, 0.10);
                              border-radius: 18px;
                              background-color: rgba(244, 244, 244, 0.62);
                              text-align: center;
                            "
                          >
                            <p
                              style="
                                margin: 0;
                                font-size: 14px;
                                line-height: 1.65;
                                color: rgba(25, 25, 26, 0.64);
                                text-align: center;
                              "
                            >
                              ${escapeHtml(
                                welcomeCopy.unsubscribeText
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
                                margin-top: 20px;
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
                                    href="${unsubscribeUrl.toString()}"
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
                                      welcomeCopy.unsubscribeButton
                                    )}
                                  </a>
                                </td>
                              </tr>
                            </table>
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
        "Resend failed to send the newsletter welcome email.",
        {
          error: emailError,
          subscriberId: confirmedSubscriber.id,
        }
      );

      return NextResponse.json(
        {
          success: false,
          error: "The welcome email could not be sent.",
        },
        { status: 500 }
      );
    }

    console.info(
      "Newsletter welcome email sent successfully.",
      {
        subscriberId: confirmedSubscriber.id,
        emailId: emailData?.id,
        status: confirmedSubscriber.status,
        language: confirmedSubscriber.language,
      }
    );

    const redirectUrl = new URL("/", siteUrl);

    redirectUrl.searchParams.set(
      "newsletter",
      "confirmed"
    );

    return NextResponse.redirect(redirectUrl);
  } catch (error) {
    console.error(
      "Unexpected newsletter confirmation error.",
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