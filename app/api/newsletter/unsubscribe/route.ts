import { createHash } from "crypto";
import { NextResponse } from "next/server";
import { Resend } from "resend";

import { prisma } from "@/lib/prisma";

export const runtime = "nodejs";

const resend = new Resend(
  process.env.RESEND_NEWSLETTER_API_KEY
);

function hashToken(token: string) {
  return createHash("sha256").update(token).digest("hex");
}

export async function GET(request: Request) {
  try {
    const siteUrl = process.env.BETTER_AUTH_URL;
    const newsletterSegmentId =
      process.env.RESEND_NEWSLETTER_SEGMENT_ID;

    if (!siteUrl || !newsletterSegmentId) {
      console.error(
        "Missing required newsletter unsubscribe environment variables.",
        {
          hasSiteUrl: Boolean(siteUrl),
          hasNewsletterSegmentId: Boolean(
            newsletterSegmentId
          ),
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

    const requestUrl = new URL(request.url);
    const token = requestUrl.searchParams.get("token")?.trim();

    if (!token) {
      const redirectUrl = new URL("/", siteUrl);

      redirectUrl.searchParams.set(
        "newsletter",
        "invalid_unsubscribe"
      );

      return NextResponse.redirect(redirectUrl);
    }

    const unsubscribeTokenHash = hashToken(token);

    const subscriber =
      await prisma.newsletterSubscriber.findFirst({
        where: {
          unsubscribeTokenHash,
          status: "ACTIVE",
        },
      });

    if (!subscriber) {
      const redirectUrl = new URL("/", siteUrl);

      redirectUrl.searchParams.set(
        "newsletter",
        "invalid_or_used_unsubscribe"
      );

      return NextResponse.redirect(redirectUrl);
    }

    const updatedSubscriber =
      await prisma.newsletterSubscriber.update({
        where: {
          id: subscriber.id,
        },
        data: {
          status: "UNSUBSCRIBED",
          unsubscribedAt: new Date(),
          unsubscribeTokenHash: null,
        },
      });

    const { error: segmentError } =
      await resend.contacts.segments.remove({
        email: updatedSubscriber.email,
        segmentId: newsletterSegmentId,
      });

    if (segmentError) {
      console.error(
        "Resend failed to remove the contact from the newsletter segment.",
        {
          error: segmentError,
          subscriberId: updatedSubscriber.id,
        }
      );
    }

    const { error: contactError } =
      await resend.contacts.update({
        email: updatedSubscriber.email,
        unsubscribed: true,
      });

    if (contactError) {
      console.error(
        "Resend failed to mark the newsletter contact as unsubscribed.",
        {
          error: contactError,
          subscriberId: updatedSubscriber.id,
        }
      );
    }

    console.info(
      "Newsletter subscription unsubscribed successfully.",
      {
        subscriberId: updatedSubscriber.id,
        status: updatedSubscriber.status,
      }
    );

    const redirectUrl = new URL("/", siteUrl);

    redirectUrl.searchParams.set(
      "newsletter",
      "unsubscribed"
    );

    return NextResponse.redirect(redirectUrl);
  } catch (error) {
    console.error(
      "Unexpected newsletter unsubscribe error.",
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