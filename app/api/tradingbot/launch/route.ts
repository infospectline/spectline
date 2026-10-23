import { createHmac } from "node:crypto";
import { headers } from "next/headers";
import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const session = await auth.api.getSession({
    headers: await headers(),
  });

  if (!session?.user?.id) {
    return NextResponse.redirect(
      new URL("/login", request.url)
    );
  }

  const secret = process.env.TRADINGBOT_SHARED_SECRET;
  const tradingbotUrl = process.env.NEXT_PUBLIC_TRADINGBOT_URL;

  if (!secret || !tradingbotUrl) {
    return new Response(
      "Tradingbot integration is not configured.",
      { status: 500 }
    );
  }

  const payload = Buffer.from(
    JSON.stringify({
      sub: session.user.id,
      aud: "spectline-tradingbot",
      exp: Math.floor(Date.now() / 1000) + 900,
    })
  ).toString("base64url");

  const signature = createHmac("sha256", secret)
    .update(payload)
    .digest("base64url");

  const target = new URL(tradingbotUrl);
  target.hash = new URLSearchParams({
    ticket: `${payload}.${signature}`,
  }).toString();

  return NextResponse.redirect(target);
}