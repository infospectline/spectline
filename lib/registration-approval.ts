
import "server-only";

import { createHmac, timingSafeEqual } from "node:crypto";
import { Resend } from "resend";

export type RegistrationApprovalAction = "approve" | "reject";

export type ApprovalTokenPayload = {
  action: RegistrationApprovalAction;
  userId: string;
  issuedAt: number;
};

export type RegistrationApprovalUser = {
  id: string;
  name: string;
  email: string;
  firstName?: string | null;
  lastName?: string | null;
  address?: string | null;
  city?: string | null;
  phone?: string | null;
  company?: string | null;
  website?: string | null;
};

const DEFAULT_APPROVAL_TOKEN_HOURS = 48;

function getRequiredEnv(name: string): string {
  const value = process.env[name]?.trim();

  if (!value) {
    throw new Error(`Required environment variable is missing: ${name}`);
  }

  return value;
}

function getApprovalTokenHours(): number {
  const rawValue = process.env.APPROVAL_TOKEN_HOURS;

  if (rawValue === undefined) {
    return DEFAULT_APPROVAL_TOKEN_HOURS;
  }

  const hours = Number(rawValue);

  if (!Number.isSafeInteger(hours) || hours < 1) {
    throw new Error(
      "APPROVAL_TOKEN_HOURS must be a positive integer.",
    );
  }

  return hours;
}

function signPayload(payload: string): string {
  const secret = getRequiredEnv("APPROVAL_TOKEN_SECRET");

  return createHmac("sha256", secret)
    .update(payload)
    .digest("base64url");
}

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (character) => {
    const entities: Record<string, string> = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    };

    return entities[character];
  });
}

function getApprovalUrl(token: string): string {
  const baseUrl = getRequiredEnv("APP_BASE_URL");
  const parsedBaseUrl = new URL(baseUrl);

  if (
    parsedBaseUrl.protocol !== "https:" &&
    parsedBaseUrl.protocol !== "http:"
  ) {
    throw new Error("APP_BASE_URL must use HTTP or HTTPS.");
  }

  return new URL(
    `/admin/approval/${encodeURIComponent(token)}`,
    parsedBaseUrl.origin,
  ).toString();
}

export function createApprovalToken(
  action: RegistrationApprovalAction,
  userId: string,
): string {
  if (action !== "approve" && action !== "reject") {
    throw new Error("Invalid registration approval action.");
  }

  const normalizedUserId = userId.trim();

  if (!normalizedUserId) {
    throw new Error("A user ID is required to create an approval token.");
  }

  const payload: ApprovalTokenPayload = {
    action,
    userId: normalizedUserId,
    issuedAt: Math.floor(Date.now() / 1000),
  };

  const encodedPayload = Buffer.from(
    JSON.stringify(payload),
    "utf8",
  ).toString("base64url");

  const signature = signPayload(encodedPayload);

  return `${encodedPayload}.${signature}`;
}

export function verifyApprovalToken(
  token: string,
): ApprovalTokenPayload | null {
  if (!token || token.length > 4096) {
    return null;
  }

  const parts = token.split(".");

  if (parts.length !== 2 || !parts[0] || !parts[1]) {
    return null;
  }

  const [encodedPayload, suppliedSignature] = parts;
  const expectedSignature = signPayload(encodedPayload);

  const suppliedBytes = Buffer.from(suppliedSignature, "utf8");
  const expectedBytes = Buffer.from(expectedSignature, "utf8");

  if (
    suppliedBytes.length !== expectedBytes.length ||
    !timingSafeEqual(suppliedBytes, expectedBytes)
  ) {
    return null;
  }

  try {
    const decoded: unknown = JSON.parse(
      Buffer.from(encodedPayload, "base64url").toString("utf8"),
    );

    if (
      typeof decoded !== "object" ||
      decoded === null
    ) {
      return null;
    }

    const payload = decoded as Record<string, unknown>;

    if (
      (payload.action !== "approve" && payload.action !== "reject") ||
      typeof payload.userId !== "string" ||
      !payload.userId.trim() ||
      typeof payload.issuedAt !== "number" ||
      !Number.isSafeInteger(payload.issuedAt)
    ) {
      return null;
    }

    const now = Math.floor(Date.now() / 1000);
    const maxAgeSeconds = getApprovalTokenHours() * 60 * 60;

    if (payload.issuedAt > now + 60) {
      return null;
    }

    if (now - payload.issuedAt > maxAgeSeconds) {
      return null;
    }

    return {
      action: payload.action,
      userId: payload.userId,
      issuedAt: payload.issuedAt,
    };
  } catch {
    return null;
  }
}

export async function sendAdminApprovalEmail(
  user: RegistrationApprovalUser,
): Promise<void> {
  if (!user.id.trim() || !user.name.trim() || !user.email.trim()) {
    throw new Error(
      "The user ID, name, and email are required for approval.",
    );
  }

  const adminEmail = getRequiredEnv("ADMIN_EMAIL");
  const fromEmail = getRequiredEnv("RESEND_FROM_EMAIL");
  const resendApiKey = getRequiredEnv("RESEND_API_KEY");

  const approvalToken = createApprovalToken("approve", user.id);
  const rejectionToken = createApprovalToken("reject", user.id);

  const approvalUrl = getApprovalUrl(approvalToken);
  const rejectionUrl = getApprovalUrl(rejectionToken);

  const details: Array<[string, string | null | undefined]> = [
    ["Name", user.name],
    ["First name", user.firstName],
    ["Last name", user.lastName],
    ["Email", user.email],
    ["Address", user.address],
    ["City", user.city],
    ["Phone", user.phone],
    ["Company", user.company],
    ["Website", user.website],
  ];

  const populatedDetails = details.filter(
    (entry): entry is [string, string] =>
      typeof entry[1] === "string" && entry[1].trim().length > 0,
  );

  const htmlDetails = populatedDetails
    .map(
      ([label, value]) => `
        <tr>
          <td style="padding: 8px 12px; border-bottom: 1px solid #eeeeee;">
            <strong>${escapeHtml(label)}</strong>
          </td>
          <td style="padding: 8px 12px; border-bottom: 1px solid #eeeeee;">
            ${escapeHtml(value)}
          </td>
        </tr>
      `,
    )
    .join("");

  const textDetails = populatedDetails
    .map(([label, value]) => `${label}: ${value}`)
    .join("\n");

  const tokenHours = getApprovalTokenHours();

  const html = `
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>New Spectline Registration</title>
      </head>
      <body style="margin:0; padding:30px; background:#f4f4f4; font-family:Arial,sans-serif; color:#19191a;">
        <main style="max-width:640px; margin:0 auto; padding:30px; background:#ffffff; border-radius:12px;">
          <h1 style="font-size:24px;">New Spectline Registration</h1>

          <p>A new user has registered on Spectline and is waiting for your decision.</p>

          <table style="width:100%; border-collapse:collapse; margin:24px 0;">
            <tbody>
              ${htmlDetails}
            </tbody>
          </table>

          <p style="margin:28px 0 16px;">Review the registration and select an action:</p>

          <a
            href="${escapeHtml(approvalUrl)}"
            style="display:inline-block; padding:13px 22px; background:#15803d; color:#ffffff; text-decoration:none; border-radius:6px; font-weight:bold; margin-right:10px;"
          >
            APPROVE
          </a>

          <a
            href="${escapeHtml(rejectionUrl)}"
            style="display:inline-block; padding:13px 22px; background:#b91c1c; color:#ffffff; text-decoration:none; border-radius:6px; font-weight:bold;"
          >
            REJECT
          </a>

          <p style="margin-top:30px; color:#666666; font-size:13px;">
            These links expire after ${tokenHours} hours.
            Opening a link should display a confirmation page before the decision is applied.
          </p>
        </main>
      </body>
    </html>
  `;

  const text = [
    "New Spectline Registration",
    "",
    "A new user has registered and is waiting for your decision.",
    "",
    textDetails,
    "",
    `APPROVE: ${approvalUrl}`,
    `REJECT: ${rejectionUrl}`,
    "",
    `These links expire after ${tokenHours} hours.`,
    "Confirm the decision on the page opened by the link.",
  ].join("\n");

  const resend = new Resend(resendApiKey);

  const { error } = await resend.emails.send({
    from: fromEmail,
    to: [adminEmail],
    subject: "New Spectline user registration",
    html,
    text,
  });

  if (error) {
    throw new Error(
      "Failed to send the registration approval email through Resend.",
    );
  }
}
