import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { verifyApprovalToken } from "@/lib/registration-approval";

export const runtime = "nodejs";

type ApprovalResult =
  | "approved"
  | "rejected"
  | "invalid"
  | "already_processed"
  | "error";

function redirectToHome(status: ApprovalResult) {
  const siteUrl =
    process.env.APP_BASE_URL || process.env.BETTER_AUTH_URL;

  if (!siteUrl) {
    console.error("Missing APP_BASE_URL for approval redirect.");

    return NextResponse.json(
      { success: false, error: "Server configuration error." },
      { status: 500 },
    );
  }

  const url = new URL("/", siteUrl);
  url.searchParams.set("approval", status);

  return NextResponse.redirect(url);
}

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ token: string }> },
) {
  try {
    const { token } = await params;
    const payload = await verifyApprovalToken(token);

    if (
      !payload ||
      (payload.action !== "approve" && payload.action !== "reject") ||
      !payload.userId
    ) {
      return redirectToHome("invalid");
    }

    const nextStatus =
      payload.action === "approve" ? "approved" : "rejected";

    const result = await prisma.user.updateMany({
      where: {
        id: payload.userId,
        approvalStatus: "pending",
      },
      data: {
        approvalStatus: nextStatus,
        approvedAt:
          nextStatus === "approved" ? new Date() : null,
      },
    });

    if (result.count !== 1) {
      const existingUser = await prisma.user.findUnique({
        where: { id: payload.userId },
        select: { id: true },
      });

      return redirectToHome(
        existingUser ? "already_processed" : "invalid",
      );
    }

    return redirectToHome(nextStatus);
  } catch (error) {
    console.error("Registration approval failed.", error);
    return redirectToHome("error");
  }
}