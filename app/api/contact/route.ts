import { NextResponse } from "next/server";
import { Resend } from "resend";

export const runtime = "nodejs";

const resend = new Resend(process.env.RESEND_API_KEY);

export async function POST(request: Request) {
  try {
    const formData = await request.formData();

    const name = String(formData.get("name") ?? "").trim();
    const system = String(formData.get("system") ?? "").trim();
    const email = String(formData.get("email") ?? "").trim();
    const message = String(formData.get("message") ?? "").trim();

    if (!name || !email || !message) {
      return NextResponse.json(
        {
          success: false,
          error: "Please fill in all required fields.",
        },
        { status: 400 }
      );
    }

    const fromEmail = process.env.CONTACT_FROM_EMAIL;
    const toEmail = process.env.CONTACT_TO_EMAIL;
    const resendApiKey = process.env.RESEND_API_KEY;

    if (!resendApiKey || !fromEmail || !toEmail) {
      console.error("Missing required contact form environment variables.", {
        hasResendApiKey: Boolean(resendApiKey),
        hasFromEmail: Boolean(fromEmail),
        hasToEmail: Boolean(toEmail),
      });

      return NextResponse.json(
        {
          success: false,
          error: "The server is not configured correctly.",
        },
        { status: 500 }
      );
    }

    const { data, error } = await resend.emails.send({
      from: `Spectline Contact Form <${fromEmail}>`,
      to: [toEmail],
      replyTo: email,
      subject: `New Spectline message from ${name}`,
      text: [
        `Name: ${name}`,
        `Email: ${email}`,
        `System / Project: ${system || "Not provided"}`,
        "",
        "Message:",
        message,
      ].join("\n"),
      html: `
        <h2>New message from Spectline</h2>

        <p><strong>Name:</strong> ${escapeHtml(name)}</p>
        <p><strong>Email:</strong> ${escapeHtml(email)}</p>
        <p><strong>System / Project:</strong> ${
          system ? escapeHtml(system) : "Not provided"
        }</p>

        <hr />

        <p><strong>Message:</strong></p>
        <p>${escapeHtml(message).replace(/\n/g, "<br />")}</p>
      `,
    });

    if (error) {
      console.error("Resend failed to send the contact form email.", {
        error,
        senderEmail: fromEmail,
        recipientEmail: toEmail,
      });

      return NextResponse.json(
        {
          success: false,
          error: "The email could not be sent.",
        },
        { status: 500 }
      );
    }

    console.info("Contact form email sent successfully.", {
      emailId: data?.id,
      recipientEmail: toEmail,
    });

    return NextResponse.json({
      success: true,
      id: data?.id,
    });
  } catch (error) {
    console.error("Unexpected contact form error.", error);

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