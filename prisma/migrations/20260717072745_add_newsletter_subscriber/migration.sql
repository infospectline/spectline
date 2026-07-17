-- CreateEnum
CREATE TYPE "NewsletterStatus" AS ENUM ('PENDING', 'ACTIVE', 'UNSUBSCRIBED');

-- CreateTable
CREATE TABLE "newsletter_subscriber" (
    "id" TEXT NOT NULL,
    "email" TEXT NOT NULL,
    "status" "NewsletterStatus" NOT NULL DEFAULT 'PENDING',
    "language" TEXT NOT NULL DEFAULT 'sk',
    "confirmationTokenHash" TEXT,
    "confirmationExpiresAt" TIMESTAMP(3),
    "unsubscribeTokenHash" TEXT,
    "confirmedAt" TIMESTAMP(3),
    "unsubscribedAt" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "newsletter_subscriber_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "newsletter_subscriber_email_key" ON "newsletter_subscriber"("email");

-- CreateIndex
CREATE UNIQUE INDEX "newsletter_subscriber_confirmationTokenHash_key" ON "newsletter_subscriber"("confirmationTokenHash");

-- CreateIndex
CREATE UNIQUE INDEX "newsletter_subscriber_unsubscribeTokenHash_key" ON "newsletter_subscriber"("unsubscribeTokenHash");

-- CreateIndex
CREATE INDEX "newsletter_subscriber_status_idx" ON "newsletter_subscriber"("status");

-- CreateIndex
CREATE INDEX "newsletter_subscriber_createdAt_idx" ON "newsletter_subscriber"("createdAt");
