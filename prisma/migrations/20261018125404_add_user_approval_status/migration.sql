-- AlterTable
ALTER TABLE "user" ADD COLUMN     "approvalStatus" TEXT NOT NULL DEFAULT 'pending',
ADD COLUMN     "approvedAt" TIMESTAMP(3);
