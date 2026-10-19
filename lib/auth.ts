import { betterAuth } from "better-auth";
import { APIError } from "better-auth/api";
import { prismaAdapter } from "better-auth/adapters/prisma";

import { prisma } from "./prisma";
import { sendAdminApprovalEmail } from "./registration-approval";

type RegistrationUserFields = {
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

export const auth = betterAuth({
  database: prismaAdapter(prisma, {
    provider: "postgresql",
  }),

  emailAndPassword: {
    enabled: true,
    autoSignIn: false,
  },

  user: {
    additionalFields: {
      firstName: {
        type: "string",
        required: false,
        input: true,
        returned: true,
      },

      lastName: {
        type: "string",
        required: false,
        input: true,
        returned: true,
      },

      address: {
        type: "string",
        required: false,
        input: true,
        returned: true,
      },

      city: {
        type: "string",
        required: false,
        input: true,
        returned: true,
      },

      phone: {
        type: "string",
        required: false,
        input: true,
        returned: true,
      },

      company: {
        type: "string",
        required: false,
        input: true,
        returned: true,
      },

      website: {
        type: "string",
        required: false,
        input: true,
        returned: true,
      },

      approvalStatus: {
        type: ["pending", "approved", "rejected"],
        required: false,
        defaultValue: "pending",
        input: false,
        returned: true,
      },

      approvedAt: {
        type: "date",
        required: false,
        input: false,
        returned: true,
      },
    },
  },

  databaseHooks: {
    user: {
      create: {
        after: async (user) => {
          const registrationUser =
            user as typeof user & RegistrationUserFields;

          try {
            console.log("[approval-debug] User creation hook fired", {
              userId: user.id,
              email: user.email,
            });
            await sendAdminApprovalEmail({
              id: registrationUser.id,
              name: registrationUser.name,
              email: registrationUser.email,
              firstName: registrationUser.firstName,
              lastName: registrationUser.lastName,
              address: registrationUser.address,
              city: registrationUser.city,
              phone: registrationUser.phone,
              company: registrationUser.company,
              website: registrationUser.website,
            });
          } catch (error) {
            console.error(
              "Failed to send the registration approval email.",
              error,
            );

            throw error;
          }
        },
      },
    },

    session: {
      create: {
        before: async (session) => {
          const user = await prisma.user.findUnique({
            where: {
              id: session.userId,
            },
            select: {
              approvalStatus: true,
            },
          });

          if (!user) {
            throw new APIError("UNAUTHORIZED", {
              message: "User account could not be found.",
            });
          }

          if (user.approvalStatus !== "approved") {
            const message =
              user.approvalStatus === "rejected"
                ? "Your registration has been rejected."
                : "Your registration is awaiting administrator approval.";

            throw new APIError("FORBIDDEN", {
              message,
            });
          }
        },
      },
    },
  },
});