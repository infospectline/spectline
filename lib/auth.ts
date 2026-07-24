import { betterAuth } from "better-auth";
import { prismaAdapter } from "better-auth/adapters/prisma";
import { prisma } from "./prisma";

export const auth = betterAuth({
  database: prismaAdapter(prisma, {
    provider: "postgresql",
  }),

  emailAndPassword: {
    enabled: true,
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
    },
  },
});