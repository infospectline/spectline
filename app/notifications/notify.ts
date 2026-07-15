import "server-only";
import { createClient } from "@supabase/supabase-js";

export type NotificationSeverity = "info" | "success" | "error";

export async function pushNotification(input: {
  userId: string;
  category: string;
  event: string;
  title: string;
  message: string;
  severity: NotificationSeverity;
  source?: string;
  entityType?: string | null;
}) {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !key) return;

  const supabase = createClient(url, key, { auth: { persistSession: false } });

  const { error } = await supabase.from("notifications").insert({
    user_id: input.userId,
    category: input.category,
    event: input.event,
    title: input.title,
    message: input.message,
    severity: input.severity,
    source: input.source ?? "api",
    entity_type: input.entityType ?? null,
  });

  if (error) throw error;
}
