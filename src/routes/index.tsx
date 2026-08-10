import { createFileRoute, redirect } from "@tanstack/react-router";
import { supabase } from "@/integrations/supabase/client";

export const Route = createFileRoute("/")({
  ssr: false,
  head: () => ({
    meta: [
      { title: "Vigilant Eye — AI Smart Surveillance" },
      {
        name: "description",
        content:
          "Command-center console for multi-camera AI monitoring and suspicious cheating activity detection.",
      },
      { property: "og:title", content: "Vigilant Eye — AI Smart Surveillance" },
      {
        property: "og:description",
        content:
          "Command-center console for multi-camera AI monitoring and suspicious cheating activity detection.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  beforeLoad: async () => {
    const { data } = await supabase.auth.getSession();
    throw redirect({ to: data.session ? "/dashboard" : "/login" });
  },
  component: () => null,
});
