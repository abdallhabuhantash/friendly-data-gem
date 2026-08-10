import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useQueryClient } from "@tanstack/react-query";
import { Panel } from "@/components/common/Panel";
import { PageContainer } from "@/components/layout/PageContainer";
import { TopBar } from "@/components/layout/TopBar";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/hooks/use-auth";
import { formatRelative } from "@/lib/format";

export const Route = createFileRoute("/_authenticated/profile")({
  head: () => ({
    meta: [
      { title: "Profile — Vigilant Eye AI Smart Surveillance" },
      { name: "description", content: "Signed-in operator account details and session controls." },
    ],
  }),
  component: ProfilePage,
});

function ProfilePage() {
  const { user, signOut } = useAuth();
  const queryClient = useQueryClient();
  const navigate = useNavigate();

  const handleSignOut = async () => {
    await queryClient.cancelQueries();
    queryClient.clear();
    await signOut();
    await navigate({ to: "/login", replace: true });
  };
  return (
    <>
      <TopBar title="Profile" subtitle="Current session" />
      <PageContainer>
        <Panel title="Account" bodyClassName="space-y-3 p-3">
          <dl className="grid grid-cols-2 gap-y-2 text-[13px]">
            <dt className="label-tech text-muted-foreground">Name</dt>
            <dd className="text-right text-foreground">{user?.fullName ?? "—"}</dd>
            <dt className="label-tech text-muted-foreground">Email</dt>
            <dd className="text-right font-mono text-[12px] text-foreground">
              {user?.email ?? "—"}
            </dd>
            <dt className="label-tech text-muted-foreground">Role</dt>
            <dd className="text-right font-mono text-[12px] uppercase text-primary">
              {user?.role ?? "—"}
            </dd>
            <dt className="label-tech text-muted-foreground">Last active</dt>
            <dd className="text-right font-mono text-[12px] text-muted-foreground">
              {user ? formatRelative(user.lastActiveAt) : "—"}
            </dd>
          </dl>
          <Button variant="outline" className="h-8 text-xs" onClick={() => void handleSignOut()}>
            Sign out
          </Button>
        </Panel>
      </PageContainer>
    </>
  );
}
