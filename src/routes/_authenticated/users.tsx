import { requireAdministrator } from "@/lib/require-admin";
import { createFileRoute } from "@tanstack/react-router";
import { useServerFn } from "@tanstack/react-start";
import { useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";
import { toast } from "sonner";
import { Panel } from "@/components/common/Panel";
import { StatusDot } from "@/components/common/StatusDot";
import { PageContainer } from "@/components/layout/PageContainer";
import { TopBar } from "@/components/layout/TopBar";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useAuth } from "@/hooks/use-auth";
import { useUsers } from "@/hooks/use-monitoring";
import { createAccount } from "@/lib/admin-users.functions";
import { formatRelative } from "@/lib/format";

export const Route = createFileRoute("/_authenticated/users")({
  beforeLoad: requireAdministrator,
  head: () => ({
    meta: [
      { title: "Users & Roles — Vigilant Eye AI Smart Surveillance" },
      {
        name: "description",
        content: "Administrator and operator accounts with role-based access to the system.",
      },
    ],
  }),
  component: UsersPage,
});

function UsersPage() {
  const users = useUsers();
  const { isAdministrator } = useAuth();
  return (
    <>
      <TopBar title="Users & Roles" subtitle="Administrator access" />
      <PageContainer>
        {isAdministrator && <NewAccountPanel />}
        <Panel
          title="Accounts"
          subtitle="Operators review events; administrators also manage cameras, rules and users."
          bodyClassName="p-0"
        >
          <table className="w-full text-left text-[13px]">
            <thead>
              <tr className="border-b border-border/70">
                <th className="label-tech px-3 py-2">Status</th>
                <th className="label-tech px-3 py-2">Name</th>
                <th className="label-tech px-3 py-2">Email</th>
                <th className="label-tech px-3 py-2">Role</th>
                <th className="label-tech px-3 py-2">Last active</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/50">
              {(users.data ?? []).map((user) => (
                <tr key={user.id} className="hover:bg-surface-2/60">
                  <td className="px-3 py-2">
                    <div className="flex items-center gap-2">
                      <StatusDot
                        tone={user.status === "active" ? "online" : "offline"}
                        pulse={user.status === "active"}
                      />
                      <span className="label-tech text-muted-foreground">{user.status}</span>
                    </div>
                  </td>
                  <td className="px-3 py-2 text-foreground">{user.fullName}</td>
                  <td className="px-3 py-2 font-mono text-[11px] text-muted-foreground">
                    {user.email}
                  </td>
                  <td className="px-3 py-2">
                    <span className="rounded-[3px] border border-primary/40 bg-primary/10 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.1em] text-primary">
                      {user.role}
                    </span>
                  </td>
                  <td className="px-3 py-2 font-mono text-[11px] text-muted-foreground">
                    {formatRelative(user.lastActiveAt)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      </PageContainer>
    </>
  );
}

function NewAccountPanel() {
  const queryClient = useQueryClient();
  const create = useServerFn(createAccount);
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<"administrator" | "operator">("operator");
  const [pending, setPending] = useState(false);

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setPending(true);
    try {
      await create({ data: { fullName, email, password, role } });
      toast.success("Account created");
      setFullName("");
      setEmail("");
      setPassword("");
      void queryClient.invalidateQueries({ queryKey: ["users"] });
    } catch (caught) {
      toast.error(caught instanceof Error ? caught.message : "Could not create the account.");
    } finally {
      setPending(false);
    }
  };

  return (
    <Panel
      title="Create account"
      subtitle="Administrators provision every account; there is no public registration."
      bodyClassName="p-3"
    >
      <form onSubmit={submit} className="grid gap-2 md:grid-cols-5">
        <Input
          placeholder="Full name"
          required
          value={fullName}
          onChange={(event) => setFullName(event.target.value)}
          className="h-8 text-xs"
        />
        <Input
          type="email"
          placeholder="Email"
          required
          value={email}
          onChange={(event) => setEmail(event.target.value)}
          className="h-8 font-mono text-xs"
        />
        <Input
          type="password"
          placeholder="Temporary password"
          required
          minLength={8}
          value={password}
          onChange={(event) => setPassword(event.target.value)}
          className="h-8 font-mono text-xs"
        />
        <select
          value={role}
          onChange={(event) => setRole(event.target.value as "administrator" | "operator")}
          className="h-8 rounded-[4px] border border-border bg-surface-2 px-2 font-mono text-xs uppercase tracking-[0.1em] text-foreground"
        >
          <option value="operator">operator</option>
          <option value="administrator">administrator</option>
        </select>
        <Button type="submit" disabled={pending} className="h-8 text-xs">
          {pending ? "Creating…" : "Create account"}
        </Button>
      </form>
    </Panel>
  );
}