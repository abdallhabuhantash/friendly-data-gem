import { Link } from "@tanstack/react-router";
import {
  Activity,
  BarChart3,
  Cctv,
  LayoutDashboard,
  ScanEye,
  ClipboardList,
  Settings,
  ShieldAlert,
  UserCog,
  Users,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useAuth } from "@/hooks/use-auth";

interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
  adminOnly?: boolean;
}

const primaryNav: NavItem[] = [
  { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { to: "/monitoring", label: "Live Monitoring", icon: Activity },
  { to: "/events", label: "Events", icon: ShieldAlert },
  { to: "/cameras", label: "Cameras", icon: Cctv, adminOnly: true },
  { to: "/ai-rules", label: "AI Rules", icon: ScanEye, adminOnly: true },
  { to: "/exam-sessions", label: "Exam Sessions", icon: ClipboardList },
  { to: "/reports", label: "Reports", icon: BarChart3 },
];

const secondaryNav: NavItem[] = [
  { to: "/users", label: "Users", icon: Users, adminOnly: true },
  { to: "/settings", label: "Settings", icon: Settings, adminOnly: true },
  { to: "/profile", label: "Profile", icon: UserCog },
];

const linkClass =
  "flex items-center gap-2.5 rounded-[3px] border border-transparent px-2.5 py-2 text-[13px] text-sidebar-foreground/80 transition-colors hover:border-sidebar-border hover:bg-sidebar-accent hover:text-sidebar-accent-foreground";

function NavSection({ items, title }: { items: NavItem[]; title: string }) {
  const { isAdministrator } = useAuth();
  const visible = items.filter((item) => !item.adminOnly || isAdministrator);
  if (visible.length === 0) return null;

  return (
    <div className="px-2 py-3">
      <p className="label-tech px-1 pb-2">{title}</p>
      <nav className="space-y-0.5">
        {visible.map((item) => (
          <Link
            key={item.to}
            to={item.to}
            className={linkClass}
            activeProps={{
              className:
                "border-primary/40 bg-primary/10 text-primary shadow-[inset_2px_0_0_0_var(--primary)]",
            }}
          >
            <item.icon className="size-4 shrink-0" />
            <span className="truncate">{item.label}</span>
          </Link>
        ))}
      </nav>
    </div>
  );
}

export function AppSidebar() {
  const { isAdministrator } = useAuth();
  return (
    <aside className="flex w-56 shrink-0 flex-col border-r border-sidebar-border bg-sidebar">
      <div className="flex items-center gap-2 border-b border-sidebar-border px-3 py-3">
        <div className="grid size-8 place-items-center rounded-[3px] border border-primary/50 bg-primary/10 text-primary">
          <ScanEye className="size-4" />
        </div>
        <div className="leading-tight">
          <p className="font-mono text-[13px] tracking-[0.14em] text-foreground">VIGILANT EYE</p>
          <p className="label-tech">AI Smart Surveillance</p>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto divide-y divide-sidebar-border/60">
        <NavSection title="Operations" items={primaryNav} />
        {isAdministrator && <NavSection title="Administration" items={secondaryNav} />}
      </div>
      <div className="border-t border-sidebar-border px-3 py-2">
        <p className="label-tech">Build</p>
        <p className="font-mono text-[11px] text-muted-foreground">prototype 0.9.3</p>
      </div>
    </aside>
  );
}
