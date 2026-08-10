import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface PanelProps {
  title?: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}

export function Panel({
  title,
  subtitle,
  actions,
  children,
  className,
  bodyClassName,
}: PanelProps) {
  return (
    <section className={cn("panel relative", className)}>
      <span className="pointer-events-none absolute -top-px left-3 h-px w-10 bg-primary/70" />
      {(title || actions) && (
        <header className="flex items-center justify-between gap-3 border-b border-border/70 px-3 py-2">
          <div>
            {title && <h2 className="label-tech text-foreground/80">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-muted-foreground">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      <div className={cn("p-3", bodyClassName)}>{children}</div>
    </section>
  );
}
