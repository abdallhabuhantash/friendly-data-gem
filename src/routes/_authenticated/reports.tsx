import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Panel } from "@/components/common/Panel";
import { StatTile } from "@/components/common/StatTile";
import { PageContainer } from "@/components/layout/PageContainer";
import { TopBar } from "@/components/layout/TopBar";
import { Button } from "@/components/ui/button";
import { useReportSummary } from "@/hooks/use-monitoring";

export const Route = createFileRoute("/_authenticated/reports")({
  head: () => ({
    meta: [
      { title: "Reports — Vigilant Eye AI Smart Surveillance" },
      {
        name: "description",
        content: "Detection trends, per-camera event volume and operator confirmation rates.",
      },
    ],
  }),
  component: ReportsPage,
});

function ReportsPage() {
  const [range, setRange] = useState<"7d" | "30d">("7d");
  const report = useReportSummary(range);
  const peak = Math.max(1, ...(report.data?.timeline ?? []).map((point) => point.events));
  const peakCamera = Math.max(1, ...(report.data?.byCamera ?? []).map((row) => row.events));
  return (
    <>
      <TopBar title="Reports & Analytics" subtitle="Aggregated detection statistics" />
      <PageContainer>
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <StatTile label="Events in range" value={report.data?.totalEvents ?? 0} />
          <StatTile label="Confirmed" value={report.data?.confirmed ?? 0} tone="success" />
          <StatTile
            label="Confirmation rate"
            value={`${Math.round((report.data?.confirmationRate ?? 0) * 100)}%`}
            tone="primary"
          />
          <StatTile
            label="Avg review time"
            value={
              report.data?.averageReviewMinutes === null ||
              report.data?.averageReviewMinutes === undefined
                ? "—"
                : `${report.data.averageReviewMinutes.toFixed(1)}m`
            }
            tone="warning"
          />
        </div>
        <Panel
          title="Detection timeline"
          subtitle="Total AI events vs operator-confirmed events"
          actions={
            <div className="flex gap-1">
              {(["7d", "30d"] as const).map((option) => (
                <Button
                  key={option}
                  size="sm"
                  variant={range === option ? "default" : "outline"}
                  className="h-7 px-2 font-mono text-[11px]"
                  onClick={() => setRange(option)}
                >
                  {option}
                </Button>
              ))}
            </div>
          }
        >
          <div className="flex h-56 items-end gap-1.5">
            {(report.data?.timeline ?? []).map((point) => (
              <div key={point.label} className="flex flex-1 flex-col items-center gap-1.5">
                <div className="relative flex h-full w-full items-end justify-center">
                  <div
                    className="w-full rounded-t-[2px] bg-primary/25"
                    style={{ height: `${(point.events / peak) * 100}%` }}
                  >
                    <div
                      className="w-full rounded-t-[2px] bg-primary"
                      style={{
                        height: `${point.events ? (point.confirmed / point.events) * 100 : 0}%`,
                        marginTop: "auto",
                      }}
                    />
                  </div>
                </div>
                <span className="font-mono text-[10px] text-muted-foreground">{point.label}</span>
              </div>
            ))}
          </div>
        </Panel>
        <Panel title="Events by camera" bodyClassName="space-y-2.5 p-3">
          {(report.data?.byCamera ?? []).map((row) => (
            <div key={row.cameraName} className="space-y-1">
              <div className="flex items-center justify-between text-[12px]">
                <span className="text-foreground">{row.cameraName}</span>
                <span className="font-mono tabular-nums text-muted-foreground">{row.events}</span>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full bg-accent"
                  style={{ width: `${(row.events / peakCamera) * 100}%` }}
                />
              </div>
            </div>
          ))}
        </Panel>
      </PageContainer>
    </>
  );
}
