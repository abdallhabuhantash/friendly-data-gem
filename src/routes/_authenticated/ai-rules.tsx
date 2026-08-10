import { requireAdministrator } from "@/lib/require-admin";
import { createFileRoute } from "@tanstack/react-router";
import { Lock } from "lucide-react";
import { SeverityBadge } from "@/components/common/EventBadges";
import { Panel } from "@/components/common/Panel";
import { PageContainer } from "@/components/layout/PageContainer";
import { TopBar } from "@/components/layout/TopBar";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { useAiRules, useCameras, useSetRuleCameras, useUpdateRule } from "@/hooks/use-monitoring";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/_authenticated/ai-rules")({
  beforeLoad: requireAdministrator,
  head: () => ({
    meta: [
      { title: "AI Detection Rules — Vigilant Eye AI Smart Surveillance" },
      {
        name: "description",
        content:
          "Configure AI detection rules, confidence thresholds and trigger durations per camera.",
      },
    ],
  }),
  component: AiRulesPage,
});

function AiRulesPage() {
  const rules = useAiRules();
  const cameras = useCameras();
  const update = useUpdateRule();
  const setCameras = useSetRuleCameras();
  return (
    <>
      <TopBar
        title="AI Detection Rules"
        subtitle="Rule engine is modular — new use cases plug in without UI rewrites"
      />
      <PageContainer>
        <div className="grid gap-3 lg:grid-cols-2">
          {(rules.data ?? []).map((rule) => (
            <Panel
              key={rule.id}
              title={rule.name}
              subtitle={rule.description}
              actions={
                rule.available ? (
                  <Switch
                    checked={rule.enabled}
                    onCheckedChange={(value) =>
                      update.mutate({ id: rule.id, patch: { enabled: value } })
                    }
                  />
                ) : (
                  <span className="label-tech flex items-center gap-1 text-muted-foreground">
                    <Lock className="size-3" /> Planned
                  </span>
                )
              }
              bodyClassName="space-y-4 p-3"
              className={rule.available ? "" : "opacity-60"}
            >
              <div className="flex items-center gap-2">
                <SeverityBadge severity={rule.severity} />
                <span className="label-tech text-muted-foreground">
                  {rule.cameraIds.length} camera{rule.cameraIds.length === 1 ? "" : "s"} of{" "}
                  {cameras.data?.length ?? 0}
                </span>
              </div>
              <RuleSlider
                label="Trigger object confidence"
                value={Math.round(rule.confidenceThreshold * 100)}
                suffix="%"
                min={40}
                max={99}
                disabled={!rule.available}
                onCommit={(value) =>
                  update.mutate({ id: rule.id, patch: { confidenceThreshold: value / 100 } })
                }
              />
              <RuleSlider
                label="Person confidence"
                value={Math.round(rule.personConfidenceThreshold * 100)}
                suffix="%"
                min={40}
                max={99}
                disabled={!rule.available}
                onCommit={(value) =>
                  update.mutate({ id: rule.id, patch: { personConfidenceThreshold: value / 100 } })
                }
              />
              <RuleSlider
                label="Association confidence"
                value={Math.round(rule.associationConfidenceThreshold * 100)}
                suffix="%"
                min={40}
                max={99}
                disabled={!rule.available}
                onCommit={(value) =>
                  update.mutate({
                    id: rule.id,
                    patch: { associationConfidenceThreshold: value / 100 },
                  })
                }
              />
              <RuleSlider
                label="Minimum duration"
                value={Math.round(rule.minDurationSeconds * 10)}
                display={`${rule.minDurationSeconds.toFixed(1)}`}
                suffix="s"
                min={5}
                max={300}
                disabled={!rule.available}
                onCommit={(value) =>
                  update.mutate({ id: rule.id, patch: { minDurationSeconds: value / 10 } })
                }
              />
              <RuleSlider
                label="Minimum matching frames"
                value={rule.minMatchingFrames}
                suffix=""
                min={1}
                max={120}
                disabled={!rule.available}
                onCommit={(value) =>
                  update.mutate({ id: rule.id, patch: { minMatchingFrames: value } })
                }
              />
              <RuleSlider
                label="Cooldown"
                value={rule.cooldownSeconds}
                suffix="s"
                min={0}
                max={300}
                step={10}
                disabled={!rule.available}
                onCommit={(value) =>
                  update.mutate({ id: rule.id, patch: { cooldownSeconds: value } })
                }
              />
              <div className="grid grid-cols-2 gap-2">
                <ToggleRow
                  label="Save snapshot"
                  checked={rule.saveSnapshot}
                  disabled={!rule.available}
                  onChange={(value) =>
                    update.mutate({ id: rule.id, patch: { saveSnapshot: value } })
                  }
                />
                <ToggleRow
                  label="Sound alert"
                  checked={rule.soundNotification}
                  disabled={!rule.available}
                  onChange={(value) =>
                    update.mutate({ id: rule.id, patch: { soundNotification: value } })
                  }
                />
                <ToggleRow
                  label="Require person association"
                  checked={rule.requirePersonAssociation}
                  disabled={!rule.available}
                  onChange={(value) =>
                    update.mutate({ id: rule.id, patch: { requirePersonAssociation: value } })
                  }
                />
                <ToggleRow
                  label="Instant phone detection"
                  checked={rule.instantDetectionEnabled}
                  disabled={!rule.available}
                  onChange={(value) =>
                    update.mutate({ id: rule.id, patch: { instantDetectionEnabled: value } })
                  }
                />
              </div>
              <RuleSlider
                label="Instant confidence"
                value={Math.round(rule.instantConfidenceThreshold * 100)}
                suffix="%"
                min={40}
                max={99}
                disabled={!rule.available || !rule.instantDetectionEnabled}
                onCommit={(value) =>
                  update.mutate({
                    id: rule.id,
                    patch: { instantConfidenceThreshold: value / 100 },
                  })
                }
              />
              <p className="text-[11px] leading-relaxed text-muted-foreground">
                Instant detection keeps a very short visible phone appearance as a
                &quot;Mobile Phone Detected&quot; warning for human review, even if it lasted only
                one analysed frame. Because a single frame has no temporal corroboration, it
                requires this higher confidence and never claims cheating on its own.
              </p>
              <div className="space-y-1.5">
                <span className="label-tech text-muted-foreground">Assigned cameras</span>
                <div className="flex flex-wrap gap-1.5">
                  {(cameras.data ?? []).map((camera) => {
                    const active = rule.cameraIds.includes(camera.id);
                    return (
                      <button
                        key={camera.id}
                        type="button"
                        disabled={!rule.available || setCameras.isPending}
                        onClick={() =>
                          setCameras.mutate({
                            ruleId: rule.id,
                            cameraIds: active
                              ? rule.cameraIds.filter((id) => id !== camera.id)
                              : [...rule.cameraIds, camera.id],
                          })
                        }
                        className={cn(
                          "rounded-[3px] border px-2 py-1 font-mono text-[10px] uppercase tracking-[0.08em] transition-colors",
                          active
                            ? "border-primary/60 bg-primary/10 text-primary"
                            : "border-border/70 bg-surface-2/50 text-muted-foreground hover:text-foreground",
                        )}
                      >
                        CH{String(camera.channel).padStart(2, "0")} · {camera.name}
                      </button>
                    );
                  })}
                </div>
              </div>
            </Panel>
          ))}
        </div>
      </PageContainer>
    </>
  );
}

function RuleSlider({
  label,
  value,
  display,
  suffix,
  min,
  max,
  step = 1,
  disabled,
  onCommit,
}: {
  label: string;
  value: number;
  display?: string;
  suffix: string;
  min: number;
  max: number;
  step?: number;
  disabled: boolean;
  onCommit: (value: number) => void;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="label-tech text-muted-foreground">{label}</span>
        <span className="font-mono text-[12px] tabular-nums text-primary">
          {display ?? value}
          {suffix}
        </span>
      </div>
      <Slider
        value={[value]}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onValueCommit={(next) => onCommit(next[0] ?? value)}
      />
    </div>
  );
}

function ToggleRow({
  label,
  checked,
  disabled,
  onChange,
}: {
  label: string;
  checked: boolean;
  disabled: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between rounded-[4px] border border-border/70 bg-surface-2/50 px-2.5 py-2">
      <span className="label-tech text-muted-foreground">{label}</span>
      <Switch checked={checked} disabled={disabled} onCheckedChange={onChange} />
    </div>
  );
}
