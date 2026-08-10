import { cn } from "@/lib/utils";
import type { DetectionOverlay } from "@/types";

export function BoundingBox({ detection }: { detection: DetectionOverlay }) {
  const alert = detection.alertState === "alert";
  const uncertain = detection.alertState === "uncertain";
  const phone = detection.className === "cell_phone";
  return (
    <div
      className={cn(
        "absolute border",
        uncertain
          ? "border-dashed border-warning"
          : alert
            ? "border-destructive"
            : phone
              ? "border-warning"
              : "border-primary",
        alert && "shadow-[0_0_12px_color-mix(in_oklab,var(--destructive)_55%,transparent)]",
      )}
      style={{
        left: `${detection.x}%`,
        top: `${detection.y}%`,
        width: `${detection.width}%`,
        height: `${detection.height}%`,
      }}
    >
      <div
        className={cn(
          "absolute -top-5 left-0 flex h-5 items-center whitespace-nowrap px-1.5 font-mono text-[9px] font-bold uppercase",
          uncertain
            ? "bg-warning text-warning-foreground"
            : alert
              ? "bg-destructive text-destructive-foreground"
              : phone
                ? "bg-warning text-warning-foreground"
                : "bg-primary text-primary-foreground",
        )}
      >
        {uncertain ? "Uncertain association" : detection.className.replace("_", " ")}{" "}
        {detection.trackingId ? `ID ${detection.trackingId}` : ""} ·{" "}
        {Math.round(detection.confidence * 100)}%
      </div>
      <span className="absolute -left-px -top-px size-2 border-l-2 border-t-2 border-current" />
      <span className="absolute -bottom-px -right-px size-2 border-b-2 border-r-2 border-current" />
    </div>
  );
}

const center = (detection: DetectionOverlay) => ({
  x: detection.x + detection.width / 2,
  y: detection.y + detection.height / 2,
});

export function DetectionOverlayLayer({
  detections,
  visible,
}: {
  detections: DetectionOverlay[];
  visible: boolean;
}) {
  if (!visible) return null;
  // Association connectors are derived from the detection geometry, never hard-coded.
  const links = detections
    .filter((detection) => detection.associatedPersonId)
    .map((detection) => {
      const person = detections.find((item) => item.objectId === detection.associatedPersonId);
      if (!person) return null;
      const from = center(detection);
      const to = center(person);
      return {
        id: `${detection.objectId}-${person.objectId}`,
        from,
        to,
        uncertain: detection.alertState === "uncertain",
      };
    })
    .filter((link): link is NonNullable<typeof link> => link !== null);
  return (
    <div className="pointer-events-none absolute inset-0 z-20">
      {detections.map((detection) => (
        <BoundingBox key={detection.objectId} detection={detection} />
      ))}
      <svg
        className="absolute inset-0 size-full"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        aria-hidden
      >
        {links.map((link) => (
          <line
            key={link.id}
            x1={link.from.x}
            y1={link.from.y}
            x2={link.to.x}
            y2={link.to.y}
            stroke="currentColor"
            strokeWidth="0.25"
            strokeDasharray="1 1"
            className={link.uncertain ? "text-warning/80" : "text-destructive/80"}
          />
        ))}
      </svg>
    </div>
  );
}
