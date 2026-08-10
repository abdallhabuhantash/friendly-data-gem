/**
 * Operational reporting boundaries are calculated in the application timezone
 * (Asia/Amman), never in UTC and never in the browser's local timezone.
 * Stored timestamps stay UTC; only presentation boundaries are converted.
 */
export const APP_TIMEZONE = "Asia/Amman";

const partsFormatter = new Intl.DateTimeFormat("en-US", {
  timeZone: APP_TIMEZONE,
  hour12: false,
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
});

interface ZonedParts {
  year: number;
  month: number;
  day: number;
  hour: number;
  minute: number;
  second: number;
}

function zonedParts(date: Date): ZonedParts {
  const parts = partsFormatter.formatToParts(date);
  const read = (type: Intl.DateTimeFormatPartTypes): number =>
    Number(parts.find((part) => part.type === type)?.value ?? "0");
  const hour = read("hour");
  return {
    year: read("year"),
    month: read("month"),
    day: read("day"),
    // Intl can emit hour 24 for midnight with hour12: false.
    hour: hour === 24 ? 0 : hour,
    minute: read("minute"),
    second: read("second"),
  };
}

/** Milliseconds the application timezone is ahead of UTC at the given instant. */
function zoneOffsetMs(date: Date): number {
  const parts = zonedParts(date);
  const asUtc = Date.UTC(
    parts.year,
    parts.month - 1,
    parts.day,
    parts.hour,
    parts.minute,
    parts.second,
  );
  return asUtc - Math.floor(date.getTime() / 1000) * 1000;
}

/** Instant of 00:00:00 Asia/Amman for the calendar day containing `date`. */
export function startOfZonedDay(date: Date = new Date()): Date {
  const parts = zonedParts(date);
  const naiveUtc = Date.UTC(parts.year, parts.month - 1, parts.day, 0, 0, 0);
  // First guess with the current offset, then correct across DST transitions.
  let guess = new Date(naiveUtc - zoneOffsetMs(date));
  const corrected = new Date(naiveUtc - zoneOffsetMs(guess));
  if (corrected.getTime() !== guess.getTime()) guess = corrected;
  return guess;
}

export function addDays(date: Date, days: number): Date {
  return new Date(date.getTime() + days * 86_400_000);
}

/** Stable YYYY-MM-DD key of the Asia/Amman calendar day for an instant. */
export function zonedDayKey(date: Date): string {
  const parts = zonedParts(date);
  return `${parts.year}-${String(parts.month).padStart(2, "0")}-${String(parts.day).padStart(2, "0")}`;
}

const weekdayFormatter = new Intl.DateTimeFormat("en-US", {
  timeZone: APP_TIMEZONE,
  weekday: "short",
});

export function zonedWeekdayLabel(date: Date): string {
  return weekdayFormatter.format(date);
}

const shortDateFormatter = new Intl.DateTimeFormat("en-US", {
  timeZone: APP_TIMEZONE,
  month: "short",
  day: "numeric",
});

/** Short "Aug 7" style label in the application timezone. */
export function zonedShortDateLabel(date: Date): string {
  return shortDateFormatter.format(date);
}
