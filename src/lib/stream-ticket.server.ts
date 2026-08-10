import { createHmac, timingSafeEqual } from "crypto";

const TTL_MS = 5 * 60_000;

function secret(): string {
  const value = process.env["STREAM_TICKET_SECRET"];
  if (!value) throw new Error("STREAM_TICKET_SECRET is not configured");
  return value;
}

export function signStreamTicket(cameraId: string): string {
  const expires = Date.now() + TTL_MS;
  const payload = `${cameraId}.${expires}`;
  const signature = createHmac("sha256", secret()).update(payload).digest("hex");
  return `${expires}.${signature}`;
}

export function verifyStreamTicket(cameraId: string, ticket: string | null): boolean {
  if (!ticket) return false;
  const [expiresRaw, signature] = ticket.split(".");
  const expires = Number(expiresRaw);
  if (!expires || !signature || Date.now() > expires) return false;
  const expected = createHmac("sha256", secret()).update(`${cameraId}.${expires}`).digest("hex");
  const a = Buffer.from(signature);
  const b = Buffer.from(expected);
  return a.length === b.length && timingSafeEqual(a, b);
}
