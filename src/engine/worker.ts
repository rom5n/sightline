/// <reference lib="webworker" />
// Off-main-thread transport: WebSocket + msgpack. Numeric blobs ({$nd, shape, data})
// become typed arrays here; probability grids are interleaved into RGBA texel order so
// the main thread can upload them to WebGL without touching each value.
import { decode, encode } from "@msgpack/msgpack";

type Out = { kind: "msg"; msg: unknown } | { kind: "status"; state: "connecting" | "open" | "closed"; attempt: number };

let ws: WebSocket | null = null;
let url = "";
let attempt = 0;
let queue: Uint8Array[] = [];
let timer: number | undefined;

const post = (o: Out, transfer: Transferable[] = []) => (self as DedicatedWorkerGlobalScope).postMessage(o, transfer);

function toTyped(v: { $nd: string; shape: number[]; data: Uint8Array }, transfer: Transferable[]): unknown {
  const bytes = v.data;
  // copy into an aligned buffer we own (decoder views may share and be unaligned)
  const buf = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(buf).set(bytes);
  transfer.push(buf);
  const arr = v.$nd === "f4" ? new Float32Array(buf) : v.$nd === "i4" ? new Int32Array(buf) : new Uint8Array(buf);
  return { shape: v.shape, data: arr };
}

function walk(o: unknown, transfer: Transferable[]): unknown {
  if (Array.isArray(o)) return o.map((x) => walk(x, transfer));
  if (o && typeof o === "object" && !(o instanceof Uint8Array)) {
    const r = o as Record<string, unknown>;
    if (typeof r.$nd === "string" && r.data instanceof Uint8Array) return toTyped(r as never, transfer);
    const out: Record<string, unknown> = {};
    for (const k in r) out[k] = walk(r[k], transfer);
    return out;
  }
  return o;
}

/** (K,H,W) planar float32 -> ceil(K/4) RGBA textures, texel-interleaved. */
function handle(bytes: ArrayBuffer) {
  const raw = decode(new Uint8Array(bytes)) as Record<string, unknown>;
  const transfer: Transferable[] = [];
  const msg = walk(raw, transfer) as Record<string, unknown>;
  post({ kind: "msg", msg }, transfer);
}

function connect() {
  clearTimeout(timer);
  attempt++;
  post({ kind: "status", state: "connecting", attempt });
  try {
    ws = new WebSocket(url);
  } catch {
    schedule();
    return;
  }
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    attempt = 0;
    post({ kind: "status", state: "open", attempt });
    for (const q of queue) ws!.send(q);
    queue = [];
  };
  ws.onmessage = (e) => {
    if (e.data instanceof ArrayBuffer) handle(e.data);
  };
  ws.onclose = () => {
    post({ kind: "status", state: "closed", attempt });
    ws = null;
    schedule();
  };
  ws.onerror = () => ws?.close();
}

function schedule() {
  clearTimeout(timer);
  // engine may still be starting: retry fast at first, then back off
  const delay = Math.min(2000, 120 * Math.max(1, attempt));
  timer = setTimeout(connect, delay) as unknown as number;
}

self.onmessage = (e: MessageEvent) => {
  const d = e.data as { kind: "connect"; url: string } | { kind: "send"; msg: unknown };
  if (d.kind === "connect") {
    url = d.url;
    connect();
    return;
  }
  const bytes = encode(d.msg);
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(bytes);
  else queue.push(bytes);
};
