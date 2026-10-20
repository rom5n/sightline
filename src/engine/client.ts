// Main-thread side of the engine connection. All parsing happens in worker.ts.
import { engineInfo } from "../app/tauri";
import type { ClientMessage } from "./protocol.gen";

type Listener = (msg: Record<string, unknown>) => void;
export type ConnState = "connecting" | "open" | "closed";

class EngineClient {
  private worker: Worker | null = null;
  private listeners = new Map<string, Set<Listener>>();
  private connListeners = new Set<(s: ConnState) => void>();
  private rid = 1;
  private pending = new Map<number, { resolve: (m: Record<string, unknown>) => void; reject: (e: unknown) => void }>();
  conn: ConnState = "connecting";
  /** time (ms) each message arrived, for the perf overlay */
  lastFrameAt = 0;
  framesPerSec = 0;
  private frameTimes: number[] = [];

  async start() {
    if (this.worker) return;
    this.worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (e) => this.onWorker(e.data);
    const { port, token } = await engineInfo();
    this.worker.postMessage({ kind: "connect", url: `ws://127.0.0.1:${port}/ws?token=${encodeURIComponent(token)}` });
  }

  private onWorker(d: { kind: "msg"; msg: Record<string, unknown> } | { kind: "status"; state: ConnState }) {
    if (d.kind === "status") {
      this.conn = d.state;
      this.connListeners.forEach((f) => f(d.state));
      return;
    }
    const msg = d.msg;
    const type = String(msg.type);
    if (type === "frame") {
      const now = performance.now();
      this.lastFrameAt = now;
      (window as unknown as { __sightlineLastFrameAt?: number }).__sightlineLastFrameAt = now;
      this.frameTimes.push(now);
      while (this.frameTimes.length && now - this.frameTimes[0] > 1000) this.frameTimes.shift();
      this.framesPerSec = this.frameTimes.length;
    }
    const rid = msg.rid as number | null | undefined;
    if (rid != null && this.pending.has(rid)) {
      const p = this.pending.get(rid)!;
      this.pending.delete(rid);
      if (type === "error") p.reject(msg);
      else p.resolve(msg);
    }
    this.listeners.get(type)?.forEach((f) => f(msg));
    this.listeners.get("*")?.forEach((f) => f(msg));
  }

  send(msg: ClientMessage) {
    this.worker?.postMessage({ kind: "send", msg });
  }

  request<T = Record<string, unknown>>(msg: ClientMessage & { rid?: number | null }, timeoutMs = 30000): Promise<T> {
    const rid = this.rid++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(rid, { resolve: resolve as never, reject });
      this.send({ ...msg, rid } as ClientMessage);
      setTimeout(() => {
        if (this.pending.delete(rid)) reject({ type: "error", code: "timeout" });
      }, timeoutMs);
    });
  }

  on(type: string, fn: Listener): () => void {
    let s = this.listeners.get(type);
    if (!s) this.listeners.set(type, (s = new Set()));
    s.add(fn);
    return () => s!.delete(fn);
  }

  onConn(fn: (s: ConnState) => void): () => void {
    this.connListeners.add(fn);
    return () => this.connListeners.delete(fn);
  }
}

export const engine = new EngineClient();
