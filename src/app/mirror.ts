// Keeps pop-out windows in sync with the main window over a BroadcastChannel.
// The main window owns the engine connection; pop-outs mirror its state and send
// commands back through it.
import { engine } from "../engine/client";
import { RING, useSession, type SessionState } from "../engine/store";
import type { ClientMessage } from "../engine/protocol.gen";
import { useUi } from "./ui";
import { useSettings } from "./settings";

const CH = "sightline";
type Msg =
  | { kind: "hello" }
  | { kind: "snapshot"; s: Partial<SessionState>; ui: unknown; frames: unknown[] }
  | { kind: "patch"; s: Partial<SessionState> }
  | { kind: "ui"; ui: unknown }
  | { kind: "send"; msg: ClientMessage }
  | { kind: "theme"; theme: string; lang: string }
  | { kind: "popout-closed"; panel: string };

const COARSE: (keyof SessionState)[] = ["conn", "ready", "workerReady", "sklearn", "models", "presets", "model", "paramsByModel", "dataset", "status", "job", "history", "done", "explain", "probe", "speed", "scrub"];

export function startMirrorHost() {
  const bc = new BroadcastChannel(CH);
  const snapshot = (): Msg => {
    const s = useSession.getState();
    const part: Partial<SessionState> = {};
    for (const k of COARSE) (part as Record<string, unknown>)[k] = s[k];
    part.latest = s.latest;
    return { kind: "snapshot", s: part, ui: useUi.getState(), frames: s.frames.slice(-12) };
  };
  bc.onmessage = (e: MessageEvent<Msg>) => {
    const m = e.data;
    if (m.kind === "hello") bc.postMessage(snapshot());
    else if (m.kind === "send") engine.send(m.msg);
    else if (m.kind === "ui") useUi.setState(m.ui as never);
  };
  let pending: Partial<SessionState> = {};
  let raf = 0;
  useSession.subscribe((s, prev) => {
    for (const k of COARSE) if (s[k] !== prev[k]) (pending as Record<string, unknown>)[k] = s[k];
    if (s.latest !== prev.latest) pending.latest = s.latest;
    if (!raf) raf = requestAnimationFrame(() => {
      raf = 0;
      if (Object.keys(pending).length) bc.postMessage({ kind: "patch", s: pending } as Msg);
      pending = {};
    });
  });
  useUi.subscribe((u) => bc.postMessage({ kind: "ui", ui: u } as Msg));
  useSettings.subscribe((st, p) => {
    if (st.theme !== p.theme || st.lang !== p.lang) bc.postMessage({ kind: "theme", theme: st.theme, lang: st.lang } as Msg);
  });
}

export function startMirrorClient(panel: string, onTheme: (theme: string, lang: string) => void) {
  const bc = new BroadcastChannel(CH);
  let applyingUi = false;
  bc.onmessage = (e: MessageEvent<Msg>) => {
    const m = e.data;
    if (m.kind === "snapshot") {
      useSession.setState({ ...m.s, frames: m.frames as never });
      applyingUi = true;
      useUi.setState(m.ui as never);
      applyingUi = false;
    } else if (m.kind === "patch") {
      const s = useSession.getState();
      const patch: Partial<SessionState> = { ...m.s };
      if (m.s.latest && m.s.latest !== s.latest) {
        const f = m.s.latest;
        patch.frames = s.frames.length && s.frames[0].job !== f.job ? [f] : [...s.frames.slice(-(RING - 1)), f];
      }
      useSession.setState(patch);
    } else if (m.kind === "ui") {
      applyingUi = true;
      useUi.setState(m.ui as never);
      applyingUi = false;
    } else if (m.kind === "theme") onTheme(m.theme, m.lang);
  };
  useUi.subscribe((u) => {
    if (!applyingUi) bc.postMessage({ kind: "ui", ui: u } as Msg);
  });
  // engine commands from the pop-out go through the main window
  (engine as unknown as { send: (m: ClientMessage) => void }).send = (msg) => bc.postMessage({ kind: "send", msg } as Msg);
  bc.postMessage({ kind: "hello" } as Msg);
  addEventListener("beforeunload", () => bc.postMessage({ kind: "popout-closed", panel } as Msg));
  return () => bc.postMessage({ kind: "popout-closed", panel } as Msg);
}
