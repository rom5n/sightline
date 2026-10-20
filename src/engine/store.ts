import { create } from "zustand";
import { engine, type ConnState } from "./client";
import type { DatasetMsg, DoneMsg, ExplainMsg, FrameMsg, ModelSchema, ParamValue, PresetInfo, ProbeData, ReadyMsg, StatusMsg } from "./types";

export const RING = 500;
const KEEP_GRIDS = 100;
let frameSeq = 0;

export interface History {
  it: number[];
  loss: (number | null)[];
  lossTest: (number | null)[];
  mtr: (number | null)[];
  mte: (number | null)[];
}

const emptyHistory = (): History => ({ it: [], loss: [], lossTest: [], mtr: [], mte: [] });

export interface Toast { id: number; code: string; message: { en: string; ru: string } }

export interface SessionState {
  conn: ConnState;
  ready: boolean;
  workerReady: boolean;
  sklearn: string;
  models: Record<string, ModelSchema>;
  presets: PresetInfo[];
  model: string | null;
  paramsByModel: Record<string, Record<string, ParamValue>>;
  dataset: DatasetMsg | null;
  status: StatusMsg | null;
  job: number;
  /** frames for the current job (ring buffer, oldest first) */
  frames: FrameMsg[];
  latest: FrameMsg | null;
  /** frame being shown (null = live / latest) */
  scrub: number | null;
  history: History;
  done: DoneMsg | null;
  explain: ExplainMsg | null;
  probe: ProbeData | null;
  toasts: Toast[];
  speed: number;
  /** wall-clock ms at which the user last changed a parameter (for latency overlay) */
  paramChangedAt: number;
  paramLatency: number | null;
}

export const useSession = create<SessionState>(() => ({
  conn: "connecting",
  ready: false,
  workerReady: false,
  sklearn: "",
  models: {},
  presets: [],
  model: null,
  paramsByModel: {},
  dataset: null,
  status: null,
  job: 0,
  frames: [],
  latest: null,
  scrub: null,
  history: emptyHistory(),
  done: null,
  explain: null,
  probe: null,
  toasts: [],
  speed: 1,
  paramChangedAt: 0,
  paramLatency: null,
}));

const S = useSession;
let toastId = 1;

export function toast(code: string, message: { en: string; ru: string }) {
  const t = { id: toastId++, code, message };
  S.setState((s) => ({ toasts: [...s.toasts.slice(-3), t] }));
  setTimeout(() => S.setState((s) => ({ toasts: s.toasts.filter((x) => x.id !== t.id) })), 6000);
}

function resetTraining(job = S.getState().job) {
  S.setState({ frames: [], latest: null, scrub: null, history: emptyHistory(), done: null, explain: null, job });
}

export function initEngine() {
  engine.onConn((c) => S.setState({ conn: c, ...(c !== "open" ? { ready: false } : {}) }));
  engine.on("ready", (m) => {
    const r = m as unknown as ReadyMsg;
    const models: Record<string, ModelSchema> = {};
    for (const s of r.models) models[s.name] = s;
    S.setState({ ready: true, sklearn: r.sklearn, models, presets: r.presets as unknown as PresetInfo[], workerReady: !!r.worker });
    const st = S.getState();
    if (st.model) engine.send({ type: "set_model", model: st.model, params: currentParams() });
  });
  engine.on("dataset", (m) => {
    const d = m as unknown as DatasetMsg;
    const prev = S.getState().dataset;
    S.setState({ dataset: d });
    if (!prev || prev.version !== d.version || prev.model !== d.model) {
      // a new dataset or model invalidates training history; a data edit keeps it until new frames arrive
      if (!prev || prev.name !== d.name || prev.model !== d.model || prev.n_rows !== d.n_rows || d.kind !== prev.kind) resetTraining();
    }
  });
  engine.on("status", (m) => {
    const st = m as unknown as StatusMsg;
    const patch: Partial<SessionState> = { status: st };
    if (st.message === "worker_ready") patch.workerReady = true;
    if (st.state === "starting") patch.workerReady = false;
    if (st.reset) {
      resetTraining();
    }
    if ((st.state === "training" || st.state === "paused") && st.job != null && st.job !== S.getState().job) {
      resetTraining(st.job);
    }
    S.setState(patch);
  });
  engine.on("frame", (m) => {
    const f = m as unknown as FrameMsg;
    const s = S.getState();
    if (f.job !== s.job) {
      if (f.job < s.job) return;
      resetTraining(f.job);
    }
    const st = S.getState();
    const frames = st.frames.length >= RING ? [...st.frames.slice(1), f] : [...st.frames, f];
    // memory: the newest KEEP_GRIDS frames keep their heatmap grid, older ones only every 4th
    // (at most ~200 grids instead of 500; the scrubber shows the nearest earlier grid)
    const old = frames.length - 1 - KEEP_GRIDS;
    frameSeq++;
    if (old >= 0 && frames[old].viz.grid && (frameSeq - KEEP_GRIDS) % 4 !== 0) {
      frames[old] = { ...frames[old], viz: { ...frames[old].viz, grid: undefined } };
    }
    const h = st.history;
    const last = h.it.length ? h.it[h.it.length - 1] : -1;
    let history = h;
    if (f.iteration > last) {
      history = {
        it: [...h.it, f.iteration],
        loss: [...h.loss, f.loss ?? null],
        lossTest: [...h.lossTest, f.loss_test ?? null],
        mtr: [...h.mtr, f.metric_train ?? null],
        mte: [...h.mte, f.metric_test ?? null],
      };
    }
    const patch: Partial<SessionState> = { frames, latest: f, history };
    if (st.paramChangedAt && !st.paramLatency) patch.paramLatency = performance.now() - st.paramChangedAt;
    if (f.viz.probe) patch.probe = f.viz.probe;
    S.setState(patch);
  });
  engine.on("done", (m) => {
    const d = m as unknown as DoneMsg;
    if (d.job === S.getState().job) S.setState({ done: d });
  });
  engine.on("explain_steps", (m) => S.setState({ explain: m as unknown as ExplainMsg }));
  engine.on("probe", (m) => S.setState({ probe: (m as { data: ProbeData }).data }));
  engine.on("error", (m) => {
    const e = m as { code: string; message: { en: string; ru: string }; rid?: number | null };
    if (e.rid == null) toast(e.code, e.message);
  });
  void engine.start();
}

// ------------------------------------------------------------------ actions

export function currentParams(): Record<string, ParamValue> {
  const s = S.getState();
  if (!s.model) return {};
  const schema = s.models[s.model];
  const base: Record<string, ParamValue> = {};
  if (schema) for (const p of [...schema.main, ...schema.advanced]) base[p.name] = p.initial;
  return { ...base, ...(s.paramsByModel[s.model] ?? {}) };
}

export function selectModel(model: string) {
  const s = S.getState();
  if (s.model === model) return;
  S.setState({ model });
  resetTraining(s.job);
  engine.send({ type: "set_model", model, params: s.paramsByModel[model] ?? {} });
}

export function setParams(patch: Record<string, ParamValue>, retrain = true) {
  const s = S.getState();
  if (!s.model) return;
  const cur = s.paramsByModel[s.model] ?? {};
  S.setState({ paramsByModel: { ...s.paramsByModel, [s.model]: { ...cur, ...patch } }, paramChangedAt: performance.now(), paramLatency: null });
  engine.send({ type: "set_params", params: patch, retrain });
}

export function resetParams() {
  const s = S.getState();
  if (!s.model) return;
  const schema = s.models[s.model];
  const all: Record<string, ParamValue> = {};
  if (schema) for (const p of [...schema.main, ...schema.advanced]) all[p.name] = p.initial;
  S.setState({ paramsByModel: { ...s.paramsByModel, [s.model]: {} } });
  engine.send({ type: "set_params", params: all, retrain: !!S.getState().latest });
}

export const train = (fast = false) => engine.send({ type: "train", fast });
export const pause = () => engine.send({ type: "pause" });
export const resume = () => engine.send({ type: "resume" });
export const step = () => engine.send({ type: "step" });
export const resetModel = () => engine.send({ type: "reset" });
export const cancel = () => engine.send({ type: "cancel" });
export function setSpeed(speed: number) {
  S.setState({ speed });
  engine.send({ type: "set_speed", speed });
}
export const explain = (pointId: number) => engine.request({ type: "explain", point_id: pointId }).catch((e) => e);

export function loadPreset(preset: string, settings: Record<string, number> = {}) {
  return engine.request<DatasetMsg>({ type: "load_dataset", spec: { kind: "preset", preset, settings } });
}

export function frameAt(): FrameMsg | null {
  const s = S.getState();
  if (s.scrub != null && s.frames[s.scrub]) return s.frames[s.scrub];
  return s.latest;
}

export type TrainPhase = "nodata" | "incompatible" | "idle" | "training" | "paused" | "fitting" | "done" | "starting";

export function trainPhase(s: SessionState): TrainPhase {
  if (!s.dataset) return "nodata";
  if (!s.dataset.compat.ok) return "incompatible";
  const st = s.status?.state;
  if (st === "starting") return "starting";
  if (s.done && s.done.job === s.job && st !== "training" && st !== "paused" && st !== "fitting") return "done";
  if (st === "training" || st === "paused" || st === "fitting") return st;
  if (s.latest) return "done";
  return "idle";
}
