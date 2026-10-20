// React wrapper around PlotGL: pan/zoom with inertia, hover tooltips, point editing,
// frame interpolation (heat morph, gliding borders and centres), ghost border.
import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { engine } from "../engine/client";
import { frameAt, useSession } from "../engine/store";
import type { DatasetMsg, FrameMsg } from "../engine/types";
import { useUi } from "../app/ui";
import { useNav } from "../app/nav";
import { fmt } from "../ui/controls";
import { cssVar, parseColor, reducedMotion, request, subscribe } from "./loop";
import { borderSegments, isoSegments } from "./marching";
import { FLAG, PlotGL, type LineStyle, type View } from "./PlotGL";

export interface PlotLabel { x: number; y: number; text: string; cls?: string; dx?: number; dy?: number }
export interface PlotExtra { name: string; segs: Float32Array; style: LineStyle; above?: boolean }

export interface PlotProps {
  variant: "main" | "inset" | "decide";
  editable?: boolean;
  tooltip?: boolean;
  onPick?: (id: number) => void;
  highlightId?: number | null;
  dim?: boolean;
  highlightSet?: Set<number> | null;
  ringSet?: Set<number> | null;
  extra?: () => PlotExtra[];
  labels?: () => PlotLabel[];
  axes?: boolean;
  heat?: boolean;
  border?: boolean;
  ghost?: boolean;
  centers?: boolean;
  paths?: boolean;
  pointsOnly?: boolean;
  children?: ReactNode;
  testId?: string;
}

const nice = (span: number, n: number) => {
  const raw = span / n;
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
};
function ticks(lo: number, hi: number, n: number): number[] {
  const s = nice(hi - lo, n);
  const out: number[] = [];
  for (let v = Math.ceil(lo / s) * s; v <= hi + 1e-9; v += s) out.push(Math.abs(v) < s * 1e-6 ? 0 : v);
  return out;
}
const tickLabel = (v: number, step: number) => {
  const d = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  return (Math.abs(v) >= 1e4 ? v.toExponential(0) : v.toFixed(Math.min(4, d))).replace("-", "−");
};

/** The nearest grid at or before `f` in the frame ring (old frames are thinned, see store). */
let mixed = new Float32Array(0); // interpolated grid for the morphing border, reused every frame

function gridBefore(f: FrameMsg) {
  const fr = useSession.getState().frames;
  for (let i = fr.indexOf(f); i >= 0; i--) if (fr[i].viz.grid) return fr[i].viz.grid;
  return undefined;
}

function ptSize(n: number) {
  return Math.max(3, Math.min(9, 9 - 2.2 * Math.log2(Math.max(1, n / 400))));
}

function circleSegs(cx: number, cy: number, rx: number, ry: number, n = 64): Float32Array {
  const s = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    const a0 = (i / n) * Math.PI * 2;
    const a1 = ((i + 1) / n) * Math.PI * 2;
    s.set([cx + Math.cos(a0) * rx, cy + Math.sin(a0) * ry, cx + Math.cos(a1) * rx, cy + Math.sin(a1) * ry], i * 4);
  }
  return s;
}
export { circleSegs };

export function homeView(ds: DatasetMsg | null, f: FrameMsg | null): View {
  if (f?.viz.kind === "projection" && f.viz.bounds) {
    const [x0, x1, y0, y1] = f.viz.bounds;
    return { x0, x1, y0, y1 };
  }
  const b = ds?.plane?.bounds ?? [-1, 1, -1, 1];
  return { x0: b[0], x1: b[1], y0: b[2], y1: b[3] };
}

/** Fit the view to the canvas aspect so circles stay round only when axes share units (raw plane). */
function fitAspect(v: View, w: number, h: number, equal: boolean): View {
  if (!equal || w < 2 || h < 2) return v;
  const sx = (v.x1 - v.x0) / w;
  const sy = (v.y1 - v.y0) / h;
  const s = Math.max(sx, sy);
  const cx = (v.x0 + v.x1) / 2;
  const cy = (v.y0 + v.y1) / 2;
  return { x0: cx - (s * w) / 2, x1: cx + (s * w) / 2, y0: cy - (s * h) / 2, y1: cy + (s * h) / 2 };
}

export function PlotView(props: PlotProps) {
  const { t } = useTranslation();
  const wrap = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ticksX = useRef<HTMLDivElement>(null);
  const ticksY = useRef<HTMLDivElement>(null);
  const labelsRef = useRef<HTMLDivElement>(null);
  const centersRef = useRef<HTMLDivElement>(null);
  const ringRef = useRef<HTMLDivElement>(null);
  const propsRef = useRef(props);
  propsRef.current = props;
  const kick = useRef<(() => void) | null>(null);
  const [glError, setGlError] = useState(false);
  const [tip, setTip] = useState<{ id: number; x: number; y: number } | null>(null);
  const dataset = useSession((s) => s.dataset);

  useLayoutEffect(() => {
    const canvas = canvasRef.current!;
    const box = wrap.current!;
    let gl: PlotGL;
    try {
      gl = new PlotGL(canvas);
    } catch {
      setGlError(true);
      return;
    }
    // ------------------------------------------------------------ state
    let cssW = 1, cssH = 1;
    let view: View = { x0: -1, x1: 1, y0: -1, y1: 1 };
    let target: View = view;
    let home: View = view;
    let vel = { x: 0, y: 0 };
    let interacting = false;
    let lastInteract = 0;
    let sentView: string | null = null;
    let ds: DatasetMsg | null = null;
    let idIndex = new Map<number, number>();
    let xy = new Float32Array(0);
    let cls = new Float32Array(0);
    let flagsBase = new Uint8Array(0);
    let frame: FrameMsg | null = null;
    let mixStart = 0;
    let mixDur = 120;
    let lastFrameAt = 0;
    let ptsStart = 0;
    let morphStart = 0;
    let ghostSegs: Float32Array | null = null;
    let centersPrev: Float32Array | null = null;
    let centersNext: Float32Array | null = null;
    let centersStart = 0;
    let centersDur = 300;
    let hover: number | null = null;
    let drag: { kind: "pan" | "point"; id?: number; x: number; y: number; moved: boolean; wx: number; wy: number } | null = null;
    let lastMove = { x: 0, y: 0, t: 0 };
    let moveTimer: number | undefined;
    let addTimer: number | undefined; // a click adds a point unless it turns into a double-click
    let dirty = true;
    let tickKey = "";
    let equal = false;
    let flagsDirty = false;

    const colors = () => {
      gl.refreshColors(box);
      const gc = parseColor(cssVar("--gl", box));
      const ink = parseColor(cssVar("--ink", box));
      const ink2 = parseColor(cssVar("--ink2", box));
      const acc = parseColor(cssVar("--acc-t", box));
      return { gc, ink, ink2, acc };
    };
    let C = colors();

    const toPx = (x: number, y: number) => [((x - view.x0) / (view.x1 - view.x0)) * cssW, (1 - (y - view.y0) / (view.y1 - view.y0)) * cssH];
    const toWorld = (px: number, py: number) => [view.x0 + (px / cssW) * (view.x1 - view.x0), view.y1 - (py / cssH) * (view.y1 - view.y0)];

    // ------------------------------------------------------------ data -> GL
    const rebuildPoints = (newDs: DatasetMsg, anim: "fly" | "move" | "none") => {
      const n = newDs.ids.data.length;
      const oldIndex = idIndex;
      const oldXy = xy;
      idIndex = new Map();
      for (let i = 0; i < n; i++) idIndex.set(newDs.ids.data[i], i);
      xy = new Float32Array(newDs.xy.data);
      cls = new Float32Array(n);
      const isClass = newDs.label_kind === "class" && newDs.task === "classification";
      for (let i = 0; i < n; i++) cls[i] = isClass ? newDs.label.data[i] : -1;
      flagsBase = new Uint8Array(n);
      for (let i = 0; i < n; i++) if (newDs.test.data[i]) flagsBase[i] |= FLAG.test;
      const data = new Float32Array(n * 8);
      for (let i = 0; i < n; i++) {
        const id = newDs.ids.data[i];
        const j = oldIndex.get(id);
        const px = anim === "move" && j != null ? oldXy[j * 2] : xy[i * 2];
        const py = anim === "move" && j != null ? oldXy[j * 2 + 1] : xy[i * 2 + 1];
        data.set([xy[i * 2], xy[i * 2 + 1], px, py, cls[i], cls[i], flagsBase[i], (((id * 2654435761) >>> 0) % 1000) / 1000 * 0.15], i * 8);
      }
      gl.pointSize = ptSize(n) * (propsRef.current.variant === "inset" ? 0.55 : 1);
      gl.singleColor = !isClass && newDs.task !== "none";
      gl.setPointData(data, n);
      gl.center = [(home.x0 + home.x1) / 2, (home.y0 + home.y1) / 2];
      gl.ptsAnim = reducedMotion() || anim === "none" ? 0 : anim === "fly" ? 1 : 2;
      gl.ptsDur = anim === "fly" ? 0.25 : 0.35;
      ptsStart = performance.now();
      applyFrameToPoints(frame, false);
    };

    /** Labels / positions / flags that come from the model (clusters, PCA, support vectors). */
    const applyFrameToPoints = (f: FrameMsg | null, animate: boolean) => {
      const d = gl.pointData;
      const n = gl.nPts;
      if (!n || !ds) return;
      const v = f?.viz;
      let changedCls = false;
      let moved = false;
      const sup = v?.support ? new Set(v.support.data) : null;
      const P = propsRef.current;
      for (let i = 0; i < n; i++) {
        let fl = flagsBase[i];
        const id = ds.ids.data[i];
        if (sup?.has(id)) fl |= FLAG.support;
        if (P.highlightSet?.has(id)) fl |= FLAG.outlier;
        if (P.ringSet?.has(id)) fl |= FLAG.support;
        if (P.dim && P.highlightId !== id) fl |= FLAG.dim;
        if (P.highlightId === id) fl |= FLAG.selected;
        d[i * 8 + 6] = fl;
      }
      if (v && (v.kind === "clusters" || v.kind === "dbscan") && v.labels && v.labels.data.length === n) {
        for (let i = 0; i < n; i++) {
          const c = v.labels.data[i];
          const cur = d[i * 8 + 4];
          if (cur !== c) {
            d[i * 8 + 5] = animate ? cur : c;
            d[i * 8 + 4] = c;
            changedCls = true;
          }
          if (v.kind === "dbscan" && c < 0) d[i * 8 + 6] = d[i * 8 + 6] | FLAG.hollow;
        }
        gl.singleColor = false;
      } else if (!v || (v.kind !== "clusters" && v.kind !== "dbscan")) {
        for (let i = 0; i < n; i++) if (d[i * 8 + 4] !== cls[i]) { d[i * 8 + 4] = cls[i]; d[i * 8 + 5] = cls[i]; }
      }
      if (v?.kind === "projection" && v.xy && v.xy.data.length === n * 2) {
        for (let i = 0; i < n; i++) {
          const nx = v.xy.data[i * 2], ny = v.xy.data[i * 2 + 1];
          if (d[i * 8] !== nx || d[i * 8 + 1] !== ny) {
            d[i * 8 + 2] = d[i * 8];
            d[i * 8 + 3] = d[i * 8 + 1];
            d[i * 8] = nx;
            d[i * 8 + 1] = ny;
            moved = true;
          }
        }
      }
      if (changedCls) {
        morphStart = performance.now();
        gl.morph = animate && !reducedMotion() ? 0 : 1;
      }
      if (moved) {
        gl.ptsAnim = reducedMotion() ? 0 : 2;
        gl.ptsDur = 0.4;
        ptsStart = performance.now();
      }
      gl.uploadPoints();
    };

    const applyFrame = (f: FrameMsg | null, live: boolean) => {
      const prev = frame;
      frame = f;
      const v = f?.viz;
      const now = performance.now();
      // older frames in the ring keep only every 4th grid; scrubbing shows the nearest earlier one
      const g = v?.grid ?? (f && !live ? gridBefore(f) : undefined);
      if (v && g) {
        gl.heatMode = v.kind === "value" ? "value" : v.kind === "clusters" ? "clusters" : "proba";
        if (v.range) gl.valueRange = v.range;
        const morph = live && !!prev && prev.job === f!.job && !reducedMotion();
        gl.setHeat({ bounds: g.bounds, w: g.w, h: g.h, k: g.k, planar: g.data.data }, morph);
        if (lastFrameAt) mixDur = Math.max(60, Math.min(260, now - lastFrameAt));
        lastFrameAt = now;
        mixStart = now;
      } else {
        gl.setHeat(null, false);
      }
      // ghost border: the grid from 10 frames back
      ghostSegs = null;
      if (propsRef.current.ghost && v?.kind === "proba" && f) {
        const fr = useSession.getState().frames;
        const idx = fr.indexOf(f);
        const back = idx >= 0 ? fr[idx - 10] : fr[fr.length - 11];
        const g = back ? back.viz.grid ?? gridBefore(back) : undefined;
        if (g && g.k === v.grid?.k) ghostSegs = borderSegments(g.data.data, g.k, g.w, g.h, g.bounds);
      }
      // kmeans centres glide
      if (v?.centers) {
        const c = v.centers.data;
        centersPrev = centersNext && centersNext.length === c.length ? currentCenters() : new Float32Array(c);
        centersNext = new Float32Array(c);
        centersStart = now;
        centersDur = reducedMotion() ? 0 : Math.max(150, Math.min(600, mixDur * 2));
      } else {
        centersPrev = centersNext = null;
      }
      if (v?.kind === "projection" && v.bounds) {
        const hv = homeView(ds, f);
        if (Math.abs(hv.x0 - home.x0) + Math.abs(hv.x1 - home.x1) > 1e-6) {
          home = fitAspect(hv, cssW, cssH, equal);
          target = home;
        }
      }
      applyFrameToPoints(f, live);
      dirty = true;
      request(loop);
    };

    const currentCenters = (): Float32Array => {
      if (!centersNext) return new Float32Array(0);
      if (!centersPrev || centersDur === 0) return centersNext;
      const k = Math.min(1, (performance.now() - centersStart) / centersDur);
      const e = 1 - Math.pow(1 - k, 3);
      const out = new Float32Array(centersNext.length);
      for (let i = 0; i < out.length; i++) out[i] = centersPrev[i] + (centersNext[i] - centersPrev[i]) * e;
      return out;
    };

    const setDataset = (d: DatasetMsg | null, prevDs: DatasetMsg | null) => {
      ds = d;
      if (!d || !d.plane) {
        gl.setPointData(new Float32Array(0), 0);
        gl.setHeat(null, false);
        dirty = true;
        request(loop);
        return;
      }
      equal = false; // the design stretches the plane to the panel, like the artboards
      const sameShape = prevDs && prevDs.plane && prevDs.name === d.name && prevDs.model === d.model;
      home = fitAspect(homeView(d, frame), cssW, cssH, equal);
      if (!sameShape) {
        view = target = home;
        frame = null;
        gl.setHeat(null, false);
      }
      // same data with edits: keep the user's view so points don't jump under the cursor
      const anim = !prevDs || !prevDs.plane || prevDs.model !== d.model ? "fly" : prevDs.preset === d.preset && prevDs.name === d.name ? "move" : "fly";
      rebuildPoints(d, anim);
      dirty = true;
      request(loop);
    };

    // ------------------------------------------------------------ render loop
    const loop = (now: number, dt: number): boolean => {
      let anim = false;
      if (flagsDirty) {
        flagsDirty = false;
        applyFrameToPoints(frame, false);
      }
      // view easing + inertia
      if (!interacting && (Math.abs(vel.x) > 1e-6 || Math.abs(vel.y) > 1e-6)) {
        const dxw = vel.x * dt, dyw = vel.y * dt;
        target = { x0: target.x0 + dxw, x1: target.x1 + dxw, y0: target.y0 + dyw, y1: target.y1 + dyw };
        const decay = Math.exp(-dt * 6);
        vel = { x: vel.x * decay, y: vel.y * decay };
        if (Math.hypot(vel.x / (view.x1 - view.x0), vel.y / (view.y1 - view.y0)) < 0.02) vel = { x: 0, y: 0 };
        anim = true;
      }
      const k = reducedMotion() ? 1 : 1 - Math.exp(-dt * 16);
      const nv = {
        x0: view.x0 + (target.x0 - view.x0) * k, x1: view.x1 + (target.x1 - view.x1) * k,
        y0: view.y0 + (target.y0 - view.y0) * k, y1: view.y1 + (target.y1 - view.y1) * k,
      };
      const span = view.x1 - view.x0;
      if (Math.abs(nv.x0 - target.x0) + Math.abs(nv.x1 - target.x1) + Math.abs(nv.y0 - target.y0) + Math.abs(nv.y1 - target.y1) > span * 1e-4) anim = true;
      else Object.assign(nv, target);
      const viewChanged = nv.x0 !== view.x0 || nv.x1 !== view.x1 || nv.y0 !== view.y0 || nv.y1 !== view.y1;
      view = nv;
      gl.view = view;

      // heat morph
      const mix = gl.mix < 1 ? Math.min(1, (now - mixStart) / mixDur) : 1;
      gl.mix = mix;
      if (mix < 1) anim = true;
      // points
      if (gl.ptsAnim) {
        gl.ptsT = (now - ptsStart) / 1000;
        if (gl.ptsT > gl.ptsDur + 0.16) gl.ptsAnim = 0;
        else anim = true;
      }
      if (gl.morph < 1) {
        gl.morph = Math.min(1, (now - morphStart) / 320);
        anim = true;
      }
      if (centersNext && centersPrev && now - centersStart < centersDur) anim = true;

      const P = propsRef.current;
      const ui = useUi.getState();
      const v = frame?.viz;
      const showHeat = P.heat !== false && !P.pointsOnly && !!gl.heat.b;
      gl.heatAlpha = ui.heatLayer === "border" && P.variant === "main" && v?.kind === "proba" ? 0 : 1;
      if (v?.kind === "clusters" && P.variant === "main" && !ui.kmLayers.includes("regions")) gl.heatAlpha = 0;

      // borders (marching squares on the interpolated grid)
      const below: string[] = ["grid"];
      const above: string[] = [];
      const { a, b } = gl.heat;
      if (P.border !== false && !P.pointsOnly && b && (v?.kind === "proba" || v?.kind === "value")) {
        let probs = b.planar;
        if (a && mix < 1 && a.w === b.w && a.h === b.h && a.k === b.k && a.bounds.every((x, i) => x === b.bounds[i])) {
          if (mixed.length !== b.planar.length) mixed = new Float32Array(b.planar.length);
          probs = mixed;
          for (let i = 0; i < probs.length; i++) probs[i] = a.planar[i] + (b.planar[i] - a.planar[i]) * mix;
        }
        if (v.kind === "proba") {
          gl.setLines("border", borderSegments(probs, b.k, b.w, b.h, b.bounds), { color: C.ink, width: 2.5 }, 2);
        } else if (v.range) {
          const [lo, hi] = v.range;
          const lv = [1, 2, 3, 4].map((i) => lo + ((hi - lo) * i) / 5);
          gl.setLines("border", isoSegments(probs, b.w, b.h, b.bounds, lv), { color: C.ink2, width: 1.2, alpha: 0.7 }, 2);
        }
        above.push("border");
      } else gl.setLines("border", null, { color: C.ink, width: 2 });
      if (P.ghost && ghostSegs && !P.pointsOnly) {
        gl.setLines("ghost", ghostSegs, { color: C.ink2, width: 1.5, dash: 8 }, 1);
        above.unshift("ghost");
      }
      // regression curve
      if (v?.kind === "curve" && v.x && v.y) {
        const xs = v.x.data, ys = v.y.data;
        const s = new Float32Array((xs.length - 1) * 4);
        for (let i = 0; i < xs.length - 1; i++) s.set([xs[i], ys[i], xs[i + 1], ys[i + 1]], i * 4);
        gl.setLines("curve", s, { color: C.acc, width: 2.5 }, 2);
        above.push("curve");
      } else gl.setLines("curve", null, { color: C.acc, width: 2 });
      // kmeans paths ("draws itself" up to the gliding centre)
      const cc = currentCenters();
      if (v?.path && (P.paths ?? true) && !(P.variant === "main" && !ui.kmLayers.includes("paths")) && !P.pointsOnly) {
        const [iters, kk] = v.path.shape;
        const pd = v.path.data;
        const segs: number[] = [];
        for (let c = 0; c < kk; c++) {
          for (let it = 0; it < iters - 2; it++) {
            const o = (it * kk + c) * 2, o2 = ((it + 1) * kk + c) * 2;
            segs.push(pd[o], pd[o + 1], pd[o2], pd[o2 + 1]);
          }
          if (iters >= 2 && cc.length) {
            const o = ((iters - 2) * kk + c) * 2;
            segs.push(pd[o], pd[o + 1], cc[c * 2], cc[c * 2 + 1]);
          }
        }
        gl.setLines("paths", new Float32Array(segs), { color: C.ink2, width: 1.5, dash: 8 }, 1);
        above.push("paths");
      } else gl.setLines("paths", null, { color: C.ink2, width: 1 });
      // PCA: how the original features point in the projected plane
      let arrowLabs: PlotLabel[] = [];
      if (v?.kind === "projection" && v.arrows?.length) {
        const segs: number[] = [];
        for (const a of v.arrows.slice(0, 12)) segs.push(0, 0, a.dx, a.dy);
        gl.setLines("arrows", new Float32Array(segs), { color: C.acc, width: 2 }, 2);
        above.push("arrows");
        arrowLabs = v.arrows.slice(0, 12).map((a) => ({ x: a.dx * 1.08, y: a.dy * 1.08, text: a.name, cls: "plab plain" }));
      } else gl.setLines("arrows", null, { color: C.acc, width: 2 });
      // gridlines at tick positions
      const xt = ticks(view.x0, view.x1, Math.max(3, Math.round(cssW / 130)));
      const yt = ticks(view.y0, view.y1, Math.max(3, Math.round(cssH / 90)));
      if (viewChanged || dirty) {
        const g: number[] = [];
        for (const x of xt) g.push(x, view.y0 - 1e3 * (view.y1 - view.y0), x, view.y1 + 1e3 * (view.y1 - view.y0));
        for (const y of yt) g.push(view.x0 - 1e3 * (view.x1 - view.x0), y, view.x1 + 1e3 * (view.x1 - view.x0), y);
        gl.setLines("grid", new Float32Array(g), { color: C.gc, width: 1 }, 0);
      }
      // extra layers from the caller (decide visuals, tree regions)
      for (const e of P.extra?.() ?? []) {
        gl.setLines("x:" + e.name, e.segs, e.style, 3);
        (e.above === false ? below : above).push("x:" + e.name);
      }
      gl.render({ heat: showHeat, points: true, linesBelow: below, linesAbove: above });

      // ---------------------------------------------------------- DOM overlays
      if (P.axes !== false && (viewChanged || dirty)) {
        const key = `${xt.join()}|${yt.join()}|${cssW}|${cssH}|${view.x0}|${view.y0}`;
        if (key !== tickKey) {
          tickKey = key;
          placeTicks(ticksX.current, xt, true);
          placeTicks(ticksY.current, yt, false);
        }
      }
      // kmeans centres
      const ce = centersRef.current;
      if (ce) {
        const n = cc.length / 2;
        while (ce.children.length < n) {
          const d = document.createElement("div");
          d.className = "ctr";
          ce.appendChild(d);
        }
        for (let i = 0; i < ce.children.length; i++) {
          const el = ce.children[i] as HTMLDivElement;
          if (i >= n || P.centers === false) { el.style.display = "none"; continue; }
          const [px, py] = toPx(cc[i * 2], cc[i * 2 + 1]);
          el.style.display = "flex";
          el.textContent = String(i + 1);
          el.style.transform = `translate(${px}px, ${py}px) translate(-50%, -50%)`;
        }
      }
      // labels
      const lr = labelsRef.current;
      if (lr) {
        const labs = [...(P.labels?.() ?? []), ...arrowLabs];
        while (lr.children.length < labs.length) lr.appendChild(document.createElement("span"));
        for (let i = 0; i < lr.children.length; i++) {
          const el = lr.children[i] as HTMLSpanElement;
          const l = labs[i];
          if (!l) { el.style.display = "none"; continue; }
          const [px, py] = toPx(l.x, l.y);
          el.style.display = "";
          el.className = l.cls ?? "plab";
          if (el.textContent !== l.text) el.textContent = l.text;
          el.style.transform = `translate(${px + (l.dx ?? 0)}px, ${py + (l.dy ?? 0)}px) translate(-50%, -50%)`;
        }
      }
      // hover / highlight ring
      const ring = ringRef.current;
      if (ring) {
        const id = P.highlightId ?? hover;
        const i = id != null ? idIndex.get(id) : undefined;
        if (i != null && i < gl.nPts) {
          const d = gl.pointData;
          const [px, py] = toPx(d[i * 8], d[i * 8 + 1]);
          ring.style.display = "block";
          ring.style.transform = `translate(${px}px, ${py}px) translate(-50%, -50%)`;
        } else ring.style.display = "none";
      }

      // ask the engine for a sharper grid once the view settles
      if (!anim && !interacting && P.variant === "main" && ds?.plane && performance.now() - lastInteract > 180) {
        const atHome = Math.abs(view.x0 - home.x0) + Math.abs(view.x1 - home.x1) + Math.abs(view.y0 - home.y0) + Math.abs(view.y1 - home.y1) < (home.x1 - home.x0) * 0.01;
        const vw = Math.min(320, Math.max(80, Math.round(cssW / 4)));
        const vh = Math.min(240, Math.max(50, Math.round((vw * cssH) / cssW)));
        const key = atHome ? "home" : `${view.x0.toFixed(5)},${view.x1.toFixed(5)},${view.y0.toFixed(5)},${view.y1.toFixed(5)}`;
        if (key !== sentView && !(sentView === null && atHome)) {
          sentView = key;
          engine.send({ type: "set_view", view: atHome ? null : { x0: view.x0, x1: view.x1, y0: view.y0, y1: view.y1, w: vw, h: vh } });
        }
      }
      dirty = false;
      return anim;
    };

    const placeTicks = (host: HTMLDivElement | null, vals: number[], isX: boolean) => {
      if (!host) return;
      const step = vals.length > 1 ? vals[1] - vals[0] : 1;
      while (host.children.length < vals.length) {
        const s = document.createElement("span");
        s.className = "tk";
        host.appendChild(s);
      }
      for (let i = 0; i < host.children.length; i++) {
        const el = host.children[i] as HTMLSpanElement;
        if (i >= vals.length) { el.style.display = "none"; continue; }
        el.style.display = "";
        el.textContent = tickLabel(vals[i], step);
        if (isX) el.style.transform = `translate(${((vals[i] - view.x0) / (view.x1 - view.x0)) * cssW}px, 0) translateX(-50%)`;
        else el.style.transform = `translate(0, ${(1 - (vals[i] - view.y0) / (view.y1 - view.y0)) * cssH}px) translateY(-50%)`;
      }
    };

    const unsubLoop = subscribe(loop);
    kick.current = () => {
      flagsDirty = true;
      request(loop);
    };

    // ------------------------------------------------------------ sizing
    const ro = new ResizeObserver(() => {
      const r = box.getBoundingClientRect();
      cssW = Math.max(1, r.width);
      cssH = Math.max(1, r.height);
      gl.resize(cssW, cssH, devicePixelRatio || 1);
      if (ds) {
        const newHome = fitAspect(homeView(ds, frame), cssW, cssH, equal);
        const wasHome = Math.abs(view.x0 - home.x0) + Math.abs(view.x1 - home.x1) < 1e-9;
        home = newHome;
        if (wasHome || sentView === null) view = target = home;
      }
      dirty = true;
      request(loop);
    });
    ro.observe(box);

    // ------------------------------------------------------------ store subscriptions
    let curDs = useSession.getState().dataset;
    setDataset(curDs, null);
    const f0 = frameAt();
    if (f0) applyFrame(f0, false);
    const unsubS = useSession.subscribe((s, prev) => {
      if (s.dataset !== prev.dataset) {
        const p = curDs;
        curDs = s.dataset;
        setDataset(s.dataset, p);
      }
      const f = s.scrub != null ? s.frames[s.scrub] ?? s.latest : s.latest;
      const pf = prev.scrub != null ? prev.frames[prev.scrub] ?? prev.latest : prev.latest;
      if (f !== pf) applyFrame(f, s.scrub == null);
    });
    const unsubU = useUi.subscribe(() => {
      dirty = true;
      request(loop);
    });
    const mo = new MutationObserver(() => {
      C = colors();
      dirty = true;
      request(loop);
    });
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

    // ------------------------------------------------------------ interaction
    const nearest = (px: number, py: number, maxPx = 9): number | null => {
      const d = gl.pointData;
      let best = -1, bd = maxPx * maxPx;
      const sx = cssW / (view.x1 - view.x0), sy = cssH / (view.y1 - view.y0);
      for (let i = 0; i < gl.nPts; i++) {
        const dx = (d[i * 8] - view.x0) * sx - px;
        const dy = (view.y1 - d[i * 8 + 1]) * sy - py;
        const dd = dx * dx + dy * dy;
        if (dd < bd) { bd = dd; best = i; }
      }
      return best >= 0 && ds ? ds.ids.data[best] : null;
    };
    const local = (e: PointerEvent | WheelEvent | MouseEvent) => {
      const r = box.getBoundingClientRect();
      return [e.clientX - r.left, e.clientY - r.top];
    };
    let tipTimer: number | undefined;
    const setHover = (id: number | null, px = 0, py = 0) => {
      if (id === hover) return;
      hover = id;
      useUi.setState({ hoverId: id });
      clearTimeout(tipTimer);
      if (id == null) {
        tipTimer = setTimeout(() => setTip(null), 250) as unknown as number;
      } else if (propsRef.current.tooltip) {
        tipTimer = setTimeout(() => setTip({ id, x: px, y: py }), 300) as unknown as number;
      }
      if (useSession.getState().model === "MLPClassifier" && id != null) engine.send({ type: "probe", point_id: id });
      dirty = true;
      request(loop);
    };
    const editable = () => !!propsRef.current.editable && !!ds?.plane?.editable;
    const sendMove = (id: number, x: number, y: number) => {
      clearTimeout(moveTimer);
      moveTimer = setTimeout(() => engine.send({ type: "edit_points", add: [], move: [[id, x, y]], delete: [], retrain: true }), 150) as unknown as number;
    };

    const onDown = (e: PointerEvent) => {
      if (e.button !== 0) return;
      const [px, py] = local(e);
      const id = nearest(px, py);
      const [wx, wy] = toWorld(px, py);
      if (id != null && e.altKey && editable()) {
        engine.send({ type: "edit_points", add: [], move: [], delete: [id], retrain: true });
        setHover(null);
        return;
      }
      box.setPointerCapture(e.pointerId);
      interacting = true;
      vel = { x: 0, y: 0 };
      drag = { kind: id != null && editable() && !propsRef.current.onPick ? "point" : "pan", id: id ?? undefined, x: px, y: py, moved: false, wx, wy };
      lastMove = { x: px, y: py, t: performance.now() };
    };
    const onMove = (e: PointerEvent) => {
      const [px, py] = local(e);
      if (!drag) {
        setHover(nearest(px, py), px, py);
        return;
      }
      const dx = px - drag.x, dy = py - drag.y;
      if (!drag.moved && Math.hypot(dx, dy) > 3) drag.moved = true;
      if (!drag.moved) return;
      setTip(null);
      if (drag.kind === "point" && drag.id != null) {
        const i = idIndex.get(drag.id);
        if (i != null) {
          const [wx, wy] = toWorld(px, py);
          const d = gl.pointData;
          d[i * 8] = d[i * 8 + 2] = wx;
          d[i * 8 + 1] = d[i * 8 + 3] = wy;
          gl.uploadPoints();
          sendMove(drag.id, wx, wy);
        }
      } else {
        const sx = (view.x1 - view.x0) / cssW, sy = (view.y1 - view.y0) / cssH;
        const ox = -(px - lastMove.x) * sx, oy = (py - lastMove.y) * sy;
        view = target = { x0: view.x0 + ox, x1: view.x1 + ox, y0: view.y0 + oy, y1: view.y1 + oy };
        const now = performance.now();
        const dt = Math.max(1, now - lastMove.t) / 1000;
        vel = { x: (ox / dt) * 0.6 + vel.x * 0.4, y: (oy / dt) * 0.6 + vel.y * 0.4 };
        lastMove = { x: px, y: py, t: now };
      }
      lastInteract = performance.now();
      dirty = true;
      request(loop);
    };
    const onUp = (e: PointerEvent) => {
      const d = drag;
      drag = null;
      interacting = false;
      lastInteract = performance.now();
      if (!d) return;
      if (performance.now() - lastMove.t > 80) vel = { x: 0, y: 0 };
      if (!d.moved) {
        const P = propsRef.current;
        const [px, py] = local(e);
        const id = nearest(px, py);
        if (id != null && P.onPick) P.onPick(id);
        else if (id != null && P.tooltip) setTip({ id, x: px, y: py });
        else if (id == null && editable() && P.variant === "main") {
          const [wx, wy] = toWorld(px, py);
          clearTimeout(addTimer);
          addTimer = setTimeout(() => engine.send({ type: "edit_points", add: [[wx, wy, useUi.getState().addClass]], move: [], delete: [], retrain: true }), 260) as unknown as number;
        }
      }
      request(loop);
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const [px, py] = local(e);
      const [wx, wy] = toWorld(px, py);
      const f = Math.exp(Math.max(-0.5, Math.min(0.5, e.deltaY * (e.deltaMode ? 0.05 : 0.0016))));
      const tv = target;
      target = { x0: wx + (tv.x0 - wx) * f, x1: wx + (tv.x1 - wx) * f, y0: wy + (tv.y0 - wy) * f, y1: wy + (tv.y1 - wy) * f };
      lastInteract = performance.now();
      setTip(null);
      request(loop);
    };
    const onDbl = () => {
      clearTimeout(addTimer);
      target = home;
      vel = { x: 0, y: 0 };
      lastInteract = performance.now();
      request(loop);
    };
    const onLeave = () => !drag && setHover(null);
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "0" || e.key === "Home") onDbl();
      const pan = (fx: number, fy: number) => {
        const w = target.x1 - target.x0, h = target.y1 - target.y0;
        target = { x0: target.x0 + w * fx, x1: target.x1 + w * fx, y0: target.y0 + h * fy, y1: target.y1 + h * fy };
        request(loop);
      };
      if (e.key === "ArrowLeft") pan(-0.1, 0);
      else if (e.key === "ArrowRight") pan(0.1, 0);
      else if (e.key === "ArrowUp") pan(0, 0.1);
      else if (e.key === "ArrowDown") pan(0, -0.1);
      else if (e.key === "+" || e.key === "=" || e.key === "-") {
        const f = e.key === "-" ? 1.25 : 0.8;
        const cx = (target.x0 + target.x1) / 2, cy = (target.y0 + target.y1) / 2;
        target = { x0: cx + (target.x0 - cx) * f, x1: cx + (target.x1 - cx) * f, y0: cy + (target.y0 - cy) * f, y1: cy + (target.y1 - cy) * f };
        request(loop);
      } else return;
      lastInteract = performance.now();
      e.preventDefault();
    };
    box.addEventListener("pointerdown", onDown);
    box.addEventListener("pointermove", onMove);
    box.addEventListener("pointerup", onUp);
    box.addEventListener("pointercancel", onUp);
    box.addEventListener("pointerleave", onLeave);
    box.addEventListener("wheel", onWheel, { passive: false });
    box.addEventListener("dblclick", onDbl);
    box.addEventListener("keydown", onKey);
    request(loop);

    return () => {
      unsubLoop();
      unsubS();
      unsubU();
      ro.disconnect();
      mo.disconnect();
      clearTimeout(moveTimer);
      clearTimeout(addTimer);
      clearTimeout(tipTimer);
      box.removeEventListener("pointerdown", onDown);
      box.removeEventListener("pointermove", onMove);
      box.removeEventListener("pointerup", onUp);
      box.removeEventListener("pointercancel", onUp);
      box.removeEventListener("pointerleave", onLeave);
      box.removeEventListener("wheel", onWheel);
      box.removeEventListener("dblclick", onDbl);
      box.removeEventListener("keydown", onKey);
      gl.dispose();
    };
  }, []);

  // props that only change point flags: re-apply without rebuilding GL
  useEffect(() => {
    kick.current?.();
  }, [props.highlightId, props.dim, props.highlightSet, props.ringSet, props.extra, props.labels]);

  const ax = dataset?.plane?.axes ?? ["x1", "x2"];
  return (
    <div className={`plot plot-${props.variant}`} data-testid={props.testId}>
      <div ref={wrap} className="plot-area" tabIndex={0} role="img"
        aria-label={t("plot.aria", { x: ax[0], y: ax[1] || "" })}>
        <canvas ref={canvasRef} className="plot-canvas" />
        <div ref={centersRef} className="plot-centers" />
        <div ref={labelsRef} className="plot-labels" />
        <div ref={ringRef} className="ring" style={{ display: "none" }} />
        {glError && <div className="plot-error">{t("plot.noWebgl")}</div>}
        {props.children}
        {tip && props.tooltip && <PointTip id={tip.id} x={tip.x} y={tip.y} onClose={() => setTip(null)} />}
      </div>
      {props.axes !== false && (
        <>
          <div ref={ticksX} className="ticks-x" aria-hidden="true" />
          <div ref={ticksY} className="ticks-y" aria-hidden="true" />
          <span className="axl axl-x">{ax[0]} →</span>
          {ax[1] && <span className="axl axl-y">{ax[1]} ↑</span>}
        </>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ hover tooltip

function PointTip({ id, x, y, onClose }: { id: number; x: number; y: number; onClose: () => void }) {
  const { t } = useTranslation();
  const ds = useSession((s) => s.dataset);
  const latest = useSession((s) => s.latest);
  const model = useSession((s) => s.model);
  if (!ds) return null;
  const idx = ds.ids.data.indexOf(id);
  const row = ds.table.rows.find((r) => r[0] === id);
  const cols = ds.table.columns;
  const cls = ds.task === "classification" ? ds.classes[ds.label.data[idx]] : null;
  const pp = latest?.final ? latest.viz.point_pred : undefined;
  let pred: string | null = null;
  if (pp && idx >= 0) {
    if (pp.shape.length === 2) {
      const k = pp.shape[1];
      const pr = Array.from(pp.data.subarray(idx * k, idx * k + k));
      const best = pr.indexOf(Math.max(...pr));
      const right = ds.classes[best] === cls;
      const score = latest?.viz.score;
      pred = t(score ? "tip.predScore" : "tip.pred", { cls: ds.classes[best], p: fmt(pr[best]), verdict: right ? t("tip.correct") : t("tip.wrong") });
    } else pred = t("tip.predValue", { v: fmt(pp.data[idx]) });
  }
  const isTest = idx >= 0 && ds.test.data[idx] === 1;
  const left = x + 22;
  const goExplain = () => {
    useUi.setState({ decidePoint: id });
    onClose();
    useNav.getState().go("decide");
  };
  return (
    <div className="tip interactive ptip" style={{ position: "absolute", left, top: Math.max(4, y - 150), width: 260 }}
      onPointerDown={(e) => e.stopPropagation()} role="dialog" aria-label={t("tip.point", { id })}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6, gap: 8 }}>
        <b>{t("tip.point", { id })}</b>
        {cls != null && <span className="tag" style={{ height: 22 }}><span className={`dot k${ds.label.data[idx] % 5} s${ds.label.data[idx] % 4}`} />{t("tip.true", { cls })}</span>}
      </div>
      <div className="mono" style={{ fontSize: 12, display: "grid", gridTemplateColumns: "auto 1fr", gap: "2px 12px" }}>
        {row && cols.map((c, j) => (
          <span key={c.name} style={{ display: "contents" }}>
            <span>{c.name}</span>
            <span style={{ color: "var(--ink)" }}>{typeof row[j + 1] === "number" ? fmt(row[j + 1] as number) : String(row[j + 1] ?? "—")}</span>
          </span>
        ))}
        {pred && (<><span>{t("tip.model")}</span><span style={{ color: "var(--ink)" }}>{pred}</span></>)}
      </div>
      <div className="hint" style={{ marginTop: 4 }}>{isTest ? t("tip.testPoint") : t("tip.trainPoint")}</div>
      {latest && model && <button className="linkbtn" style={{ marginTop: 8 }} onClick={goExplain}>{t("tip.explain")}</button>}
    </div>
  );
}
