// Canvas 2-D line chart for Loss / metric curves. The y-axis eases to new ranges,
// the newest point glides in with a soft glow, never a jump.
import { useEffect, useRef } from "react";
import { backdrop, css, cssVar, reducedMotion, request, subscribe } from "../gl/loop";

export interface Series { xs: number[]; ys: (number | null)[]; color: "acc" | "ink2" | "ink3"; dash?: boolean; width?: number; dots?: boolean }
export interface Marker { x: number; label?: string; dash?: boolean }

export interface LineChartProps {
  series: Series[];
  xMin?: number;
  xMax: number;
  yMin?: number;
  yMax?: number;
  zeroBased?: boolean;
  markers?: Marker[];
  point?: { x: number; y: number; label?: string } | null;
  glow?: boolean;
  xLabel?: string;
  yFmt?: (v: number) => string;
  xTicks?: number[];
  cursor?: number | null;
  height?: number | string;
  ariaLabel?: string;
}

const PAD = { l: 38, r: 8, t: 8, b: 20 };

function niceStep(span: number, n: number) {
  const raw = span / n;
  const p = Math.pow(10, Math.floor(Math.log10(raw || 1)));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
}

export function LineChart(props: LineChartProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const box = useRef<HTMLDivElement>(null);
  const P = useRef(props);
  P.current = props;
  const state = useRef({ y0: 0, y1: 1, init: false, lead: 1, leadFrom: 0, leadT0: 0, lastN: 0, w: 1, h: 1 });
  /** user zoom/pan on the x axis (null = fit) */
  const xr = useRef<{ x0: number; x1: number } | null>(null);
  const draw = useRef<() => void>(() => undefined);

  useEffect(() => {
    const c = ref.current!;
    const ctx = c.getContext("2d")!;
    const s = state.current;
    const loop = (now: number, dt: number): boolean => {
      if (!box.current) return false;
      const p = P.current;
      const dpr = devicePixelRatio || 1;
      const W = s.w, H = s.h;
      if (c.width !== Math.round(W * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = css(backdrop(c));
      ctx.fillRect(0, 0, W, H);
      // hairlines: whole device pixels, centred on a pixel, so they stay crisp at 125–200 % scaling
      const hw = Math.max(1, Math.round(dpr)), hair = hw / dpr;
      const snap = (v: number) => (Math.round(v * dpr - (hw % 2) * 0.5) + (hw % 2) * 0.5) / dpr;
      const fx0 = p.xMin ?? 0, fx1 = Math.max(p.xMax, fx0 + 1e-9);
      const vx0 = xr.current?.x0 ?? fx0, vx1 = xr.current?.x1 ?? fx1;
      // target y range (only what is in view)
      let lo = Infinity, hi = -Infinity;
      for (const se of p.series) se.ys.forEach((v, i) => { const x = se.xs[i]; if (v != null && Number.isFinite(v) && x >= vx0 && x <= vx1) { lo = Math.min(lo, v); hi = Math.max(hi, v); } });
      if (p.point) { lo = Math.min(lo, p.point.y); hi = Math.max(hi, p.point.y); }
      if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
      if (p.zeroBased) lo = Math.min(0, lo);
      if (p.yMin != null) lo = p.yMin;
      if (p.yMax != null) hi = p.yMax;
      if (hi - lo < 1e-9) { hi = lo + (Math.abs(lo) || 1) * 0.1; lo = lo - (Math.abs(lo) || 1) * 0.1; }
      const padY = (hi - lo) * 0.08;
      const ty0 = p.yMin != null ? lo : lo - (p.zeroBased ? 0 : padY), ty1 = p.yMax != null ? hi : hi + padY;
      let anim = false;
      if (!s.init || reducedMotion()) { s.y0 = ty0; s.y1 = ty1; s.init = true; }
      else {
        const k = 1 - Math.exp(-dt * 7);
        s.y0 += (ty0 - s.y0) * k;
        s.y1 += (ty1 - s.y1) * k;
        if (Math.abs(ty0 - s.y0) + Math.abs(ty1 - s.y1) > (ty1 - ty0) * 1e-3) anim = true;
        else { s.y0 = ty0; s.y1 = ty1; }
      }
      // newest point glides in
      const n = p.series[0]?.xs.length ?? 0;
      if (n !== s.lastN) {
        s.leadFrom = s.lastN > 0 && n === s.lastN + 1 ? 0 : 1;
        s.leadT0 = now;
        s.lastN = n;
      }
      const lead = s.leadFrom === 1 || reducedMotion() ? 1 : Math.min(1, (now - s.leadT0) / 140);
      if (lead < 1) anim = true;

      const x0 = vx0, x1 = vx1;
      const iw = W - PAD.l - PAD.r, ih = H - PAD.t - PAD.b;
      ctx.save();
      const X = (x: number) => PAD.l + ((x - x0) / (x1 - x0)) * iw;
      const Y = (y: number) => PAD.t + (1 - (y - s.y0) / (s.y1 - s.y0)) * ih;
      const col = (k: string) => cssVar(k === "acc" ? "--acc-t" : k === "ink2" ? "--ink2" : "--ink3", box.current!);
      const font = `500 12px ${cssVar("--mono", box.current!) || "monospace"}`;
      // grid + y ticks
      ctx.font = font;
      ctx.fillStyle = cssVar("--ink3", box.current!);
      ctx.strokeStyle = cssVar("--gl", box.current!);
      ctx.lineWidth = hair;
      const st = niceStep(s.y1 - s.y0, Math.max(2, Math.round(ih / 40)));
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let v = Math.ceil(s.y0 / st) * st; v <= s.y1 + 1e-9; v += st) {
        const y = snap(Y(v));
        if (y < PAD.t - 1 || y > H - PAD.b + 1) continue;
        ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(W - PAD.r, y); ctx.stroke();
        ctx.fillText((p.yFmt ? p.yFmt(v) : fmtTick(v, st)), PAD.l - 6, y);
      }
      ctx.strokeStyle = cssVar("--ink3", box.current!);
      ctx.beginPath(); ctx.moveTo(PAD.l, snap(H - PAD.b)); ctx.lineTo(W - PAD.r, snap(H - PAD.b)); ctx.stroke();
      // x ticks
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      const xt = p.xTicks ?? [x0, (x0 + x1) / 2, x1].map((v) => (x1 - x0 > 4 ? Math.round(v) : v));
      xt.forEach((v, i) => {
        ctx.textAlign = i === 0 ? "left" : i === xt.length - 1 ? "right" : "center";
        const label = i === xt.length - 1 && p.xLabel ? `${fmtTick(v, 1)} ${p.xLabel}` : fmtTick(v, x1 - x0 > 4 ? 1 : 0.1);
        ctx.fillText(label, i === 0 ? PAD.l : i === xt.length - 1 ? W - PAD.r : X(v), H - PAD.b + 4);
      });
      // markers
      for (const m of p.markers ?? []) {
        const x = snap(X(m.x));
        ctx.strokeStyle = cssVar("--ink3", box.current!);
        ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, H - PAD.b); ctx.stroke();
        ctx.setLineDash([]);
        if (m.label) {
          ctx.fillStyle = cssVar("--ink2", box.current!);
          ctx.textAlign = X(m.x) > W * 0.7 ? "right" : "left";
          ctx.textBaseline = "top";
          ctx.fillText(m.label, x + (X(m.x) > W * 0.7 ? -6 : 6), PAD.t);
        }
      }
      // series (clipped to the plot area while zoomed)
      ctx.beginPath(); ctx.rect(PAD.l, 0, iw, H); ctx.clip();
      let leadPt: [number, number] | null = null;
      p.series.forEach((se, si) => {
        ctx.strokeStyle = col(se.color);
        ctx.lineWidth = se.width ?? (se.dash ? 1.5 : 2);
        ctx.setLineDash(se.dash ? [4, 4] : []);
        ctx.lineJoin = "round";
        ctx.beginPath();
        let started = false;
        const m = se.xs.length;
        for (let i = 0; i < m; i++) {
          const v = se.ys[i];
          if (v == null || !Number.isFinite(v)) { started = false; continue; }
          let x = X(se.xs[i]), y = Y(v);
          if (i === m - 1 && i > 0 && lead < 1 && se.ys[i - 1] != null) {
            const px = X(se.xs[i - 1]), py = Y(se.ys[i - 1] as number);
            const e = 1 - Math.pow(1 - lead, 3);
            x = px + (x - px) * e;
            y = py + (y - py) * e;
          }
          if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
          if (i === m - 1 && si === 0) leadPt = [x, y];
        }
        ctx.stroke();
        ctx.setLineDash([]);
        if (se.dots) {
          ctx.fillStyle = col(se.color);
          for (let i = 0; i < m; i++) {
            const v = se.ys[i];
            if (v == null) continue;
            ctx.beginPath(); ctx.arc(X(se.xs[i]), Y(v), 3.5, 0, Math.PI * 2); ctx.fill();
          }
        }
      });
      const acc = cssVar("--acc", box.current!);
      const glowPt = p.point ? [X(p.point.x), Y(p.point.y)] as [number, number] : p.glow ? leadPt : null;
      if (glowPt) {
        const [gx, gy] = glowPt;
        ctx.fillStyle = cssVar("--acc-soft", box.current!);
        ctx.beginPath(); ctx.arc(gx, gy, 8, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = acc;
        ctx.shadowColor = acc;
        ctx.shadowBlur = 8;
        ctx.beginPath(); ctx.arc(gx, gy, 4.5, 0, Math.PI * 2); ctx.fill();
        ctx.shadowBlur = 0;
        if (p.point?.label) {
          ctx.fillStyle = cssVar("--ink", box.current!);
          ctx.textAlign = gx > W * 0.75 ? "right" : "left";
          ctx.textBaseline = "bottom";
          ctx.fillText(p.point.label, gx + (gx > W * 0.75 ? -10 : 10), gy - 6);
        }
      }
      ctx.restore();
      if (p.cursor != null) {
        const x = snap(X(p.cursor));
        ctx.strokeStyle = cssVar("--ink", box.current!);
        ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, H - PAD.b); ctx.stroke();
      }
      return anim;
    };
    const un = subscribe(loop);
    draw.current = () => request(loop);
    const ro = new ResizeObserver(() => {
      if (!box.current) return;
      const r = box.current.getBoundingClientRect();
      s.w = Math.max(10, r.width);
      s.h = Math.max(10, r.height);
      request(loop);
    });
    ro.observe(box.current!);
    const mo = new MutationObserver(() => request(loop));
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    // wheel zooms the x axis around the cursor, drag pans, double-click resets
    const el = c;
    const full = () => { const p = P.current; const a = p.xMin ?? 0; return { x0: a, x1: Math.max(p.xMax, a + 1e-9) }; };
    const span = () => xr.current ?? full();
    const xAt = (clientX: number) => { const r = el.getBoundingClientRect(); const v = span(); return v.x0 + ((clientX - r.left - PAD.l) / Math.max(1, r.width - PAD.l - PAD.r)) * (v.x1 - v.x0); };
    const clampTo = (v: { x0: number; x1: number }) => {
      const f = full(), w = Math.min(v.x1 - v.x0, f.x1 - f.x0);
      if (w >= f.x1 - f.x0 - 1e-9) return null;
      const a = Math.max(f.x0, Math.min(v.x0, f.x1 - w));
      return { x0: a, x1: a + w };
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const v = span(), x = xAt(e.clientX), k = Math.exp(Math.max(-0.4, Math.min(0.4, e.deltaY * 0.0016)));
      xr.current = clampTo({ x0: x + (v.x0 - x) * k, x1: x + (v.x1 - x) * k });
      request(loop);
    };
    let drag: number | null = null;
    const onDown = (e: PointerEvent) => { if (xr.current) { drag = e.clientX; el.setPointerCapture(e.pointerId); } };
    const onMove = (e: PointerEvent) => {
      if (drag == null || !xr.current) return;
      const r = el.getBoundingClientRect(), v = xr.current;
      const dx = ((e.clientX - drag) / Math.max(1, r.width - PAD.l - PAD.r)) * (v.x1 - v.x0);
      drag = e.clientX;
      xr.current = clampTo({ x0: v.x0 - dx, x1: v.x1 - dx }) ?? v;
      request(loop);
    };
    const onUp = () => { drag = null; };
    const onDbl = () => { xr.current = null; request(loop); };
    el.addEventListener("wheel", onWheel, { passive: false });
    el.addEventListener("pointerdown", onDown);
    el.addEventListener("pointermove", onMove);
    el.addEventListener("pointerup", onUp);
    el.addEventListener("dblclick", onDbl);
    return () => {
      un(); ro.disconnect(); mo.disconnect();
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("pointerdown", onDown);
      el.removeEventListener("pointermove", onMove);
      el.removeEventListener("pointerup", onUp);
      el.removeEventListener("dblclick", onDbl);
    };
  }, []);

  useEffect(() => draw.current());

  return (
    <div ref={box} className="chart-box" style={{ height: props.height ?? "100%" }} role="img" aria-label={props.ariaLabel}>
      <canvas ref={ref} style={{ width: "100%", height: "100%", display: "block" }} />
    </div>
  );
}

function fmtTick(v: number, step: number) {
  const a = Math.abs(v);
  if (a >= 10000) return Math.round(v).toLocaleString("en-US").replace(/,/g, " ");
  const d = step >= 1 ? 0 : Math.min(4, Math.ceil(-Math.log10(step) - 1e-9));
  return v.toFixed(d).replace("-", "−");
}
