// Lightweight Canvas-2D scatter for thumbnails, previews and the pair plot.
// Points tween to new positions when the data changes (same index = same point).
// Supports wheel zoom, drag pan and double-click reset.
import { useEffect, useRef } from "react";
import { backdrop, css, cssVar, reducedMotion, request, subscribe } from "../gl/loop";

export interface ScatterProps {
  xy: Float32Array | null;
  labels?: Float32Array | null;
  single?: boolean;
  size?: number;
  bounds?: [number, number, number, number] | null;
  pad?: number;
  highlight?: number | null; // index
  rings?: Set<number> | null; // indices drawn with a warn ring
  line?: [number, number, number, number] | null; // x0,y0,x1,y1 in data coords (dashed)
  onHover?: (index: number | null) => void;
  interactive?: boolean;
  ariaLabel?: string;
  className?: string;
}

const SHAPES = ["circle", "square", "triangle", "diamond"] as const;

function drawShape(ctx: CanvasRenderingContext2D, k: number, x: number, y: number, s: number) {
  const sh = SHAPES[((k % 4) + 4) % 4];
  ctx.beginPath();
  if (sh === "circle") ctx.arc(x, y, s / 2, 0, Math.PI * 2);
  else if (sh === "square") ctx.rect(x - s * 0.45, y - s * 0.45, s * 0.9, s * 0.9);
  else if (sh === "triangle") { const r = s * 0.62; ctx.moveTo(x, y - r * 0.8); ctx.lineTo(x + r * 0.75, y + r * 0.5); ctx.lineTo(x - r * 0.75, y + r * 0.5); ctx.closePath(); }
  else { const r = s * 0.55; ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y); ctx.closePath(); }
  ctx.fill();
}

export function Scatter(props: ScatterProps) {
  const cv = useRef<HTMLCanvasElement>(null);
  const P = useRef(props);
  P.current = props;
  const kick = useRef<() => void>(() => undefined);
  useEffect(() => {
    const c = cv.current!;
    const ctx = c.getContext("2d")!;
    let W = 1, H = 1;
    let from: Float32Array | null = null;
    let to: Float32Array | null = null;
    let t0 = 0;
    let shown: Float32Array | null = null;
    let zoom = { x0: 0, x1: 1, y0: 0, y1: 1 };
    let home = zoom;
    let custom = false;
    let lastBounds = "";
    const pal = () => [0, 1, 2, 3, 4].map((i) => cssVar(`--c${i}`, c));
    const computeHome = () => {
      const p = P.current;
      if (p.bounds) return { x0: p.bounds[0], x1: p.bounds[1], y0: p.bounds[2], y1: p.bounds[3] };
      const d = to;
      if (!d || !d.length) return { x0: 0, x1: 1, y0: 0, y1: 1 };
      let a = Infinity, b = -Infinity, e = Infinity, f = -Infinity;
      for (let i = 0; i < d.length; i += 2) { a = Math.min(a, d[i]); b = Math.max(b, d[i]); e = Math.min(e, d[i + 1]); f = Math.max(f, d[i + 1]); }
      const px = (b - a || 1) * (p.pad ?? 0.06), py = (f - e || 1) * (p.pad ?? 0.06);
      return { x0: a - px, x1: b + px, y0: e - py, y1: f + py };
    };
    const loop = (now: number): boolean => {
      const p = P.current;
      const dpr = devicePixelRatio || 1;
      if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = css(backdrop(c));
      ctx.fillRect(0, 0, W, H);
      if (!to) return false;
      const k = from && from.length === to.length && !reducedMotion() ? Math.min(1, (now - t0) / 350) : 1;
      const e = 1 - Math.pow(1 - k, 3);
      const n = to.length / 2;
      if (!shown || shown.length !== to.length) shown = new Float32Array(to.length);
      for (let i = 0; i < to.length; i++) shown[i] = from && from.length === to.length ? from[i] + (to[i] - from[i]) * e : to[i];
      const X = (x: number) => ((x - zoom.x0) / (zoom.x1 - zoom.x0)) * W;
      const Y = (y: number) => (1 - (y - zoom.y0) / (zoom.y1 - zoom.y0)) * H;
      if (p.line) {
        ctx.strokeStyle = cssVar("--ink", c);
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(X(p.line[0]), Y(p.line[1])); ctx.lineTo(X(p.line[2]), Y(p.line[3])); ctx.stroke();
        ctx.setLineDash([]);
      }
      const colors = pal();
      const s = p.size ?? 6;
      const lab = p.labels;
      for (let i = 0; i < n; i++) {
        const cls = p.single || !lab ? 0 : Math.max(0, lab[i]);
        ctx.fillStyle = colors[cls % 5];
        ctx.globalAlpha = p.highlight != null && p.highlight !== i ? 0.55 : 0.85;
        drawShape(ctx, p.single ? 0 : cls, X(shown[i * 2]), Y(shown[i * 2 + 1]), s);
      }
      ctx.globalAlpha = 1;
      if (p.rings) {
        ctx.strokeStyle = cssVar("--warn", c);
        ctx.lineWidth = 2;
        for (const i of p.rings) if (i < n) { ctx.beginPath(); ctx.arc(X(shown[i * 2]), Y(shown[i * 2 + 1]), s * 0.9 + 2, 0, Math.PI * 2); ctx.stroke(); }
      }
      if (p.highlight != null && p.highlight < n) {
        const i = p.highlight;
        ctx.strokeStyle = cssVar("--ink", c);
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(X(shown[i * 2]), Y(shown[i * 2 + 1]), s + 3, 0, Math.PI * 2); ctx.stroke();
      }
      return k < 1;
    };
    const un = subscribe(loop);
    kick.current = () => {
      const p = P.current;
      if (p.xy !== to) {
        from = shown ? new Float32Array(shown) : null;
        to = p.xy;
        t0 = performance.now();
      }
      const bkey = p.bounds ? p.bounds.join() : "auto";
      if (!custom || bkey !== lastBounds) {
        home = computeHome();
        if (!custom) zoom = home;
      }
      lastBounds = bkey;
      request(loop);
    };
    const ro = new ResizeObserver(() => {
      const r = c.getBoundingClientRect();
      W = Math.max(1, r.width);
      H = Math.max(1, r.height);
      request(loop);
    });
    ro.observe(c);
    const mo = new MutationObserver(() => request(loop));
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    // interaction
    let drag: { x: number; y: number } | null = null;
    const local = (e: MouseEvent) => { const r = c.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    const onWheel = (e: WheelEvent) => {
      if (!P.current.interactive) return;
      e.preventDefault();
      const [px, py] = local(e);
      const wx = zoom.x0 + (px / W) * (zoom.x1 - zoom.x0), wy = zoom.y1 - (py / H) * (zoom.y1 - zoom.y0);
      const f = Math.exp(Math.max(-0.4, Math.min(0.4, e.deltaY * 0.0016)));
      zoom = { x0: wx + (zoom.x0 - wx) * f, x1: wx + (zoom.x1 - wx) * f, y0: wy + (zoom.y0 - wy) * f, y1: wy + (zoom.y1 - wy) * f };
      custom = true;
      request(loop);
    };
    const onDown = (e: PointerEvent) => { if (P.current.interactive) { drag = { x: e.clientX, y: e.clientY }; c.setPointerCapture(e.pointerId); } };
    const onMove = (e: PointerEvent) => {
      const p = P.current;
      if (drag) {
        const dx = ((e.clientX - drag.x) / W) * (zoom.x1 - zoom.x0), dy = ((e.clientY - drag.y) / H) * (zoom.y1 - zoom.y0);
        zoom = { x0: zoom.x0 - dx, x1: zoom.x1 - dx, y0: zoom.y0 + dy, y1: zoom.y1 + dy };
        drag = { x: e.clientX, y: e.clientY };
        custom = true;
        request(loop);
        return;
      }
      if (!p.onHover || !shown) return;
      const [px, py] = local(e);
      let best = -1, bd = 64;
      for (let i = 0; i < shown.length / 2; i++) {
        const dx = ((shown[i * 2] - zoom.x0) / (zoom.x1 - zoom.x0)) * W - px;
        const dy = (1 - (shown[i * 2 + 1] - zoom.y0) / (zoom.y1 - zoom.y0)) * H - py;
        const d = dx * dx + dy * dy;
        if (d < bd) { bd = d; best = i; }
      }
      p.onHover(best >= 0 ? best : null);
    };
    const onUp = () => { drag = null; };
    const onLeave = () => P.current.onHover?.(null);
    const onDbl = () => { zoom = home; custom = false; request(loop); };
    c.addEventListener("wheel", onWheel, { passive: false });
    c.addEventListener("pointerdown", onDown);
    c.addEventListener("pointermove", onMove);
    c.addEventListener("pointerup", onUp);
    c.addEventListener("pointerleave", onLeave);
    c.addEventListener("dblclick", onDbl);
    kick.current();
    return () => {
      un(); ro.disconnect(); mo.disconnect();
      c.removeEventListener("wheel", onWheel);
      c.removeEventListener("pointerdown", onDown);
      c.removeEventListener("pointermove", onMove);
      c.removeEventListener("pointerup", onUp);
      c.removeEventListener("pointerleave", onLeave);
      c.removeEventListener("dblclick", onDbl);
    };
  }, []);
  useEffect(() => kick.current());
  return <canvas ref={cv} className={props.className} style={{ width: "100%", height: "100%", display: "block", cursor: props.interactive ? "grab" : undefined }} role="img" aria-label={props.ariaLabel} />;
}
