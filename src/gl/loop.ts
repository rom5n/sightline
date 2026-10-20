// One requestAnimationFrame loop for every animated surface. Renderers ask for frames
// while they animate; the loop sleeps when nobody needs it.
import { perfStats } from "../app/perf";

type Fn = (now: number, dt: number) => boolean | void; // return true = keep animating

const subs = new Set<Fn>();
const wanted = new Set<Fn>();
let raf = 0;
let last = 0;

function tick(now: number) {
  raf = 0;
  const dt = last ? Math.min(0.1, (now - last) / 1000) : 1 / 60;
  last = now;
  const t0 = performance.now();
  const run = [...wanted];
  wanted.clear();
  for (const f of run) {
    if (!subs.has(f)) continue;
    if (f(now, dt)) wanted.add(f);
  }
  perfStats.renderMs = perfStats.renderMs * 0.9 + (performance.now() - t0) * 0.1;
  if (wanted.size) raf = requestAnimationFrame(tick);
  else last = 0;
}

export function subscribe(f: Fn): () => void {
  subs.add(f);
  return () => {
    subs.delete(f);
    wanted.delete(f);
  };
}

export function request(f: Fn) {
  if (!subs.has(f)) return;
  wanted.add(f);
  if (!raf) raf = requestAnimationFrame(tick);
}

export const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

export function cssVar(name: string, el: Element = document.documentElement): string {
  return getComputedStyle(el).getPropertyValue(name).trim();
}

/** "#rgb" | "#rgba" | "#rrggbb" | "#rrggbbaa" | "rgb(a)(…)" -> [r,g,b,a] in 0..1.
 *  The production CSS minifier rewrites rgba(255,255,255,.06) as #ffffff0f, so hex alpha matters. */
export function parseColor(c: string): [number, number, number, number] {
  c = c.trim();
  if (c.startsWith("#")) {
    let h = c.slice(1);
    if (h.length <= 4) h = h.split("").map((x) => x + x).join("");
    const v = (i: number) => parseInt(h.slice(i, i + 2), 16) / 255;
    return [v(0), v(2), v(4), h.length >= 8 ? v(6) : 1];
  }
  const m = c.match(/rgba?\(([^)]+)\)/);
  if (m) {
    // rgba(1, 2, 3, .5) or rgb(1 2 3 / 50%)
    const p = m[1].split(/[\s,/]+/).filter(Boolean).map((x) => (x.endsWith("%") ? parseFloat(x) / 100 : parseFloat(x)));
    return [p[0] / 255, p[1] / 255, p[2] / 255, p.length > 3 ? p[3] : 1];
  }
  return [1, 1, 1, 1];
}

/** The opaque colour behind `el` (first ancestor with a solid background). Canvases paint it
 *  themselves, so a chart copied or saved from the context menu is not a transparent image. */
export function backdrop(el: Element): [number, number, number, number] {
  for (let e = el.parentElement; e; e = e.parentElement) {
    const c = parseColor(getComputedStyle(e).backgroundColor);
    if (c[3] >= 0.99) return c;
  }
  return parseColor(cssVar("--solid"));
}

export const css = (c: [number, number, number, number]) =>
  `rgba(${Math.round(c[0] * 255)}, ${Math.round(c[1] * 255)}, ${Math.round(c[2] * 255)}, ${c[3]})`;
