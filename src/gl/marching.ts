// Marching squares on a (h × w) scalar field sampled at cell centres; returns line
// segments [x0,y0,x1,y1,...] in world coordinates for the 0-level of the field.
export function marching(field: Float32Array, w: number, h: number, bounds: [number, number, number, number], out?: number[]): Float32Array {
  const [x0, x1, y0, y1] = bounds;
  const dx = (x1 - x0) / w;
  const dy = (y1 - y0) / h;
  const ox = x0 + dx / 2;
  const oy = y0 + dy / 2;
  const seg = out ?? [];
  const lerp = (a: number, b: number) => (a === b ? 0.5 : a / (a - b));
  for (let j = 0; j < h - 1; j++) {
    for (let i = 0; i < w - 1; i++) {
      const a = field[j * w + i]; // bottom-left
      const b = field[j * w + i + 1]; // bottom-right
      const c = field[(j + 1) * w + i + 1]; // top-right
      const d = field[(j + 1) * w + i]; // top-left
      let code = 0;
      if (a > 0) code |= 1;
      if (b > 0) code |= 2;
      if (c > 0) code |= 4;
      if (d > 0) code |= 8;
      if (code === 0 || code === 15) continue;
      const X = ox + i * dx;
      const Y = oy + j * dy;
      // edge points: bottom (a-b), right (b-c), top (d-c), left (a-d)
      const eb = () => [X + lerp(a, b) * dx, Y];
      const er = () => [X + dx, Y + lerp(b, c) * dy];
      const et = () => [X + lerp(d, c) * dx, Y + dy];
      const el = () => [X, Y + lerp(a, d) * dy];
      const push = (p: number[], q: number[]) => seg.push(p[0], p[1], q[0], q[1]);
      switch (code) {
        case 1: case 14: push(el(), eb()); break;
        case 2: case 13: push(eb(), er()); break;
        case 3: case 12: push(el(), er()); break;
        case 4: case 11: push(er(), et()); break;
        case 6: case 9: push(eb(), et()); break;
        case 7: case 8: push(el(), et()); break;
        case 5: case 10: {
          const center = (a + b + c + d) / 4;
          if ((center > 0) === (code === 5)) {
            push(el(), et());
            push(eb(), er());
          } else {
            push(el(), eb());
            push(er(), et());
          }
          break;
        }
      }
    }
  }
  return out ? new Float32Array(0) : tidy(seg, Math.min(dx, dy) * 0.15);
}

/** Joins marching-squares segments into polylines and drops the points that lie within `tol`
 *  of a straight run (Ramer–Douglas–Peucker). Hundreds of cell-sized pieces become a few long
 *  ones, so the border reads as one straight or smoothly curved line instead of a jagged chain. */
export function tidy(seg: number[], tol: number): Float32Array {
  const n = seg.length / 4;
  if (n < 2) return new Float32Array(seg);
  const q = tol * 1e-3 || 1e-9;
  const key = (x: number, y: number) => `${Math.round(x / q)},${Math.round(y / q)}`;
  const ends = new Map<string, number[]>(); // endpoint -> segment ids touching it
  for (let s = 0; s < n; s++) {
    for (const e of [0, 2]) {
      const k = key(seg[s * 4 + e], seg[s * 4 + e + 1]);
      const l = ends.get(k);
      if (l) l.push(s); else ends.set(k, [s]);
    }
  }
  const used = new Uint8Array(n);
  const out: number[] = [];
  const next = (x: number, y: number, from: number) => (ends.get(key(x, y)) ?? []).find((s) => s !== from && !used[s]);
  for (let s0 = 0; s0 < n; s0++) {
    if (used[s0]) continue;
    used[s0] = 1;
    // grow both ways from s0
    const pts: number[] = [seg[s0 * 4], seg[s0 * 4 + 1], seg[s0 * 4 + 2], seg[s0 * 4 + 3]];
    for (const forward of [true, false]) {
      let cur = s0;
      for (;;) {
        const x = forward ? pts[pts.length - 2] : pts[0], y = forward ? pts[pts.length - 1] : pts[1];
        const s = next(x, y, cur);
        if (s === undefined) break;
        used[s] = 1;
        const a = key(seg[s * 4], seg[s * 4 + 1]) === key(x, y);
        const px = a ? seg[s * 4 + 2] : seg[s * 4], py = a ? seg[s * 4 + 3] : seg[s * 4 + 1];
        if (forward) pts.push(px, py); else pts.unshift(px, py);
        cur = s;
      }
    }
    const keep = rdp(pts, tol);
    for (let i = 0; i + 1 < keep.length; i++) out.push(pts[keep[i] * 2], pts[keep[i] * 2 + 1], pts[keep[i + 1] * 2], pts[keep[i + 1] * 2 + 1]);
  }
  return new Float32Array(out);
}

function rdp(pts: number[], tol: number): number[] {
  const m = pts.length / 2;
  const keep = new Uint8Array(m);
  keep[0] = keep[m - 1] = 1;
  const stack: [number, number][] = [[0, m - 1]];
  while (stack.length) {
    const [a, b] = stack.pop()!;
    const ax = pts[a * 2], ay = pts[a * 2 + 1], bx = pts[b * 2], by = pts[b * 2 + 1];
    const dx = bx - ax, dy = by - ay, L = Math.hypot(dx, dy) || 1e-12;
    let best = -1, far = tol;
    for (let i = a + 1; i < b; i++) {
      const d = Math.abs((pts[i * 2] - ax) * dy - (pts[i * 2 + 1] - ay) * dx) / L;
      if (d > far) { far = d; best = i; }
    }
    if (best >= 0) { keep[best] = 1; stack.push([a, best], [best, b]); }
  }
  const idx: number[] = [];
  keep.forEach((k, i) => { if (k) idx.push(i); });
  return idx;
}

// scratch field reused across calls: borders are recomputed every animation frame while a grid morphs
let scratch = new Float32Array(0);
function field(n: number) {
  if (scratch.length < n) scratch = new Float32Array(n);
  return scratch;
}

/** Decision border(s) from a (K,H,W) probability grid: p1 − 0.5 for 2 classes,
 *  p_k − max_other for more (each border appears from both sides; drawn once is enough). */
export function borderSegments(probs: Float32Array, k: number, w: number, h: number, bounds: [number, number, number, number]): Float32Array {
  const n = w * h;
  if (k === 1) return new Float32Array(0);
  const f = field(n);
  if (k === 2) {
    for (let i = 0; i < n; i++) f[i] = probs[n + i] - 0.5;
    return marching(f, w, h, bounds);
  }
  const out: number[] = [];
  for (let c = 0; c < k - 1; c++) {
    let any = false;
    for (let i = 0; i < n; i++) {
      let best = -1;
      for (let o = 0; o < k; o++) if (o !== c && probs[o * n + i] > best) best = probs[o * n + i];
      f[i] = probs[c * n + i] - best;
      if (f[i] > 0) any = true;
    }
    if (any) marching(f, w, h, bounds, out);
  }
  return tidy(out, Math.min((bounds[1] - bounds[0]) / w, (bounds[3] - bounds[2]) / h) * 0.15);
}

/** Iso-lines of a value grid (regression plane) at `levels`. */
export function isoSegments(values: Float32Array, w: number, h: number, bounds: [number, number, number, number], levels: number[]): Float32Array {
  const out: number[] = [];
  const f = field(values.length);
  for (const lv of levels) {
    for (let i = 0; i < values.length; i++) f[i] = values[i] - lv;
    marching(f, w, h, bounds, out);
  }
  return tidy(out, Math.min((bounds[1] - bounds[0]) / w, (bounds[3] - bounds[2]) / h) * 0.15);
}
