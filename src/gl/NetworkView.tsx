import { useTranslation } from "react-i18next";
// MLP network graph: WebGL2 edges (thickness/opacity = |weight|, animated as weights
// change; a light wave travels layer by layer on a forward pass) + DOM neurons whose
// fill follows activation.
import { useEffect, useLayoutEffect, useRef } from "react";
import type { NetworkData } from "../engine/types";
import { fmt } from "../ui/controls";
import { backdrop, cssVar, parseColor, reducedMotion, request, subscribe } from "./loop";

const MAXN = 16;

const VS = `#version 300 es
in vec2 aCorner; in vec2 aP0; in vec2 aP1; in float aW; in float aSign; in float aLayer;
uniform vec2 uPx; uniform float uScale;
out float vSide; out float vW; out float vSign; out float vLayer; out float vT; flat out float vLen;
void main(){
  vec2 a = aP0 * uPx, b = aP1 * uPx; vec2 d = b - a; float L = length(d); vec2 dir = L > 0.0 ? d / L : vec2(1, 0); vec2 n = vec2(-dir.y, dir.x);
  float w = (0.6 + aW * 2.6) * uScale; float hw = w * 0.5 + 1.0;
  vec2 p = mix(a, b, aCorner.x) + n * aCorner.y * hw;
  vSide = aCorner.y * hw; vW = w; vSign = aSign; vLayer = aLayer; vT = aCorner.x; vLen = L;
  gl_Position = vec4(p / uPx * 2.0 - 1.0, 0.0, 1.0); gl_Position.y = -gl_Position.y;
}`;
const FS = `#version 300 es
precision highp float;
in float vSide; in float vW; in float vSign; in float vLayer; in float vT; flat in float vLen;
uniform vec3 uPos; uniform vec3 uNeg; uniform float uWave; uniform float uAlpha; uniform float uDash; uniform float uScale;
out vec4 o;
void main(){
  float a = clamp(vW * 0.5 + 0.5 - abs(vSide), 0.0, 1.0);
  vec3 c = vSign > 0.0 ? uPos : uNeg;
  // dashes measured along the edge, so every negative edge gets the same even pattern
  if (vSign < 0.0 && uDash > 0.0) { if (mod(vT * vLen, 6.0 * uScale) > 3.3 * uScale) discard; }
  float alpha = (0.12 + vW * 0.16) * uAlpha;
  if (uWave >= 0.0) {
    float pos = vLayer + vT;
    float g = exp(-pow((pos - uWave) * 3.0, 2.0));
    float passed = pos < uWave ? 1.0 : 0.25;
    alpha = alpha * passed + g * 0.9;
    c = mix(c, vec3(1.0), g * 0.35);
  }
  a *= clamp(alpha, 0.0, 1.0);
  o = vec4(c * a, a);
}`;

function pickIdx(n: number): number[] {
  if (n <= MAXN) return Array.from({ length: n }, (_, i) => i);
  const half = MAXN / 2;
  return [...Array.from({ length: half }, (_, i) => i), ...Array.from({ length: half }, (_, i) => n - half + i)];
}

export interface NetworkViewProps {
  net: NetworkData | null;
  acts?: number[][] | null; // per layer activations (input first)
  wave?: number | null; // -1 off; else 0..L-1 position of the light front
  shownLayers?: number; // decide: layers revealed so far (others faded)
  currentLayer?: number | null;
  values?: boolean;
  featureNames?: string[];
  outLabel?: string;
}

export function NetworkView(props: NetworkViewProps) {
  const { t } = useTranslation();
  const box = useRef<HTMLDivElement>(null);
  const cv = useRef<HTMLCanvasElement>(null);
  const nodesRef = useRef<HTMLDivElement>(null);
  const P = useRef(props);
  P.current = props;
  const kick = useRef<() => void>(() => undefined);

  useLayoutEffect(() => {
    const c = cv.current!;
    const gl = c.getContext("webgl2", { premultipliedAlpha: true, antialias: false, preserveDrawingBuffer: true });
    if (!gl) return;
    const mk = (t: number, s: string) => { const sh = gl.createShader(t)!; gl.shaderSource(sh, s); gl.compileShader(sh); return sh; };
    const prog = gl.createProgram()!;
    gl.attachShader(prog, mk(gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, mk(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    const U = (n: string) => gl.getUniformLocation(prog, n);
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    const corner = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, corner);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, -1, 1, -1, 0, 1, 1, 1]), gl.STATIC_DRAW);
    const la = gl.getAttribLocation(prog, "aCorner");
    gl.enableVertexAttribArray(la);
    gl.vertexAttribPointer(la, 2, gl.FLOAT, false, 0, 0);
    const inst = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, inst);
    const S = 7 * 4;
    const at = (n: string, size: number, off: number) => { const l = gl.getAttribLocation(prog, n); gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, size, gl.FLOAT, false, S, off * 4); gl.vertexAttribDivisor(l, 1); };
    at("aP0", 2, 0); at("aP1", 2, 2); at("aW", 1, 4); at("aSign", 1, 5); at("aLayer", 1, 6);
    gl.bindVertexArray(null);

    let W = 1, H = 1;
    let prevW: number[] | null = null;
    let nextW: number[] | null = null;
    let wT0 = 0;
    let layoutKey = "";
    let pos: { x: number; y: number }[][] = [];
    let edges: { l: number; i: number; j: number; a: number; b: number }[] = [];

    const layout = () => {
      const net = P.current.net;
      if (!net) { pos = []; edges = []; return; }
      const L = net.sizes.length;
      const idxs = net.sizes.map(pickIdx);
      pos = idxs.map((ix, l) => {
        const n = ix.length;
        const x = (l + 0.5) / L;
        const gap = n === 1 ? 0 : Math.min(0.11, 0.84 / (n - 1));
        return ix.map((_, k) => ({ x, y: 0.5 + (k - (n - 1) / 2) * gap }));
      });
      edges = [];
      for (let l = 0; l < L - 1; l++) {
        for (let a = 0; a < idxs[l].length; a++) for (let b = 0; b < idxs[l + 1].length; b++) edges.push({ l, i: idxs[l][a], j: idxs[l + 1][b], a, b });
      }
    };

    const weightsOf = (net: NetworkData) => edges.map((e) => net.coefs[e.l].data[e.i * net.coefs[e.l].shape[1] + e.j]);

    const loop = (now: number): boolean => {
      if (!box.current) return false;
      const p = P.current;
      const dpr = devicePixelRatio || 1;
      const pw = Math.round(W * dpr), ph = Math.round(H * dpr);
      if (c.width !== pw || c.height !== ph) { c.width = pw; c.height = ph; }
      gl.viewport(0, 0, pw, ph);
      const bg = backdrop(c);
      gl.clearColor(bg[0], bg[1], bg[2], 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      const net = p.net;
      if (!net || !edges.length) { placeNodes(); return false; }
      const k = prevW && nextW && prevW.length === nextW.length && !reducedMotion() ? Math.min(1, (now - wT0) / 220) : 1;
      const cur = nextW!.map((w, i) => (prevW && k < 1 ? prevW[i] + (w - prevW[i]) * k : w));
      let maxAbs = 1e-6;
      for (const w of cur) maxAbs = Math.max(maxAbs, Math.abs(w));
      const data = new Float32Array(edges.length * 7);
      const acts = p.acts;
      edges.forEach((e, n) => {
        const a = pos[e.l][e.a], b = pos[e.l + 1][e.b];
        let w = Math.abs(cur[n]) / maxAbs;
        if (acts && e.l > 0 && acts[e.l] && acts[e.l][e.i] === 0) w *= 0.25;
        if (p.shownLayers != null && e.l + 1 >= p.shownLayers) w *= 0.15;
        data.set([a.x, a.y, b.x, b.y, w, Math.sign(cur[n]) || 1, e.l], n * 7);
      });
      gl.bindBuffer(gl.ARRAY_BUFFER, inst);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.useProgram(prog);
      const pc = parseColor(cssVar("--acc-t", box.current!));
      const nc = parseColor(cssVar("--ink2", box.current!));
      gl.uniform2f(U("uPx"), pw, ph);
      gl.uniform1f(U("uScale"), dpr);
      gl.uniform3f(U("uPos"), pc[0], pc[1], pc[2]);
      gl.uniform3f(U("uNeg"), nc[0], nc[1], nc[2]);
      gl.uniform1f(U("uWave"), p.wave ?? -1);
      gl.uniform1f(U("uAlpha"), 1.6);
      gl.uniform1f(U("uDash"), dpr * 1);
      gl.bindVertexArray(vao);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, edges.length);
      gl.bindVertexArray(null);
      placeNodes();
      return k < 1;
    };

    const placeNodes = () => {
      const host = nodesRef.current;
      if (!host) return;
      const p = P.current;
      const net = p.net;
      const flat: { x: number; y: number; l: number; i: number; v: number | null; fill: number; label: string | null }[] = [];
      if (net) {
        const idxs = net.sizes.map(pickIdx);
        idxs.forEach((ix, l) => {
          const a = p.acts?.[l];
          let mx = 1e-9;
          if (a) for (const v of a) mx = Math.max(mx, Math.abs(v));
          ix.forEach((i, k) => {
            const v = a ? a[i] : null;
            let fill = 0;
            if (v != null) fill = l === 0 ? Math.min(1, Math.abs(v) / mx * 0.8 + 0.2) : l === net.sizes.length - 1 ? Math.min(1, Math.abs(v)) : Math.min(1, Math.abs(v) / mx);
            let label: string | null = null;
            if (p.values && v != null) label = l === 0 ? `${p.featureNames?.[i] ?? "x" + (i + 1)} = ${fmt(v, 2)}` : l === net.sizes.length - 1 ? `${p.outLabel ?? "P"} = ${fmt(v, 2)}` : fmt(v, 2);
            flat.push({ ...pos[l][k], l, i, v, fill, label });
          });
        });
      }
      while (host.children.length < flat.length * 2) {
        const d = document.createElement("div");
        d.className = "nr";
        d.appendChild(document.createElement("i"));
        const lab = document.createElement("span");
        lab.className = "nv";
        host.appendChild(d);
        host.appendChild(lab);
      }
      const big = Math.max(...(net?.sizes ?? [1]).map((n) => Math.min(n, MAXN))) > 10 ? 20 : P.current.values ? 34 : 24;
      for (let n = 0; n < host.children.length / 2; n++) {
        const el = host.children[n * 2] as HTMLDivElement;
        const lab = host.children[n * 2 + 1] as HTMLSpanElement;
        const f = flat[n];
        if (!f) { el.style.display = "none"; lab.style.display = "none"; continue; }
        el.style.display = "";
        el.style.width = el.style.height = `${big}px`;
        el.style.transform = `translate(${f.x * W - big / 2}px, ${f.y * H - big / 2}px)`;
        (el.firstChild as HTMLElement).style.opacity = String(f.fill);
        const shown = p.shownLayers == null || f.l < p.shownLayers;
        el.className = `nr${f.l > 0 && f.v === 0 ? " zero" : ""}${shown ? "" : " off"}${p.currentLayer === f.l ? " cur" : ""}`;
        if (f.label && shown) {
          lab.style.display = "";
          lab.textContent = f.label;
          lab.className = `nv${f.l > 0 && f.v === 0 ? " z" : ""}`;
          lab.style.transform = f.l === 0 ? `translate(${f.x * W - big / 2 - 8}px, ${f.y * H}px) translate(-100%, -50%)` : `translate(${f.x * W + big / 2 + 4}px, ${f.y * H}px) translateY(-50%)`;
        } else lab.style.display = "none";
      }
    };

    const un = subscribe(loop);
    kick.current = () => {
      const net = P.current.net;
      const key = net ? net.sizes.join("-") : "";
      if (key !== layoutKey) {
        layoutKey = key;
        layout();
        prevW = null;
      }
      if (net) {
        const w = weightsOf(net);
        prevW = nextW && nextW.length === w.length ? nextW : null;
        nextW = w;
        wT0 = performance.now();
      }
      request(loop);
    };
    const ro = new ResizeObserver(() => {
      if (!box.current) return;
      const r = box.current.getBoundingClientRect();
      W = Math.max(1, r.width);
      H = Math.max(1, r.height);
      request(loop);
    });
    ro.observe(box.current!);
    kick.current();
    return () => { un(); ro.disconnect(); };
  }, []);

  useEffect(() => kick.current(), [props.net, props.acts, props.wave, props.shownLayers, props.currentLayer, props.values]);

  return (
    <div ref={box} style={{ position: "relative", width: "100%", height: "100%" }} role="img" aria-label={t("net.graphLabel")}>
      <canvas ref={cv} style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }} />
      <div ref={nodesRef} style={{ position: "absolute", inset: 0, pointerEvents: "none" }} />
    </div>
  );
}
