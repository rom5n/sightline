// WebGL2 renderer for a 2-D plot: probability heatmap that morphs between engine frames
// on the GPU, instanced points with per-class shapes, and anti-aliased line layers
// (borders, ghost border, grid, paths). One instance per canvas.
import { backdrop, cssVar, parseColor } from "./loop";

export type View = { x0: number; x1: number; y0: number; y1: number };
export type HeatMode = "proba" | "value" | "clusters";
export type RGBA = [number, number, number, number];

export interface HeatInput {
  bounds: [number, number, number, number];
  w: number;
  h: number;
  k: number;
  /** class-major probabilities, k × h × w */
  planar: Float32Array;
}

interface HeatLayer extends HeatInput { tex: WebGLTexture[] }

export interface LineStyle { color: RGBA; width: number; dash?: number; alpha?: number }

export const FLAG = { test: 1, support: 2, outlier: 4, hollow: 8, dim: 16, selected: 32 } as const;

const VS_HEAT = `#version 300 es
in vec2 aPos;
uniform vec4 uView;
out vec2 vWorld;
void main(){ vWorld = aPos; vec2 c = ((aPos - uView.xy) * uView.zw) * 2.0 - 1.0; gl_Position = vec4(c, 0.0, 1.0); }`;

const FS_HEAT = `#version 300 es
precision highp float;
in vec2 vWorld;
uniform sampler2D uA0, uA1, uA2, uB0, uB1, uB2;
uniform vec4 uBA, uBB;
uniform float uMix, uAlpha;
uniform int uK, uMode, uHasA;
uniform vec3 uPal[10];
uniform vec2 uRange;
out vec4 o;
void main(){
  vec2 uvB = (vWorld - uBB.xz) / (uBB.yw - uBB.xz);
  vec4 b0 = texture(uB0, uvB), b1 = texture(uB1, uvB), b2 = texture(uB2, uvB);
  vec4 a0 = b0, a1 = b1, a2 = b2;
  if (uHasA == 1) {
    vec2 uvA = (vWorld - uBA.xz) / (uBA.yw - uBA.xz);
    if (all(greaterThanEqual(uvA, vec2(0.0))) && all(lessThanEqual(uvA, vec2(1.0)))) {
      a0 = texture(uA0, uvA); a1 = texture(uA1, uvA); a2 = texture(uA2, uvA);
    }
  }
  vec4 p0 = mix(a0, b0, uMix), p1 = mix(a1, b1, uMix), p2 = mix(a2, b2, uMix);
  float p[12] = float[12](p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w, p2.x, p2.y, p2.z, p2.w);
  if (uMode == 1) {
    float v = clamp((p[0] - uRange.x) / max(1e-9, uRange.y - uRange.x), 0.0, 1.0);
    vec3 col = mix(uPal[0], uPal[1], v);
    float a = (0.05 + 0.22 * abs(v - 0.5) * 2.0) * uAlpha;
    o = vec4(col * a, a);
    return;
  }
  int best = 0; float bv = -1.0;
  for (int i = 0; i < 12; i++) { if (i >= uK) break; if (p[i] > bv) { bv = p[i]; best = i; } }
  float kf = float(uK);
  float c = clamp((bv - 1.0 / kf) / (1.0 - 1.0 / kf), 0.0, 1.0);
  float a = uMode == 2 ? 0.11 * uAlpha * smoothstep(0.0, 0.35, c) : (0.03 + 0.25 * c) * uAlpha;
  vec3 col = uPal[best - (best / 10) * 10];
  o = vec4(col * a, a);
}`;

const VS_PTS = `#version 300 es
in vec2 aCorner;
in vec2 aPos; in vec2 aPrev; in float aCls; in float aPrevCls; in float aFlags; in float aDelay;
uniform vec4 uView; uniform vec2 uPx; uniform float uSize; uniform float uT; uniform float uDur; uniform int uAnim; uniform vec2 uCenter;
out vec2 vLocal; flat out float vCls; flat out float vPrevCls; flat out float vFlags;
float spring(float x){ x = clamp(x, 0.0, 1.0); return 1.0 - exp(-7.0 * x) * cos(10.0 * x); }
void main(){
  vec2 p = aPos;
  if (uAnim == 1) { p = mix(uCenter, aPos, spring((uT - aDelay) / uDur)); }
  else if (uAnim == 2) { float x = clamp(uT / uDur, 0.0, 1.0); p = mix(aPrev, aPos, 1.0 - pow(1.0 - x, 3.0)); }
  vec2 clip = ((p - uView.xy) * uView.zw) * 2.0 - 1.0;
  float sz = uSize + 12.0;
  gl_Position = vec4(clip + aCorner * sz / uPx, 0.0, 1.0);
  vLocal = aCorner * sz * 0.5;
  vCls = aCls; vPrevCls = aPrevCls; vFlags = aFlags;
}`;

const FS_PTS = `#version 300 es
precision highp float;
in vec2 vLocal; flat in float vCls; flat in float vPrevCls; flat in float vFlags;
uniform float uSize; uniform float uMorph; uniform vec3 uPal[10]; uniform vec3 uSolid; uniform vec3 uInk; uniform vec3 uInk3; uniform vec3 uWarn; uniform int uSingle;
out vec4 o;
float sdTri(vec2 p, float r){ const float k = 1.7320508; p.x = abs(p.x) - r; p.y = p.y + r / k; if (p.x + k * p.y > 0.0) p = vec2(p.x - k * p.y, -k * p.x - p.y) / 2.0; p.x -= clamp(p.x, -2.0 * r, 0.0); return -length(p) * sign(p.y); }
float sdBox(vec2 p, float b, float rr){ vec2 d = abs(p) - vec2(b - rr); return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0) - rr; }
float shape(float cls, vec2 p, float s){
  int k = int(mod(max(cls, 0.0), 4.0));
  if (uSingle == 1) k = 0;
  if (k == 0) return length(p) - s * 0.5;
  if (k == 1) return sdBox(p, s * 0.46, 1.5);
  if (k == 2) return sdTri(p - vec2(0.0, -s * 0.06), s * 0.62);
  return (abs(p.x) + abs(p.y)) * 0.70710678 - s * 0.40;
}
vec3 col(float cls){ if (cls < 0.0) return uInk3; int i = int(mod(cls, 10.0)); return uPal[i]; }
bool has(float f, float bit){ return mod(floor(f / bit), 2.0) == 1.0; }
void main(){
  float d = mix(shape(vPrevCls, vLocal, uSize), shape(vCls, vLocal, uSize), uMorph);
  vec3 fill = mix(col(vPrevCls), col(vCls), uMorph);
  float fa = clamp(0.5 - d, 0.0, 1.0);
  float oa = clamp(0.5 - (d - 1.5), 0.0, 1.0);
  vec3 c = mix(uSolid, fill, fa);
  float a = oa;
  if (has(vFlags, 8.0)) { // hollow (noise): ring only
    float ring = clamp(0.75 - abs(d + 0.6), 0.0, 1.0);
    c = uInk3; a = ring;
  }
  if (has(vFlags, 2.0)) { float r = clamp(0.9 - abs(d - 3.6), 0.0, 1.0); c = mix(c, uInk, r * (1.0 - fa)); a = max(a, r); }
  if (has(vFlags, 4.0)) { float r = clamp(1.1 - abs(d - 3.2), 0.0, 1.0); c = mix(c, uWarn, r * (1.0 - fa)); a = max(a, r); }
  if (has(vFlags, 32.0)) { float r = clamp(1.0 - abs(d - 4.0), 0.0, 1.0); c = mix(c, uInk, r * (1.0 - fa)); a = max(a, r); }
  if (has(vFlags, 16.0)) a *= 0.4;
  if (a <= 0.003) discard;
  o = vec4(c * a, a);
}`;

const VS_LINE = `#version 300 es
in vec2 aCorner; in vec2 aP0; in vec2 aP1;
uniform vec4 uView; uniform vec2 uPx; uniform float uWidth;
out float vSide; out float vAlong; flat out float vLen; flat out vec2 vDir;
vec2 px(vec2 w){ return ((w - uView.xy) * uView.zw) * uPx; }
void main(){
  vec2 a = px(aP0), b = px(aP1);
  vec2 d = b - a; float L = length(d);
  vec2 dir = L > 1e-4 ? d / L : vec2(1.0, 0.0);
  vec2 n = vec2(-dir.y, dir.x);
  float hw = uWidth * 0.5 + 1.0;
  float e = aCorner.x * 2.0 - 1.0; // -1 at a, +1 at b
  vec2 p = mix(a, b, aCorner.x) + n * aCorner.y * hw + dir * e * hw;
  vSide = aCorner.y * hw;
  vAlong = aCorner.x * L + e * hw;
  vLen = L;
  vDir = dir;
  gl_Position = vec4(p / uPx * 2.0 - 1.0, 0.0, 1.0);
}`;

// Each segment is drawn as a capsule (distance to the segment), so consecutive pieces of a
// polyline meet with round joins instead of notches.
const FS_LINE = `#version 300 es
precision highp float;
in float vSide; in float vAlong; flat in float vLen; flat in vec2 vDir;
uniform vec4 uColor; uniform float uWidth; uniform float uDash;
out vec4 o;
void main(){
  float t = vAlong < 0.0 ? vAlong : (vAlong > vLen ? vAlong - vLen : 0.0);
  float dist = length(vec2(t, vSide));
  float a = clamp(uWidth * 0.5 + 0.5 - dist, 0.0, 1.0) * uColor.a;
  // dash phase = position projected on the line direction: continuous across the pieces of a
  // contour, so dashes stay even instead of restarting per piece
  if (uDash > 0.0) { float s = mod(dot(gl_FragCoord.xy, vDir), uDash); if (s > uDash * 0.55) discard; }
  if (a <= 0.003) discard;
  o = vec4(uColor.rgb * a, a);
}`;

function compile(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const mk = (type: number, src: string) => {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "shader");
    return s;
  };
  const p = gl.createProgram()!;
  gl.attachShader(p, mk(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? "link");
  return p;
}

type Uni = Record<string, WebGLUniformLocation | null>;
function uniforms(gl: WebGL2RenderingContext, p: WebGLProgram, names: string[]): Uni {
  const u: Uni = {};
  for (const n of names) u[n] = gl.getUniformLocation(p, n);
  return u;
}

interface LineLayer { buf: WebGLBuffer; vao: WebGLVertexArrayObject; n: number; style: LineStyle; data: Float32Array; order: number }

export class PlotGL {
  readonly canvas: HTMLCanvasElement;
  private gl!: WebGL2RenderingContext;
  private heatProg!: WebGLProgram;
  private ptsProg!: WebGLProgram;
  private lineProg!: WebGLProgram;
  private uH!: Uni;
  private uP!: Uni;
  private uL!: Uni;
  private heatVao!: WebGLVertexArrayObject;
  private heatBuf!: WebGLBuffer;
  private ptsVao!: WebGLVertexArrayObject;
  private ptsBuf!: WebGLBuffer;
  private cornerBuf!: WebGLBuffer;
  private dummyTex!: WebGLTexture;
  lost = false;

  view: View = { x0: 0, x1: 1, y0: 0, y1: 1 };
  pxW = 1;
  pxH = 1;
  dpr = 1;

  // heat
  private heatA: HeatLayer | null = null;
  private heatB: HeatLayer | null = null;
  mix = 1;
  heatMode: HeatMode = "proba";
  heatAlpha = 1;
  valueRange: [number, number] = [0, 1];
  // points
  private ptsData: Float32Array = new Float32Array(0);
  nPts = 0;
  pointSize = 9;
  ptsAnim: 0 | 1 | 2 = 0;
  ptsT = 0;
  ptsDur = 0.25;
  morph = 1;
  center: [number, number] = [0, 0];
  singleColor = false;
  // lines
  private lines = new Map<string, LineLayer>();
  // colors
  private pal = new Float32Array(30);
  private texBuf = new Float32Array(0);
  private bg: RGBA = [0, 0, 0, 0];
  private solid: RGBA = [0, 0, 0, 1];
  private ink: RGBA = [1, 1, 1, 1];
  private ink3: RGBA = [.6, .6, .6, 1];
  private warn: RGBA = [1, .7, .3, 1];

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.init();
    canvas.addEventListener("webglcontextlost", (e) => {
      e.preventDefault();
      this.lost = true;
    });
    canvas.addEventListener("webglcontextrestored", () => {
      this.init();
      this.lost = false;
    });
  }

  private init() {
    // preserveDrawingBuffer: "Copy image" / "Save image as" then snapshot the last frame, not a cleared buffer
    const gl = this.canvas.getContext("webgl2", { antialias: false, premultipliedAlpha: true, alpha: true, powerPreference: "high-performance", preserveDrawingBuffer: true });
    if (!gl) throw new Error("WebGL2 unavailable");
    this.gl = gl;
    gl.getExtension("EXT_color_buffer_float");
    gl.getExtension("OES_texture_float_linear");
    this.heatProg = compile(gl, VS_HEAT, FS_HEAT);
    this.ptsProg = compile(gl, VS_PTS, FS_PTS);
    this.lineProg = compile(gl, VS_LINE, FS_LINE);
    this.uH = uniforms(gl, this.heatProg, ["uView", "uA0", "uA1", "uA2", "uB0", "uB1", "uB2", "uBA", "uBB", "uMix", "uAlpha", "uK", "uMode", "uHasA", "uPal", "uRange"]);
    this.uP = uniforms(gl, this.ptsProg, ["uView", "uPx", "uSize", "uT", "uDur", "uAnim", "uCenter", "uMorph", "uPal", "uSolid", "uInk", "uInk3", "uWarn", "uSingle"]);
    this.uL = uniforms(gl, this.lineProg, ["uView", "uPx", "uWidth", "uColor", "uDash"]);

    this.heatVao = gl.createVertexArray()!;
    this.heatBuf = gl.createBuffer()!;
    gl.bindVertexArray(this.heatVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.heatBuf);
    const la = gl.getAttribLocation(this.heatProg, "aPos");
    gl.enableVertexAttribArray(la);
    gl.vertexAttribPointer(la, 2, gl.FLOAT, false, 0, 0);

    this.cornerBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.cornerBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

    this.ptsVao = gl.createVertexArray()!;
    this.ptsBuf = gl.createBuffer()!;
    gl.bindVertexArray(this.ptsVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.cornerBuf);
    const pc = gl.getAttribLocation(this.ptsProg, "aCorner");
    gl.enableVertexAttribArray(pc);
    gl.vertexAttribPointer(pc, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.ptsBuf);
    const S = 8 * 4;
    const attr = (name: string, size: number, off: number) => {
      const l = gl.getAttribLocation(this.ptsProg, name);
      gl.enableVertexAttribArray(l);
      gl.vertexAttribPointer(l, size, gl.FLOAT, false, S, off * 4);
      gl.vertexAttribDivisor(l, 1);
    };
    attr("aPos", 2, 0);
    attr("aPrev", 2, 2);
    attr("aCls", 1, 4);
    attr("aPrevCls", 1, 5);
    attr("aFlags", 1, 6);
    attr("aDelay", 1, 7);
    gl.bindVertexArray(null);

    this.dummyTex = this.makeTex(1, 1, new Float32Array(4));
    // re-upload cached state after a context loss
    const a = this.heatA, b = this.heatB;
    this.heatA = a ? { ...a, tex: this.heatTex(a) } : null;
    this.heatB = b ? { ...b, tex: this.heatTex(b) } : null;
    if (this.nPts) this.uploadPoints();
    const old = [...this.lines.entries()];
    this.lines.clear();
    for (const [k, l] of old) this.setLines(k, l.data, l.style, l.order);
    this.refreshColors();
  }

  refreshColors(el?: Element) {
    for (let i = 0; i < 10; i++) {
      const c = parseColor(cssVar(`--c${i % 5}`, el));
      this.pal.set(c.slice(0, 3), i * 3);
    }
    this.solid = parseColor(cssVar("--solid", el));
    this.ink = parseColor(cssVar("--ink", el));
    this.ink3 = parseColor(cssVar("--ink3", el));
    this.warn = parseColor(cssVar("--warn", el));
    this.bg = backdrop(this.canvas);
  }

  resize(cssW: number, cssH: number, dpr: number) {
    this.dpr = dpr;
    const w = Math.max(1, Math.round(cssW * dpr));
    const h = Math.max(1, Math.round(cssH * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.pxW = w;
    this.pxH = h;
  }

  /** Upload k planar channels as ceil(k/4) RGBA textures (interleaved here, so frames keep one copy). */
  private heatTex(h: HeatInput): WebGLTexture[] {
    const n = h.w * h.h, out: WebGLTexture[] = [];
    if (this.texBuf.length < n * 4) this.texBuf = new Float32Array(n * 4);
    const buf = this.texBuf.subarray(0, n * 4);
    for (let t = 0; t * 4 < h.k; t++) {
      buf.fill(0);
      for (let c = 0; c < 4 && t * 4 + c < h.k; c++) {
        const src = h.planar.subarray((t * 4 + c) * n, (t * 4 + c + 1) * n);
        for (let i = 0; i < n; i++) buf[i * 4 + c] = src[i];
      }
      out.push(this.makeTex(h.w, h.h, buf));
    }
    return out;
  }

  private makeTex(w: number, h: number, data: Float32Array): WebGLTexture {
    const gl = this.gl;
    const t = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.FLOAT, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }

  private freeHeat(l: HeatLayer | null) {
    if (!l) return;
    for (const t of l.tex) this.gl.deleteTexture(t);
  }

  /** Push a new grid. The previous grid becomes the morph source (A), the new one the target (B). */
  setHeat(h: HeatInput | null, morph: boolean) {
    if (!h) {
      this.freeHeat(this.heatA);
      this.freeHeat(this.heatB);
      this.heatA = this.heatB = null;
      return;
    }
    const layer: HeatLayer = { ...h, tex: this.heatTex(h) };
    if (morph && this.heatB && this.heatB.k === h.k) {
      // if a morph is still running, the new start is wherever we are now: approximated by B
      this.freeHeat(this.heatA);
      this.heatA = this.heatB;
      this.mix = 0;
    } else {
      this.freeHeat(this.heatA);
      this.freeHeat(this.heatB);
      this.heatA = null;
      this.mix = 1;
    }
    this.heatB = layer;
  }

  get heat() {
    return { a: this.heatA, b: this.heatB };
  }

  /** points: interleaved [x, y, prevX, prevY, cls, prevCls, flags, delay] */
  setPointData(data: Float32Array, n: number) {
    this.ptsData = data;
    this.nPts = n;
    this.uploadPoints();
  }

  get pointData() {
    return this.ptsData;
  }

  uploadPoints() {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.ptsBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.ptsData.subarray(0, this.nPts * 8) as Float32Array<ArrayBuffer>, gl.DYNAMIC_DRAW);
  }

  setLines(name: string, segs: Float32Array | null, style: LineStyle, order = 0) {
    const gl = this.gl;
    let l = this.lines.get(name);
    if (!segs || segs.length < 4) {
      if (l) {
        gl.deleteBuffer(l.buf);
        gl.deleteVertexArray(l.vao);
        this.lines.delete(name);
      }
      return;
    }
    if (!l) {
      const vao = gl.createVertexArray()!;
      const buf = gl.createBuffer()!;
      gl.bindVertexArray(vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.cornerBuf);
      const c = gl.getAttribLocation(this.lineProg, "aCorner");
      gl.enableVertexAttribArray(c);
      gl.vertexAttribPointer(c, 2, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      const p0 = gl.getAttribLocation(this.lineProg, "aP0");
      const p1 = gl.getAttribLocation(this.lineProg, "aP1");
      gl.enableVertexAttribArray(p0);
      gl.vertexAttribPointer(p0, 2, gl.FLOAT, false, 16, 0);
      gl.vertexAttribDivisor(p0, 1);
      gl.enableVertexAttribArray(p1);
      gl.vertexAttribPointer(p1, 2, gl.FLOAT, false, 16, 8);
      gl.vertexAttribDivisor(p1, 1);
      gl.bindVertexArray(null);
      l = { buf, vao, n: 0, style, data: segs, order };
      this.lines.set(name, l);
    }
    l.style = style;
    l.order = order;
    l.data = segs;
    l.n = Math.floor(segs.length / 4);
    gl.bindBuffer(gl.ARRAY_BUFFER, l.buf);
    gl.bufferData(gl.ARRAY_BUFFER, segs, gl.DYNAMIC_DRAW);
  }

  private viewUniform(): [number, number, number, number] {
    const v = this.view;
    return [v.x0, v.y0, 1 / (v.x1 - v.x0 || 1), 1 / (v.y1 - v.y0 || 1)];
  }

  render(layers: { heat: boolean; points: boolean; linesBelow?: string[]; linesAbove?: string[] }) {
    if (this.lost) return;
    const gl = this.gl;
    gl.viewport(0, 0, this.pxW, this.pxH);
    gl.clearColor(this.bg[0], this.bg[1], this.bg[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    const vu = this.viewUniform();

    if (layers.heat && this.heatB) {
      const b = this.heatB;
      const a = this.heatA;
      gl.useProgram(this.heatProg);
      gl.uniform4f(this.uH.uView, ...vu);
      const [x0, x1, y0, y1] = b.bounds;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.heatBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([x0, y0, x1, y0, x0, y1, x1, y1]), gl.DYNAMIC_DRAW);
      const bind = (unit: number, t: WebGLTexture | undefined, name: string) => {
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(gl.TEXTURE_2D, t ?? this.dummyTex);
        gl.uniform1i(this.uH[name], unit);
      };
      bind(0, b.tex[0], "uB0");
      bind(1, b.tex[1], "uB1");
      bind(2, b.tex[2], "uB2");
      bind(3, a?.tex[0], "uA0");
      bind(4, a?.tex[1], "uA1");
      bind(5, a?.tex[2], "uA2");
      gl.uniform4f(this.uH.uBB, ...b.bounds);
      gl.uniform4f(this.uH.uBA, ...(a ? a.bounds : b.bounds));
      gl.uniform1i(this.uH.uHasA, a && this.mix < 1 ? 1 : 0);
      gl.uniform1f(this.uH.uMix, this.mix);
      gl.uniform1f(this.uH.uAlpha, this.heatAlpha);
      gl.uniform1i(this.uH.uK, b.k);
      gl.uniform1i(this.uH.uMode, this.heatMode === "value" ? 1 : this.heatMode === "clusters" ? 2 : 0);
      gl.uniform3fv(this.uH.uPal, this.pal);
      gl.uniform2f(this.uH.uRange, this.valueRange[0], this.valueRange[1]);
      gl.bindVertexArray(this.heatVao);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    const drawLines = (names: string[] | undefined) => {
      if (!names) return;
      gl.useProgram(this.lineProg);
      gl.uniform4f(this.uL.uView, ...vu);
      gl.uniform2f(this.uL.uPx, this.pxW, this.pxH);
      for (const n of names) {
        const l = this.lines.get(n);
        if (!l || !l.n) continue;
        const c = l.style.color;
        gl.uniform4f(this.uL.uColor, c[0], c[1], c[2], c[3] * (l.style.alpha ?? 1));
        gl.uniform1f(this.uL.uWidth, l.style.width * this.dpr);
        gl.uniform1f(this.uL.uDash, (l.style.dash ?? 0) * this.dpr);
        gl.bindVertexArray(l.vao);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, l.n);
      }
    };
    drawLines(layers.linesBelow);

    if (layers.points && this.nPts) {
      gl.useProgram(this.ptsProg);
      const u = this.uP;
      gl.uniform4f(u.uView, ...vu);
      gl.uniform2f(u.uPx, this.pxW, this.pxH);
      gl.uniform1f(u.uSize, this.pointSize * this.dpr);
      gl.uniform1f(u.uT, this.ptsT);
      gl.uniform1f(u.uDur, this.ptsDur);
      gl.uniform1i(u.uAnim, this.ptsAnim);
      gl.uniform2f(u.uCenter, this.center[0], this.center[1]);
      gl.uniform1f(u.uMorph, this.morph);
      gl.uniform3fv(u.uPal, this.pal);
      gl.uniform3f(u.uSolid, this.solid[0], this.solid[1], this.solid[2]);
      gl.uniform3f(u.uInk, this.ink[0], this.ink[1], this.ink[2]);
      gl.uniform3f(u.uInk3, this.ink3[0], this.ink3[1], this.ink3[2]);
      gl.uniform3f(u.uWarn, this.warn[0], this.warn[1], this.warn[2]);
      gl.uniform1i(u.uSingle, this.singleColor ? 1 : 0);
      gl.bindVertexArray(this.ptsVao);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.nPts);
    }
    drawLines(layers.linesAbove);
    gl.bindVertexArray(null);
  }

  /** Free the context only if the canvas really left the page (StrictMode remounts keep it). */
  dispose() {
    setTimeout(() => {
      if (!this.canvas.isConnected) this.gl.getExtension("WEBGL_lose_context")?.loseContext();
    }, 0);
  }
}
