// Centre visuals for "How it decides", one per model family.
import { motion, useReducedMotion } from "motion/react";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { useSession } from "../../engine/store";
import type { NetworkData, TreeData } from "../../engine/types";
import { NetworkView } from "../../gl/NetworkView";
import { circleSegs, PlotView, type PlotExtra, type PlotLabel } from "../../gl/PlotView";
import { backdrop, css, cssVar, parseColor } from "../../gl/loop";
import { fmt } from "../../ui/controls";
import { TreeView } from "../workspace/TreeView";

type D = Record<string, unknown>;
interface Term { name: string; x: number; xm: number; w: number; contrib: number }
const SPR = { type: "spring", stiffness: 300, damping: 30 } as const;

function blk(step: number, n: number | number[], cur?: number[]) {
  const ns = Array.isArray(n) ? n : [n];
  const first = Math.min(...ns);
  const isCur = (cur ?? ns).includes(step);
  return `blk${step < first ? " off" : ""}${isCur ? " cur" : ""}`;
}
const col = (name: string): [number, number, number, number] => parseColor(cssVar(name));

export function DecideVisual({ family, data, step, n, onPick, pid }: { family: string; data: D; step: number; n: number; onPick: (id: number) => void; pid: number | null }) {
  switch (family) {
    case "logistic": case "sgd": return <LinearVis d={data} step={step} onPick={onPick} pid={pid} />;
    case "linear_reg": return <RegVis d={data} step={step} onPick={onPick} pid={pid} />;
    case "tree": return <TreeVis d={data} step={step} n={n} onPick={onPick} pid={pid} />;
    case "kmeans": return <KMeansVis d={data} step={step} onPick={onPick} pid={pid} />;
    case "mlp": return <MlpVis d={data} step={step} n={n} onPick={onPick} pid={pid} />;
    case "forest": return <ForestVis d={data} step={step} onPick={onPick} pid={pid} />;
    case "boost": return <BoostVis d={data} step={step} onPick={onPick} pid={pid} />;
    case "ada": return <AdaVis d={data} step={step} onPick={onPick} pid={pid} />;
    case "catboost": return <CatBoostVis d={data} step={step} onPick={onPick} pid={pid} />;
    case "knn": return <KnnVis d={data} step={step} onPick={onPick} pid={pid} />;
    case "svc": return <SvcVis d={data} step={step} onPick={onPick} pid={pid} />;
    case "nb": return <NbVis d={data} step={step} onPick={onPick} pid={pid} />;
    case "dbscan": return <DbscanVis d={data} step={step} onPick={onPick} pid={pid} />;
    case "pca": return <PcaVis d={data} step={step} onPick={onPick} pid={pid} />;
  }
  return null;
}

interface VP { d: D; step: number; onPick: (id: number) => void; pid: number | null }

function PointPlot({ pid, onPick, extra, labels, ringSet, title, hint, heat = true, children }: {
  pid: number | null; onPick: (id: number) => void; extra?: () => PlotExtra[]; labels?: () => PlotLabel[]; ringSet?: Set<number> | null;
  title?: string; hint?: string; heat?: boolean; children?: ReactNode;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10, minHeight: 0, minWidth: 0 }}>
      {(title || hint) && <div className="prow"><span className="lbl">{title}</span><span className="hint">{hint}</span></div>}
      <div style={{ position: "relative", flex: 1, minHeight: 220 }}>
        <PlotView variant="decide" highlightId={pid} dim onPick={onPick} extra={extra} labels={labels} ringSet={ringSet} heat={heat} axes={false} border>{children}</PlotView>
      </div>
    </div>
  );
}

function Bars({ terms, intercept, total, step, from, cls1 }: { terms: Term[]; intercept: number; total: number; step: number; from: number; cls1?: boolean }) {
  const { t } = useTranslation();
  const vals = [...terms.map((x) => x.contrib), intercept];
  const scale = Math.max(1e-9, vals.reduce((a, b) => a + Math.abs(b), 0), Math.abs(total));
  let acc = 0;
  const pos = vals.map((v) => {
    const start = acc;
    acc += v;
    return { left: Math.min(start, acc), width: Math.abs(v) };
  });
  const minL = Math.min(0, ...pos.map((p) => p.left));
  const L = (x: number) => ((x - minL) / scale) * 100;
  const c = (v: number) => (v >= 0 ? (cls1 === false ? "var(--c0)" : "var(--c1)") : "var(--c0)");
  return (
    <>
      {terms.slice(0, 8).map((x, i) => (
        <div className={`wfr ${blk(step, from)}`} key={x.name}>
          <span className="mono" style={{ fontSize: 13 }}>{x.name} <span style={{ color: "var(--ink)" }}>{fmt(x.xm, 2)}</span> × <span className={blk(step, from + 1)}>{fmt(x.w, 2)}</span></span>
          <div className="wbar"><motion.i className={blk(step, from + 1)} style={{ left: `${L(pos[i].left)}%`, width: `${(pos[i].width / scale) * 100}%`, background: c(x.contrib) }}
            initial={{ scaleX: 0 }} animate={{ scaleX: step >= from + 1 ? 1 : 0 }} transition={SPR} /></div>
          <span className={`mono ${blk(step, from + 1)}`} style={{ textAlign: "right" }}>{x.contrib >= 0 ? "+" : ""}{fmt(x.contrib, 2)}</span>
        </div>
      ))}
      <div className={`wfr ${blk(step, from + 1)}`}>
        <span className="mono" style={{ fontSize: 13 }}>{t("decide.intercept")}</span>
        <div className="wbar"><motion.i style={{ left: `${L(pos[vals.length - 1].left)}%`, width: `${(pos[vals.length - 1].width / scale) * 100}%`, background: c(intercept) }} initial={{ scaleX: 0 }} animate={{ scaleX: step >= from + 1 ? 1 : 0 }} transition={SPR} /></div>
        <span className="mono" style={{ textAlign: "right" }}>{intercept >= 0 ? "+" : ""}{fmt(intercept, 2)}</span>
      </div>
      <div className={`wfr ${blk(step, from + 2)}`} style={{ borderTop: "1px solid var(--line2)", paddingTop: 6, minHeight: 40 }}>
        <span className="mono" style={{ fontSize: 13, fontWeight: 600 }}>{t("decide.sumZ")}</span>
        <div className="wbar"><motion.i style={{ left: `${L(Math.min(0, total))}%`, width: `${(Math.abs(total) / scale) * 100}%`, background: "var(--ink)" }} initial={{ scaleX: 0 }} animate={{ scaleX: step >= from + 2 ? 1 : 0 }} transition={SPR} /></div>
        <span className="mono" style={{ textAlign: "right", fontWeight: 600, fontSize: 15 }}>{fmt(total, 2)}</span>
      </div>
    </>
  );
}

function SigmoidChart({ z, p, cls, onScore }: { z: number; p: number; cls: number; onScore?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const reduce = useReducedMotion();
  useEffect(() => {
    const c = ref.current!;
    const ctx = c.getContext("2d")!;
    let raf = 0;
    const t0 = performance.now();
    const draw = (now: number) => {
      const r = c.getBoundingClientRect();
      const dpr = devicePixelRatio || 1;
      c.width = r.width * dpr; c.height = r.height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const W = r.width, H = r.height, pl = 34, pb = 20, pt = 6;
      ctx.fillStyle = css(backdrop(c));
      ctx.fillRect(0, 0, W, H);
      const X = (v: number) => pl + ((v + 6) / 12) * (W - pl - 8);
      const Y = (v: number) => pt + (1 - v) * (H - pt - pb);
      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = cssVar("--gl"); ctx.lineWidth = 1;
      for (const v of [0, 0.5, 1]) { ctx.beginPath(); ctx.moveTo(pl, Y(v)); ctx.lineTo(W - 8, Y(v)); ctx.stroke(); }
      ctx.beginPath(); ctx.moveTo(X(0), pt); ctx.lineTo(X(0), H - pb); ctx.stroke();
      ctx.font = `500 12px ${cssVar("--mono")}`; ctx.fillStyle = cssVar("--ink3"); ctx.textAlign = "right"; ctx.textBaseline = "middle";
      ctx.fillText("1.0", pl - 6, Y(1)); ctx.fillText("0.5", pl - 6, Y(0.5)); ctx.fillText("0.0", pl - 6, Y(0));
      ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillText("−6", X(-6), H - pb + 4); ctx.fillText("0", X(0), H - pb + 4); ctx.fillText("6", X(6), H - pb + 4);
      ctx.strokeStyle = cssVar("--acc-t"); ctx.lineWidth = 2.5; ctx.beginPath();
      for (let v = -6; v <= 6.001; v += 0.1) { const y = 1 / (1 + Math.exp(-v)); v === -6 ? ctx.moveTo(X(v), Y(y)) : ctx.lineTo(X(v), Y(y)); }
      ctx.stroke();
      const k = reduce ? 1 : Math.min(1, (now - t0) / 500);
      const e = 1 - Math.pow(1 - k, 3);
      const zc = Math.max(-6, Math.min(6, z)) * e;
      const pc = 0.5 + (p - 0.5) * e;
      ctx.setLineDash([4, 4]); ctx.strokeStyle = cssVar("--ink2"); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(X(zc), H - pb); ctx.lineTo(X(zc), Y(pc)); ctx.lineTo(pl, Y(pc)); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = cssVar(`--c${cls % 5}`);
      ctx.beginPath(); ctx.arc(X(zc), Y(pc), 6, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = cssVar("--ink"); ctx.textAlign = "left"; ctx.textBaseline = "bottom";
      ctx.fillText(`${onScore ? "z" : "P"} = ${fmt(onScore ? z : p, 2)}`, X(zc) + 10, Y(pc) - 2);
      if (k < 1) raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [z, p, cls, reduce, onScore]);
  return <canvas ref={ref} style={{ width: "100%", height: "100%", display: "block" }} role="img" aria-label={`sigmoid of ${fmt(z, 2)} is ${fmt(p, 2)}`} />;
}

function Decision({ step, at, text, cls, label }: { step: number; at: number; text: string; cls: number; label: string }) {
  return (
    <div className={`dcard ${blk(step, at)}${step === at ? " cur" : ""}`} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}><span style={{ fontSize: 14, fontWeight: 600 }}>{text}</span></div>
      <motion.span className="tag" style={{ height: 34, padding: "0 14px", fontSize: 14, color: "var(--ink)" }} animate={{ scale: step === at ? [1, 1.06, 1] : 1 }} transition={{ duration: 0.35 }}>
        <span className={`dot k${cls % 5} s${cls % 4}`} />{label}
      </motion.span>
    </div>
  );
}

// ------------------------------------------------------------------ linear / logistic / sgd

function LinearVis({ d, step, onPick, pid }: VP) {
  const { t } = useTranslation();
  const classes = useSession((s) => s.dataset?.classes ?? []);
  const pred = d.pred as number;
  if (!d.binary) {
    const per = d.per_class as { cls: number; z: number }[];
    const pr = (d.proba as number[]) ?? [];
    const zs = per.map((c) => c.z);
    const mx = Math.max(1e-9, ...zs.map(Math.abs));
    return (
      <div className="dec-body" style={{ gridTemplateColumns: "minmax(0, .8fr) minmax(0, 1fr)" }}>
        <PointPlot pid={pid} onPick={onPick} title={t("decide.where")} />
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div className={`dcard ${blk(step, 2)}${step === 2 ? " cur" : ""}`}>
            <span style={{ fontSize: 14, fontWeight: 600 }}>{t("decide.scorePer")}</span>
            {per.map((c) => (
              <div className="wfr" key={c.cls}><span style={{ display: "flex", alignItems: "center", gap: 8 }}><span className={`dot k${c.cls % 5} s${c.cls % 4}`} />{classes[c.cls]}</span>
                <div className="wbar"><motion.i style={{ left: c.z >= 0 ? "50%" : `${50 - (Math.abs(c.z) / mx) * 50}%`, width: `${(Math.abs(c.z) / mx) * 50}%`, background: `var(--c${c.cls % 5})` }} initial={{ scaleX: 0 }} animate={{ scaleX: step >= 2 ? 1 : 0 }} transition={SPR} /></div>
                <span className="mono" style={{ textAlign: "right" }}>{fmt(c.z, 2)}</span></div>
            ))}
          </div>
          <div className={`dcard ${blk(step, 3)}${step === 3 ? " cur" : ""}`}>
            <span style={{ fontSize: 14, fontWeight: 600 }}>{t("decide.softmax")}</span>
            {pr.length > 0 && <div style={{ display: "flex", height: 22, borderRadius: 5, overflow: "hidden", gap: 2, marginTop: 8 }}>{pr.map((v, i) => <motion.div key={i} className={`k${i % 5}`} animate={{ flexGrow: step >= 3 ? v : 1 / pr.length }} style={{ flexBasis: 0 }} transition={SPR} />)}</div>}
          </div>
          <Decision step={step} at={4} text={t("decide.pickBiggest")} cls={pred} label={`${classes[pred]} · ${Math.round((pr[pred] ?? 0) * 100)}%`} />
        </div>
      </div>
    );
  }
  const terms = d.terms as Term[];
  const z = d.z as number;
  const prob = (d.prob as number | undefined) ?? null;
  const sign = d.loss != null && prob == null;
  return (
    <div className="dec-body" style={{ gridTemplateColumns: "minmax(0, .8fr) minmax(0, 1fr)" }}>
      <PointPlot pid={pid} onPick={onPick} title={t("decide.where")} hint={t("decide.borderZ")} />
      <div style={{ display: "flex", flexDirection: "column", gap: 14, minHeight: 0 }}>
        <div className={`dcard ${step <= 3 ? "blk cur" : "blk"}`} style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <div className="prow"><span style={{ fontSize: 14, fontWeight: 600 }}>{t("decide.weighted")}</span><span className="mono hint">z = w·x + b</span></div>
          <Bars terms={terms} intercept={d.intercept as number} total={z} step={step} from={1} />
          <span className="hint">{t("decide.barsHint", { b: classes[1], a: classes[0] })}</span>
        </div>
        <div className={`dcard ${blk(step, 4)}${step === 4 ? " cur" : ""}`} style={{ display: "flex", flexDirection: "column", gap: 8, flex: 1, minHeight: 170 }}>
          <div className="prow"><span style={{ fontSize: 14, fontWeight: 600 }}>{sign ? t("decide.signTitle") : t("decide.sigTitle")}</span><span className="mono hint">{sign ? "sign(z)" : `P(${classes[1]}) = 1 / (1 + e^−z)`}</span></div>
          <div style={{ position: "relative", flex: 1, minHeight: 110 }}>
            {step >= 4 && <SigmoidChart z={z} p={prob ?? 1 / (1 + Math.exp(-z))} cls={pred} onScore={sign} />}
          </div>
        </div>
        <Decision step={step} at={5} text={sign ? t("decide.signRule", { z: fmt(z, 2), cls: classes[pred] }) : t("decide.probRule", { p: fmt(prob, 2), cls: classes[pred] })} cls={pred}
          label={sign ? classes[pred] : `${classes[pred]} · ${Math.round((pred === 1 ? prob ?? 0 : 1 - (prob ?? 0)) * 100)}%`} />
      </div>
    </div>
  );
}

function RegVis({ d, step, onPick, pid }: VP) {
  const { t } = useTranslation();
  const terms = d.terms as Term[];
  const y = d.prediction as number;
  const truth = d.true as number;
  return (
    <div className="dec-body" style={{ gridTemplateColumns: "minmax(0, .8fr) minmax(0, 1fr)" }}>
      <PointPlot pid={pid} onPick={onPick} title={t("decide.where")} heat />
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <div className={`dcard ${step <= 3 ? "blk cur" : "blk"}`} style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <div className="prow"><span style={{ fontSize: 14, fontWeight: 600 }}>{t("decide.weighted")}</span><span className="mono hint">ŷ = w·x + b</span></div>
          <Bars terms={terms} intercept={d.intercept as number} total={y} step={step} from={1} />
        </div>
        <div className={`dcard ${blk(step, 4)}${step === 4 ? " cur" : ""}`}>
          <div className="prow"><span style={{ fontSize: 14, fontWeight: 600 }}>{t("decide.compare")}</span></div>
          <div className="mono" style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: "6px 14px", marginTop: 8 }}>
            <span className="hint">{t("decide.predicted")}</span><span>{fmt(y, 2)}</span>
            <span className="hint">{t("decide.real")}</span><span>{fmt(truth, 2)}</span>
            <span className="hint">{t("decide.error")}</span><span style={{ color: "var(--warn)" }}>{fmt(truth - y, 2)}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ tree

function TreeVis({ d, step, n, onPick, pid }: VP & { n: number }) {
  const { t } = useTranslation();
  const classes = useSession((s) => s.dataset?.classes ?? []);
  const plane = useSession((s) => s.dataset?.plane);
  const tree = d.tree as TreeData;
  const path = d.path as { node: number; next: number; name: string; feature: number; threshold_raw: number; raw: number; left: boolean }[];
  const leaf = d.leaf as { node: number; counts: number[]; pred: number; proba: number[]; n: number };
  const nodes = [...path.map((p) => p.node), leaf.node];
  const shownNodes = nodes.slice(0, Math.max(1, Math.min(nodes.length, step)));
  const current = step >= n ? leaf.node : nodes[Math.min(nodes.length - 1, step - 1)];
  const q = step === 1 ? path[0] : step <= path.length + 1 ? path[step - 2] : null;
  const acc = col("--acc-t"), ink = col("--ink"), ink3 = col("--ink3");
  const extra = useMemo(() => () => {
    if (!plane || plane.mode !== "raw") return [];
    const b = plane.bounds;
    let [x0, x1, y0, y1] = [b[0] - 1e3, b[1] + 1e3, b[2] - 1e3, b[3] + 1e3];
    const out: PlotExtra[] = [];
    const k = Math.min(path.length, Math.max(0, step - 1));
    for (let i = 0; i < Math.min(path.length, step); i++) {
      const s = path[i];
      const thr = s.threshold_raw;
      const seg = s.feature === 0 ? new Float32Array([thr, y0, thr, y1]) : new Float32Array([x0, thr, x1, thr]);
      out.push({ name: `split${i}`, segs: seg, style: { color: i === step - 1 ? acc : i < k ? ink : ink3, width: i === step - 1 ? 2.5 : 2, alpha: i === step - 1 ? 1 : 0.8 } });
      if (i < k) {
        if (s.feature === 0) { if (s.left) x1 = Math.min(x1, thr); else x0 = Math.max(x0, thr); }
        else if (s.left) y1 = Math.min(y1, thr); else y0 = Math.max(y0, thr);
      }
    }
    out.push({ name: "region", segs: new Float32Array([x0, y0, x1, y0, x1, y0, x1, y1, x1, y1, x0, y1, x0, y1, x0, y0]), style: { color: acc, width: 2 } });
    return out;
  }, [plane, path, step, acc, ink, ink3]);
  return (
    <div className="dec-body" style={{ gridTemplateRows: "minmax(0, 1fr) 230px", gap: 14, padding: "14px 18px 18px" }}>
      <div style={{ position: "relative" }}>
        <TreeView tree={tree} classes={classes} path={shownNodes} current={current} criterion="gini" />
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1.3fr)", gap: 14, minHeight: 0 }}>
        <div className="dcard" style={{ display: "flex", flexDirection: "column", gap: 8, padding: "12px 14px" }}>
          <div className="prow"><span className="lbl">{t("decide.tree.cuts")}</span><span className="hint">{plane?.axes.join(" → , ")} ↑</span></div>
          <div style={{ position: "relative", flex: 1 }}><PlotView variant="decide" highlightId={pid} dim onPick={onPick} extra={extra} axes={false} border={false} /></div>
        </div>
        <div className="dcard cur" style={{ display: "flex", flexDirection: "column", gap: 10, padding: "14px 16px" }}>
          {q && step <= path.length + 1 ? (
            <>
              <span className="lbl">{step === 1 ? t("decide.tree.rootQ") : t("decide.tree.qN", { n: step - 1 })}</span>
              <span style={{ fontSize: 18, fontWeight: 600 }}>{step === 1 ? t("decide.tree.isQ", { q: `${q.name} ≤ ${fmt(q.threshold_raw, 2)}` }) : t("decide.tree.answered", { q: `${q.name} ≤ ${fmt(q.threshold_raw, 2)}`, v: fmt(q.raw, 2), ans: t(q.left ? "tree.yes" : "tree.no") })}</span>
              <ThresholdBar thr={q.threshold_raw} v={q.raw} name={q.name} cls={leaf.pred} />
            </>
          ) : (
            <>
              <span className="lbl">{t("decide.tree.leafReached")}</span>
              <span style={{ fontSize: 18, fontWeight: 600 }}>{t("decide.tree.predict", { cls: classes[leaf.pred] })}</span>
              <div style={{ display: "flex", height: 22, borderRadius: 5, overflow: "hidden", gap: 2 }}>
                {leaf.counts.map((c, i) => c > 0 && <motion.div key={i} className={`k${i % 5}`} style={{ display: "flex", alignItems: "center", paddingLeft: 8, font: "600 12px var(--mono)", color: "#1a1200", flexBasis: 0 }} initial={{ flexGrow: 0.0001 }} animate={{ flexGrow: c }} transition={SPR}>{c / leaf.n > 0.2 ? `${Math.round(c)} ${classes[i]}` : ""}</motion.div>)}
              </div>
              <span className="hint" style={{ fontSize: 13 }}>{t("decide.tree.leafText", { mix: leaf.counts.map((c, i) => `${Math.round(c)} ${classes[i]}`).join(" · "), cls: classes[leaf.pred], pct: Math.round(leaf.proba[leaf.pred] * 100) })}</span>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function ThresholdBar({ thr, v, name, cls }: { thr: number; v: number; name: string; cls: number }) {
  const lo = Math.min(thr, v), hi = Math.max(thr, v);
  const pad = (hi - lo || Math.abs(thr) || 1) * 0.6;
  const a = lo - pad, b = hi + pad;
  const P = (x: number) => ((x - a) / (b - a)) * 100;
  return (
    <div className="thr">
      <div style={{ position: "absolute", left: 0, right: 0, top: 24, height: 4, borderRadius: 2, background: "var(--chip2)" }} />
      <motion.div style={{ position: "absolute", left: 0, top: 24, height: 4, borderRadius: 2, background: "var(--ink3)", width: `${P(thr)}%`, transformOrigin: "left" }} initial={{ scaleX: 0 }} animate={{ scaleX: 1 }} transition={SPR} />
      <div style={{ position: "absolute", left: `${P(thr)}%`, top: 12, width: 2, height: 28, background: "var(--ink)", transform: "translateX(-50%)" }} />
      <span className="mono" style={{ position: "absolute", left: `${P(thr)}%`, top: 40, transform: "translateX(-50%)", fontSize: 12 }}>{fmt(thr, 2)}</span>
      <motion.div style={{ position: "absolute", top: 26, width: 14, height: 14, borderRadius: 2, background: `var(--c${cls % 5})`, boxShadow: "0 0 0 3px var(--solid2)", left: 0 }}
        initial={{ x: 0, y: "-50%" }} animate={{ x: 0, y: "-50%", left: `calc(${P(v)}% - 7px)` }} transition={SPR} />
      <span className="mono" style={{ position: "absolute", left: `${P(v)}%`, top: -4, transform: "translateX(-50%)", fontSize: 12 }}>{name} = {fmt(v, 2)}</span>
    </div>
  );
}

// ------------------------------------------------------------------ kmeans

function KMeansVis({ d, step, onPick, pid }: VP) {
  const { t } = useTranslation();
  const centers = d.centers as number[][];
  const dist = d.distances as number[];
  const best = d.nearest as number;
  const pt = d.point as number[];
  const max = Math.max(1e-9, ...dist);
  const ink2 = col("--ink2"), acc = col("--acc-t");
  const extra = () => step < 2 ? [] : centers.map((c, i) => ({ name: `d${i}`, segs: new Float32Array([pt[0], pt[1], c[0], c[1]]),
    style: i === best && step >= 3 ? { color: acc, width: 3 } : { color: ink2, width: 1.5, dash: 9 } }));
  const labels = () => step < 2 ? [] : centers.map((c, i) => ({ x: (pt[0] + c[0]) / 2, y: (pt[1] + c[1]) / 2, text: fmt(dist[i], 1), cls: i === best && step >= 3 ? "plab best" : "plab" }));
  return (
    <div className="dec-body" style={{ gridTemplateColumns: "minmax(0, 1.55fr) minmax(0, 1fr)" }}>
      <PointPlot pid={pid} onPick={onPick} extra={extra} labels={labels} />
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <div className={`dcard ${blk(step, 2, [2, 3])}${step === 2 || step === 3 ? " cur" : ""}`} style={{ display: "flex", flexDirection: "column", gap: 6, padding: "14px 14px 10px" }}>
          <div className="prow" style={{ padding: "0 8px 4px" }}><span style={{ fontSize: 14, fontWeight: 600 }}>{t("decide.km.dist")}</span><span className="hint">{t("decide.km.shorter")}</span></div>
          {dist.map((v, i) => (
            <div className={`drow${i === best && step >= 3 ? " best" : ""}`} key={i}>
              <span style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, fontWeight: i === best && step >= 3 ? 600 : 400 }}><span className={`dot k${i % 5} s${i % 4}`} />{t("decide.km.center", { i: i + 1 })}</span>
              <div style={{ height: 8, borderRadius: 4, background: "var(--chip2)", overflow: "hidden" }}><motion.div style={{ width: `${(v / max) * 100}%`, height: "100%", borderRadius: 4, background: i === best && step >= 3 ? "var(--acc)" : "var(--ink3)", transformOrigin: "left" }} initial={{ scaleX: 0 }} animate={{ scaleX: step >= 2 ? 1 : 0 }} transition={SPR} /></div>
              <span className="mono" style={{ textAlign: "right", fontWeight: i === best && step >= 3 ? 600 : 400 }}>{fmt(v, 1)}</span>
            </div>
          ))}
          <span className="hint" style={{ padding: "4px 8px 0" }}>{t("decide.km.formula")}</span>
        </div>
        <div className={`dcard ${blk(step, 4)}${step === 4 ? " cur" : ""}`} style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <span style={{ fontSize: 14, fontWeight: 600 }}>{t("decide.km.how")}</span>
          <div style={{ display: "grid", gridTemplateColumns: "28px 1fr", gap: "8px 10px", fontSize: 13, color: "var(--ink2)" }}>
            <span className="mono" style={{ color: "var(--ink3)" }}>1</span><span>{t("decide.km.h1")}</span>
            <span className="mono" style={{ color: "var(--ink3)" }}>2</span><span>{t("decide.km.h2")}</span>
            <span className="mono" style={{ color: "var(--ink3)" }}>↻</span><span>{t("decide.km.h3", { n: d.rounds })}</span>
          </div>
          <Decision step={step} at={4} text={t("decide.km.join", { i: best + 1 })} cls={best} label={t("decide.km.group", { i: best + 1 })} />
        </div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ mlp

function MlpVis({ d, step, n, onPick, pid }: VP & { n: number }) {
  const { t } = useTranslation();
  const classes = useSession((s) => s.dataset?.classes ?? []);
  const layers = d.layers as { act: number[]; pre?: number[] }[];
  const net = d.network as NetworkData;
  const L = layers.length;
  const reduce = useReducedMotion();
  const [wave, setWave] = useState<number | null>(null);
  const shown = Math.min(L, step);
  useEffect(() => {
    if (reduce || step < 2 || step > L) { setWave(null); return; }
    const from = step - 2, to = step - 1;
    const t0 = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const k = Math.min(1, (now - t0) / 700);
      setWave(from + (to - from) * k);
      if (k < 1) raf = requestAnimationFrame(tick);
      else setTimeout(() => setWave(null), 200);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [step, L, reduce]);
  const pr = d.proba as number[];
  const cur = Math.min(L - 1, step - 1);
  const act = String(d.activation);
  let maths: { label: string; formula: string; lines: string[]; note: string };
  if (step === 1) maths = { label: t("decide.mlp.m1"), formula: `x = (${layers[0].act.map((v) => fmt(v, 2)).join(", ")})`, lines: [t("decide.mlp.m1a")], note: t("decide.mlp.m1n") };
  else if (step <= L - 1) {
    const l = step - 1;
    const pre = layers[l].pre ?? [];
    const a = layers[l].act;
    const top = a.map((v, i) => [v, i] as const).sort((x, y) => y[0] - x[0]);
    const off = a.findIndex((v) => v <= 0);
    maths = { label: t("decide.mlp.mh", { l }), formula: l === 1 ? `h = ${act}(W·x + b)` : `h${l} = ${act}(W${l}·h${l - 1} + b${l})`,
      lines: [t("decide.mlp.neuron", { i: top[0][1] + 1, s: fmt(pre[top[0][1]], 2), a: fmt(top[0][0], 2) }), ...(off >= 0 ? [t("decide.mlp.neuron", { i: off + 1, s: fmt(pre[off], 2), a: fmt(a[off], 2) })] : [])],
      note: act === "relu" ? t("decide.mlp.reluNote") : t("decide.mlp.actNote", { act }) };
  } else if (step === L) {
    const z = layers[L - 1].pre ?? [];
    maths = { label: t("decide.mlp.mo"), formula: z.length === 1 ? `P(${classes[1]}) = σ(z), z = ${fmt(z[0], 2)}` : `softmax(z)`, lines: [pr.map((v, i) => `P(${classes[i]}) = ${fmt(v, 2)}`).join(" · ")], note: t("decide.mlp.outNote") };
  } else maths = { label: t("decide.mlp.md"), formula: `${fmt(pr[d.pred as number], 2)} → ${classes[d.pred as number]}`, lines: [], note: d.true === d.pred ? t("decide.correct") : t("decide.incorrect") };
  return (
    <div className="dec-body" style={{ gridTemplateColumns: "minmax(0, 1fr) 280px", gap: 18, padding: "16px 20px 20px" }}>
      <div style={{ display: "flex", flexDirection: "column", minHeight: 0 }}>
        <div style={{ display: "grid", gridTemplateColumns: `repeat(${L}, minmax(0, 1fr))`, gap: 4, paddingBottom: 6, textAlign: "center" }}>
          {net.sizes.map((s, i) => <span key={i} className="lbl" style={{ color: i < shown ? "var(--ink)" : undefined }}>{i === 0 ? t("net.input") : i === L - 1 ? t("net.output") : t("net.hidden", { i })}<br /><span className="mono">{i === 0 ? t("decide.mlp.features", { n: s }) : i === L - 1 ? `${s} · ${d.out_activation}` : `${s} · ${act}`}</span></span>)}
        </div>
        <div style={{ position: "relative", flex: 1, minHeight: 0 }}>
          <NetworkView net={net} acts={layers.map((l) => l.act)} wave={wave} shownLayers={step >= n ? L : shown} currentLayer={step > L ? L - 1 : cur} values featureNames={d.features as string[]} outLabel={`P(${classes[1] ?? ""})`} />
        </div>
        <div className="hint" style={{ display: "flex", flexWrap: "wrap", gap: "6px 16px", paddingTop: 8 }}>
          <span style={{ display: "flex", alignItems: "center", gap: 6 }}><i className="k-acc" />{t("net.pos")}</span>
          <span style={{ display: "flex", alignItems: "center", gap: 6 }}><i className="k-ink" />{t("net.neg")}</span>
          <span>{t("decide.mlp.legend")}</span>
        </div>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 14, minHeight: 0 }}>
        <div className="dcard" style={{ display: "flex", flexDirection: "column", gap: 8, padding: 12 }}>
          <span className="lbl">{t("decide.thePoint")}</span>
          <div style={{ position: "relative", height: 170 }}><PlotView variant="decide" highlightId={pid} dim onPick={onPick} axes={false} /></div>
        </div>
        <motion.div key={step} className="dcard cur" style={{ display: "flex", flexDirection: "column", gap: 10, padding: "14px 16px" }} initial={reduce ? { opacity: 0 } : { opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={SPR}>
          <span className="lbl">{maths.label}</span>
          <span className="mono" style={{ fontSize: 15, fontWeight: 600, overflowWrap: "anywhere" }}>{maths.formula}</span>
          {maths.lines.map((l, i) => <span key={i} className="mono" style={{ fontSize: 12, color: "var(--ink2)", lineHeight: 1.5 }}>{l}</span>)}
          <span className="hint" style={{ fontSize: 13 }}>{maths.note}</span>
        </motion.div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ ensembles

function ForestVis({ d, step, onPick, pid }: VP) {
  const { t } = useTranslation();
  const classes = useSession((s) => s.dataset?.classes ?? []);
  const trees = (d.trees as { i: number; proba: number[] }[]).slice(0, 12);
  const avg = d.avg as number[];
  const pred = d.pred as number;
  return (
    <div className="dec-body" style={{ gridTemplateColumns: "minmax(0, .8fr) minmax(0, 1fr)" }}>
      <PointPlot pid={pid} onPick={onPick} title={t("decide.where")} />
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <div className={`dcard ${blk(step, [1, 2])}${step <= 2 ? " cur" : ""}`} style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <span style={{ fontSize: 14, fontWeight: 600 }}>{t("decide.forest.each")}</span>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(6, minmax(0, 1fr))", gap: 6 }}>
            {trees.map((tr, i) => (
              <motion.div key={tr.i} className="tree-mini" initial={{ opacity: 0, y: 6 }} animate={{ opacity: step >= 2 ? 1 : 0.35, y: 0 }} transition={{ ...SPR, delay: step === 2 ? i * 0.04 : 0 }}>
                <span className="mono hint">#{tr.i + 1}</span>
                <div style={{ height: 5, borderRadius: 3, display: "flex", overflow: "hidden" }}>{tr.proba.map((p, j) => <div key={j} className={`k${j % 5}`} style={{ flex: p }} />)}</div>
                <span className="mono" style={{ fontSize: 12, color: "var(--ink)" }}>{step >= 2 ? fmt(tr.proba[pred], 2) : "…"}</span>
              </motion.div>
            ))}
          </div>
          <span className="hint">{t("decide.forest.showing", { k: trees.length, n: d.n, cls: classes[pred] })}</span>
        </div>
        <div className={`dcard ${blk(step, 3)}${step === 3 ? " cur" : ""}`} style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span className="mono hint" style={{ width: 110 }}>{t("decide.forest.avg", { cls: classes[pred] })}</span>
          <div style={{ flex: 1, height: 12, borderRadius: 4, display: "flex", overflow: "hidden", gap: 2 }}>{avg.map((p, j) => <motion.div key={j} className={`k${j % 5}`} animate={{ flexGrow: step >= 3 ? p : 1 / avg.length }} style={{ flexBasis: 0 }} transition={SPR} />)}</div>
          <span className="mono" style={{ fontWeight: 600 }}>{fmt(avg[pred], 2)}</span>
        </div>
        <Decision step={step} at={4} text={t("decide.forest.wins")} cls={pred} label={`${classes[pred]} · ${Math.round(avg[pred] * 100)}%`} />
      </div>
    </div>
  );
}

function BoostVis({ d, step, onPick, pid }: VP) {
  const { t } = useTranslation();
  const classes = useSession((s) => s.dataset?.classes ?? []);
  const pred = d.pred as number;
  const binary = !!d.binary;
  const ci = binary ? 0 : pred;
  const base = (d.base as number[])[ci];
  const contribs = (d.contribs as number[][]).map((c) => c[ci]);
  const score = (d.score as number[])[ci];
  const pr = d.proba as number[];
  const first = contribs.slice(0, 3);
  const rest = contribs.slice(3).reduce((a, b) => a + b, 0);
  const rows = [{ k: t("decide.boost.base"), v: base, at: 1 }, ...first.map((v, i) => ({ k: t("decide.boost.tree", { i: i + 1 }), v, at: 2 })), ...(contribs.length > 3 ? [{ k: t("decide.boost.rest", { a: 4, b: contribs.length }), v: rest, at: 2 }] : [])];
  const tot = Math.max(1e-9, rows.reduce((a, r) => a + Math.abs(r.v), 0));
  let acc = 0;
  const minL = Math.min(0, ...rows.map((r) => { const s = acc; acc += r.v; return Math.min(s, acc); }));
  acc = 0;
  const P = (x: number) => ((x - minL) / tot) * 100;
  return (
    <div className="dec-body" style={{ gridTemplateColumns: "minmax(0, .8fr) minmax(0, 1fr)" }}>
      <PointPlot pid={pid} onPick={onPick} title={t("decide.where")} />
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <div className={`dcard ${step <= 3 ? "blk cur" : "blk"}`} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span style={{ fontSize: 14, fontWeight: 600, marginBottom: 6 }}>{binary ? t("decide.boost.title") : t("decide.boost.titleMulti", { cls: classes[pred] })}</span>
          {rows.map((r, i) => {
            const s = acc;
            acc += r.v;
            return (
              <div className={`wf ${blk(step, r.at)}`} key={i}><span>{r.k}</span>
                <div style={{ position: "relative", height: 10 }}><motion.div style={{ position: "absolute", left: `${P(Math.min(s, acc))}%`, width: `${(Math.abs(r.v) / tot) * 100}%`, height: "100%", borderRadius: 2, background: r.v >= 0 ? "var(--c1)" : "var(--c0)", opacity: i === rows.length - 1 && contribs.length > 3 ? 0.6 : 1, transformOrigin: "left" }} initial={{ scaleX: 0 }} animate={{ scaleX: step >= r.at ? 1 : 0 }} transition={{ ...SPR, delay: step === r.at ? i * 0.05 : 0 }} /></div>
                <span style={{ textAlign: "right" }}>{r.v >= 0 ? "+" : ""}{fmt(r.v, 2)}</span></div>
            );
          })}
          <div className={`wf ${blk(step, 3)}`} style={{ borderTop: "1px solid var(--line2)", paddingTop: 4, minHeight: 28, color: "var(--ink)", fontWeight: 600 }}><span>{t("decide.boost.score")}</span>
            <div style={{ position: "relative", height: 10 }}><motion.div style={{ position: "absolute", left: `${P(Math.min(0, score))}%`, width: `${(Math.abs(score) / tot) * 100}%`, height: "100%", background: "var(--ink)", borderRadius: 2, transformOrigin: "left" }} initial={{ scaleX: 0 }} animate={{ scaleX: step >= 3 ? 1 : 0 }} transition={SPR} /></div>
            <span style={{ textAlign: "right" }}>{fmt(score, 2)}</span></div>
          <span className="hint">{t("decide.boost.hint")}</span>
        </div>
        <Decision step={step} at={4} text={binary ? t("decide.boost.squash", { s: fmt(score, 2), p: fmt(pr[1], 2) }) : t("decide.boost.softmax")} cls={pred} label={`${classes[pred]} · ${Math.round(pr[pred] * 100)}%`} />
      </div>
    </div>
  );
}

// CatBoost: the raw row (categories as text) -> one question per tree level (oblivious trees) ->
// every tree adds its leaf value -> the raw score becomes a probability or the predicted value.
function CatBoostVis({ d, step, onPick, pid }: VP) {
  const { t } = useTranslation();
  const classes = useSession((s) => s.dataset?.classes ?? []);
  type Lv = { feature: string; kind: string; border?: number; value: unknown; yes: boolean };
  const reg = d.task === "regression";
  const pred = d.pred as number;
  const ci = reg || d.binary ? 0 : pred;
  const cols = d.columns as { name: string; value: unknown; cat: boolean }[];
  const tree = (d.trees as { levels: Lv[]; leaf: number; leaves: number; value: number[] }[])[0];
  const contribs = (d.contribs as number[][]).map((c) => c[ci]);
  const base = (d.base as number[])[ci];
  const score = (d.score as number[])[ci];
  const rows = [{ k: t("decide.boost.base"), v: base }, ...contribs.slice(0, 3).map((v, i) => ({ k: t("decide.boost.tree", { i: i + 1 }), v })),
    ...(contribs.length > 3 ? [{ k: t("decide.boost.rest", { a: 4, b: contribs.length }), v: contribs.slice(3).reduce((a, b) => a + b, 0) }] : [])];
  const tot = Math.max(1e-9, rows.reduce((a, r) => a + Math.abs(r.v), 0));
  let acc = 0;
  const minL = Math.min(0, ...rows.map((r) => { const s = acc; acc += r.v; return Math.min(s, acc); }));
  acc = 0;
  const X = (x: number) => ((x - minL) / tot) * 100;
  return (
    <div className="dec-body" style={{ gridTemplateColumns: "minmax(0, .8fr) minmax(0, 1fr)" }}>
      <PointPlot pid={pid} onPick={onPick} title={t("decide.where")} />
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <div className={`dcard ${blk(step, 1)}${step === 1 ? " cur" : ""}`} style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <span style={{ fontSize: 14, fontWeight: 600 }}>{t("decide.cat.rowTitle")}</span>
          {cols.map((c) => (
            <div className="wf" key={c.name} style={{ gridTemplateColumns: "110px minmax(0, 1fr) auto" }}><span>{c.name}</span>
              <span style={{ color: "var(--ink)" }}>{c.cat ? `"${String(c.value)}"` : fmt(c.value as number, 2)}</span>
              <span>{c.cat && <span className="tag" style={{ height: 20, fontSize: 12 }}>{t("decide.cat.asText")}</span>}</span></div>
          ))}
          <span className="hint">{cols.some((c) => c.cat) ? t("decide.cat.rowHint") : t("decide.cat.rowHintNoCats")}</span>
        </div>
        {tree && (
          <div className={`dcard ${blk(step, 2)}${step === 2 ? " cur" : ""}`} style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <span style={{ fontSize: 14, fontWeight: 600 }}>{t("decide.cat.treeTitle", { n: tree.levels.length, leaves: tree.leaves })}</span>
            {tree.levels.map((l, i) => (
              <motion.div className="wf" key={i} initial={false} animate={{ opacity: step >= 2 ? 1 : 0.4 }} transition={{ delay: step === 2 ? i * 0.12 : 0 }}>
                <span>{t("decide.cat.level", { i: i + 1 })}</span>
                <span style={{ color: "var(--ink)", minWidth: 0, overflowWrap: "anywhere" }}>
                  {l.kind === "num" ? `${l.feature} = ${fmt(l.value as number, 2)} > ${fmt(l.border as number, 2)}?` : t("decide.cat.ctrAsk", { f: l.feature, v: String(l.value) })}
                </span>
                <span style={{ textAlign: "right", color: l.yes ? "var(--acc-t)" : "var(--ink2)" }}>{l.yes ? t("decide.cat.yes") : t("decide.cat.no")}</span>
              </motion.div>
            ))}
            <div className="wf" style={{ borderTop: "1px solid var(--line2)", paddingTop: 4, color: "var(--ink)", fontWeight: 600 }}>
              <span>{t("decide.cat.leaf", { i: tree.leaf + 1, n: tree.leaves })}</span><span /><span style={{ textAlign: "right" }}>{fmt(tree.value[ci], 3)}</span></div>
            <span className="hint">{t("decide.cat.treeHint")}</span>
          </div>
        )}
        <div className={`dcard ${blk(step, 3)}${step === 3 ? " cur" : ""}`} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span style={{ fontSize: 14, fontWeight: 600 }}>{reg || d.binary ? t("decide.boost.title") : t("decide.boost.titleMulti", { cls: classes[pred] })}</span>
          {rows.map((r, i) => {
            const s = acc;
            acc += r.v;
            return (
              <div className="wf" key={i}><span>{r.k}</span>
                <div style={{ position: "relative", height: 10 }}><motion.div style={{ position: "absolute", left: `${X(Math.min(s, acc))}%`, width: `${(Math.abs(r.v) / tot) * 100}%`, height: "100%", borderRadius: 2, background: r.v >= 0 ? "var(--c1)" : "var(--c0)", transformOrigin: "left" }} initial={{ scaleX: 0 }} animate={{ scaleX: step >= 3 ? 1 : 0 }} transition={{ ...SPR, delay: step === 3 ? i * 0.05 : 0 }} /></div>
                <span style={{ textAlign: "right" }}>{r.v >= 0 ? "+" : ""}{fmt(r.v, 3)}</span></div>
            );
          })}
          <div className="wf" style={{ borderTop: "1px solid var(--line2)", paddingTop: 4, color: "var(--ink)", fontWeight: 600 }}><span>{t("decide.boost.score")}</span><span /><span style={{ textAlign: "right" }}>{fmt(score, 3)}</span></div>
        </div>
        {reg ? (
          <div className={`dcard ${blk(step, 4)}${step === 4 ? " cur" : ""}`} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
            <span style={{ fontSize: 14, fontWeight: 600 }}>{t("decide.cat.regOut")}</span>
            <span className="tag" style={{ height: 34, padding: "0 14px", fontSize: 14, color: "var(--ink)" }}>{fmt(d.value as number, 2)}</span>
          </div>
        ) : (
          <Decision step={step} at={4} text={d.binary ? t("decide.boost.squash", { s: fmt(score, 2), p: fmt((d.proba as number[])[1], 2) }) : t("decide.boost.softmax")} cls={pred}
            label={`${classes[pred]} · ${Math.round((d.proba as number[])[pred] * 100)}%`} />
        )}
      </div>
    </div>
  );
}

function AdaVis({ d, step, onPick, pid }: VP) {
  const { t } = useTranslation();
  const classes = useSession((s) => s.dataset?.classes ?? []);
  const votes = (d.votes as { i: number; cls: number; weight: number }[]);
  const top = [...votes].sort((a, b) => b.weight - a.weight).slice(0, 8);
  const mx = Math.max(1e-9, ...top.map((v) => v.weight));
  const share = d.share as number[];
  const pred = d.pred as number;
  return (
    <div className="dec-body" style={{ gridTemplateColumns: "minmax(0, .8fr) minmax(0, 1fr)" }}>
      <PointPlot pid={pid} onPick={onPick} title={t("decide.where")} />
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <div className={`dcard ${blk(step, 1)}${step === 1 ? " cur" : ""}`} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span style={{ fontSize: 14, fontWeight: 600, marginBottom: 6 }}>{t("decide.ada.title", { n: votes.length })}</span>
          {top.map((v) => (
            <div className="wf" key={v.i}><span style={{ display: "flex", alignItems: "center", gap: 6 }}>#{v.i + 1} <span className={`dot k${v.cls % 5} s${v.cls % 4}`} />{classes[v.cls]}</span>
              <div style={{ height: 8, borderRadius: 4, background: "var(--chip2)", overflow: "hidden" }}><motion.div style={{ width: `${(v.weight / mx) * 100}%`, height: "100%", background: `var(--c${v.cls % 5})`, transformOrigin: "left" }} initial={{ scaleX: 0 }} animate={{ scaleX: 1 }} transition={SPR} /></div>
              <span style={{ textAlign: "right" }}>{fmt(v.weight, 2)}</span></div>
          ))}
          <span className="hint">{t("decide.ada.hint")}</span>
        </div>
        <div className={`dcard ${blk(step, 2)}${step === 2 ? " cur" : ""}`}>
          <span style={{ fontSize: 14, fontWeight: 600 }}>{t("decide.ada.share")}</span>
          <div style={{ display: "flex", height: 22, borderRadius: 5, overflow: "hidden", gap: 2, marginTop: 8 }}>{share.map((v, i) => <motion.div key={i} className={`k${i % 5}`} animate={{ flexGrow: step >= 2 ? Math.max(0.0001, v) : 1 / share.length }} style={{ flexBasis: 0 }} transition={SPR} />)}</div>
        </div>
        <Decision step={step} at={3} text={t("decide.ada.pick")} cls={pred} label={`${classes[pred]} · ${Math.round(share[pred] * 100)}%`} />
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ knn / svc / nb / dbscan / pca

function KnnVis({ d, step, onPick, pid }: VP) {
  const { t } = useTranslation();
  const classes = useSession((s) => s.dataset?.classes ?? []);
  const plane = useSession((s) => s.dataset?.plane);
  const scaled = useSession((s) => s.dataset?.scale);
  const nb = d.neighbors as { id: number; dist: number; cls: number; w: number; xy: number[] }[];
  const votes = d.votes as number[];
  const pred = d.pred as number;
  const pt = d.point as number[];
  const ring = useMemo(() => (step >= 2 ? new Set(nb.map((x) => x.id)) : null), [nb, step]);
  const ink2 = col("--ink2"), acc = col("--acc-t");
  const r = Math.max(...nb.map((x) => Math.hypot(x.xy[0] - pt[0], x.xy[1] - pt[1])));
  const extra = () => {
    const out: PlotExtra[] = [];
    if (step >= 1) for (const x of nb) out.push({ name: `n${x.id}`, segs: new Float32Array([pt[0], pt[1], x.xy[0], x.xy[1]]), style: { color: step >= 3 && x.cls === pred ? acc : ink2, width: 1.5, dash: step >= 3 ? 0 : 8, alpha: 0.8 } });
    if (step >= 2 && plane?.mode === "raw" && !scaled) out.push({ name: "circle", segs: circleSegs(pt[0], pt[1], r, r), style: { color: ink2, width: 1.5, dash: 8 } });
    return out;
  };
  const total = votes.reduce((a, b) => a + b, 0) || 1;
  return (
    <div className="dec-body" style={{ gridTemplateColumns: "minmax(0, 1.4fr) minmax(0, 1fr)" }}>
      <PointPlot pid={pid} onPick={onPick} extra={extra} ringSet={ring} title={t("decide.where")} hint={t("decide.knn.ringed")} />
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <div className={`dcard ${blk(step, [1, 2])}${step <= 2 ? " cur" : ""}`} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span style={{ fontSize: 14, fontWeight: 600, marginBottom: 6 }}>{t("decide.knn.closest", { k: d.k })}</span>
          {nb.slice(0, 10).map((x) => (
            <div className="wf" key={x.id}><span style={{ display: "flex", alignItems: "center", gap: 6 }}><span className={`dot k${x.cls % 5} s${x.cls % 4}`} />#{x.id}</span>
              <div style={{ height: 8, borderRadius: 4, background: "var(--chip2)", overflow: "hidden" }}><motion.div style={{ width: `${(x.dist / Math.max(1e-9, nb[nb.length - 1].dist)) * 100}%`, height: "100%", background: "var(--ink3)", transformOrigin: "left" }} initial={{ scaleX: 0 }} animate={{ scaleX: 1 }} transition={SPR} /></div>
              <span style={{ textAlign: "right" }}>{fmt(x.dist, 2)}</span></div>
          ))}
        </div>
        <div className={`dcard ${blk(step, 3)}${step === 3 ? " cur" : ""}`} style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <span style={{ fontSize: 14, fontWeight: 600 }}>{t("decide.knn.vote")}</span>
          {votes.map((v, i) => v > 0 && (
            <div className="wf" key={i}><span style={{ display: "flex", alignItems: "center", gap: 6 }}><span className={`dot k${i % 5} s${i % 4}`} />{classes[i]}</span>
              <div style={{ height: 10, borderRadius: 4, background: "var(--chip2)", overflow: "hidden" }}><motion.div className={`k${i % 5}`} style={{ width: `${(v / total) * 100}%`, height: "100%", transformOrigin: "left" }} initial={{ scaleX: 0 }} animate={{ scaleX: step >= 3 ? 1 : 0 }} transition={SPR} /></div>
              <span style={{ textAlign: "right" }}>{Number.isInteger(v) ? v : fmt(v, 2)}</span></div>
          ))}
        </div>
        <Decision step={step} at={4} text={t("decide.knn.majority")} cls={pred} label={`${classes[pred]} · ${Math.round((d.proba as number[])[pred] * 100)}%`} />
      </div>
    </div>
  );
}

function SvcVis({ d, step, onPick, pid }: VP) {
  const { t } = useTranslation();
  const classes = useSession((s) => s.dataset?.classes ?? []);
  const pred = d.pred as number;
  const top = (d.top as { id: number; alpha: number; k: number; contrib: number; xy: number[] }[] | undefined) ?? [];
  const pt = (d.xy as number[]) ?? [0, 0];
  const acc = col("--acc-t"), c0 = col("--c0"), c1 = col("--c1");
  const extra = () => step < 1 || !d.binary ? [] : top.slice(0, 6).map((x) => ({ name: `sv${x.id}`, segs: new Float32Array([pt[0], pt[1], x.xy[0], x.xy[1]]), style: { color: step >= 2 ? (x.contrib >= 0 ? c1 : c0) : acc, width: 1 + Math.min(3, Math.abs(x.contrib) * 2), alpha: 0.8 } }));
  const dec = d.decision as number[];
  const mx = Math.max(1e-9, ...top.map((x) => Math.abs(x.contrib)));
  return (
    <div className="dec-body" style={{ gridTemplateColumns: "minmax(0, 1.3fr) minmax(0, 1fr)" }}>
      <PointPlot pid={pid} onPick={onPick} extra={extra} title={t("decide.where")} hint={t("decide.svc.ringed", { n: d.n_sv, of: d.n_train })} />
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        {d.binary ? (
          <div className={`dcard ${blk(step, [1, 2])}${step <= 2 ? " cur" : ""}`} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span style={{ fontSize: 14, fontWeight: 600, marginBottom: 6 }}>{t("decide.svc.compare")}</span>
            {top.slice(0, 7).map((x) => (
              <div className="wf" key={x.id}><span>SV #{x.id}</span>
                <div style={{ position: "relative", height: 10 }}><motion.div style={{ position: "absolute", left: x.contrib >= 0 ? "50%" : `${50 - (Math.abs(x.contrib) / mx) * 50}%`, width: `${(Math.abs(x.contrib) / mx) * 50}%`, height: "100%", borderRadius: 2, background: x.contrib >= 0 ? "var(--c1)" : "var(--c0)", transformOrigin: x.contrib >= 0 ? "left" : "right" }} initial={{ scaleX: 0 }} animate={{ scaleX: step >= 2 ? 1 : 0.15 }} transition={SPR} /></div>
                <span style={{ textAlign: "right" }}>{fmt(x.contrib, 2)}</span></div>
            ))}
            <div className="wf"><span>{t("decide.svc.rest")}</span><span /><span style={{ textAlign: "right" }}>{fmt(d.rest as number, 2)}</span></div>
            <div className={`wf ${blk(step, 2)}`} style={{ borderTop: "1px solid var(--line2)", paddingTop: 4, color: "var(--ink)", fontWeight: 600 }}><span>{t("decide.svc.sum")}</span><span className="hint">b = {fmt(d.intercept as number, 2)}</span><span style={{ textAlign: "right" }}>{fmt(dec[0], 2)}</span></div>
          </div>
        ) : (
          <div className={`dcard ${blk(step, 2)}${step === 2 ? " cur" : ""}`}>
            <span style={{ fontSize: 14, fontWeight: 600 }}>{t("decide.svc.ovr")}</span>
            {dec.map((v, i) => <div className="wf" key={i}><span>{classes[i] ?? i}</span><span /><span style={{ textAlign: "right" }}>{fmt(v, 2)}</span></div>)}
          </div>
        )}
        <Decision step={step} at={3} text={d.binary ? t("decide.svc.sign", { v: fmt(dec[0], 2) }) : t("decide.svc.biggest")} cls={pred} label={classes[pred]} />
      </div>
    </div>
  );
}

function Bells({ means, stds, x, classes }: { means: number[]; stds: number[]; x: number; classes: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current!;
    const r = c.getBoundingClientRect();
    const dpr = devicePixelRatio || 1;
    c.width = r.width * dpr; c.height = r.height * dpr;
    const ctx = c.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const W = r.width, H = r.height;
    ctx.fillStyle = css(backdrop(c));
    ctx.fillRect(0, 0, W, H);
    const lo = Math.min(x, ...means.map((m, i) => m - 3 * stds[i])), hi = Math.max(x, ...means.map((m, i) => m + 3 * stds[i]));
    const X = (v: number) => ((v - lo) / (hi - lo || 1)) * W;
    ctx.clearRect(0, 0, W, H);
    for (let k = 0; k < classes; k++) {
      ctx.strokeStyle = cssVar(`--c${k % 5}`); ctx.lineWidth = 2; ctx.beginPath();
      for (let i = 0; i <= 60; i++) {
        const v = lo + ((hi - lo) * i) / 60;
        const y = Math.exp(-0.5 * ((v - means[k]) / stds[k]) ** 2);
        i === 0 ? ctx.moveTo(X(v), H - 4 - y * (H - 10)) : ctx.lineTo(X(v), H - 4 - y * (H - 10));
      }
      ctx.stroke();
    }
    ctx.strokeStyle = cssVar("--ink"); ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(X(x), 0); ctx.lineTo(X(x), H); ctx.stroke();
  }, [means, stds, x, classes]);
  return <canvas ref={ref} style={{ width: "100%", height: "100%", display: "block" }} aria-hidden="true" />;
}

function NbVis({ d, step, onPick, pid }: VP) {
  const { t } = useTranslation();
  const classes = useSession((s) => s.dataset?.classes ?? []);
  const feats = (d.features as string[]).slice(0, 3);
  const x = d.x as number[];
  const means = d.means as number[][];
  const stds = d.stds as number[][];
  const lik = d.lik as number[][];
  const pri = d.priors as number[];
  const post = d.posterior as number[];
  const pred = d.pred as number;
  const K = pri.length;
  return (
    <div className="dec-body" style={{ gridTemplateColumns: "minmax(0, .8fr) minmax(0, 1fr)" }}>
      <PointPlot pid={pid} onPick={onPick} title={t("decide.where")} />
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <div className={`dcard ${blk(step, 1)}${step === 1 ? " cur" : ""}`}>
          <span style={{ fontSize: 14, fontWeight: 600 }}>{t("decide.nb.priors")}</span>
          <div style={{ display: "flex", gap: 8, marginTop: 8, flexWrap: "wrap" }}>{pri.map((p, i) => <span className="tag" key={i}><span className={`dot k${i % 5} s${i % 4}`} />{classes[i]} {fmt(p, 2)}</span>)}</div>
        </div>
        <div className={`dcard ${blk(step, 2)}${step === 2 ? " cur" : ""}`} style={{ display: "grid", gridTemplateColumns: `repeat(${feats.length}, minmax(0, 1fr))`, gap: 10 }}>
          {feats.map((f, j) => (
            <div key={f} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span className="mono hint">{f} = {fmt(x[j], 2)}</span>
              <div className="bells"><Bells means={means.map((m) => m[j])} stds={stds.map((s) => s[j])} x={x[j]} classes={K} /></div>
              <span className="mono" style={{ fontSize: 12, color: "var(--ink2)" }}>{lik.map((l, k) => `${classes[k]} ${fmt(l[j], 2)}`).join(" · ")}</span>
            </div>
          ))}
        </div>
        <div className={`dcard ${blk(step, 3)}${step === 3 ? " cur" : ""}`}>
          <div className="mono" style={{ fontSize: 12, color: "var(--ink2)", display: "grid", gridTemplateColumns: "auto 1fr", gap: "2px 12px" }}>
            {pri.map((p, k) => <span key={k} style={{ display: "contents" }}><span style={{ color: `var(--c${k % 5})` }}>{classes[k]}</span><span>{fmt(p, 2)} × {lik[k].slice(0, 3).map((v) => fmt(v, 2)).join(" × ")}{lik[k].length > 3 ? " × …" : ""}</span></span>)}
          </div>
        </div>
        <Decision step={step} at={4} text={t("decide.nb.normalize")} cls={pred} label={`${classes[pred]} · ${Math.round(post[pred] * 100)}%`} />
      </div>
    </div>
  );
}

function DbscanVis({ d, step, onPick, pid }: VP) {
  const { t } = useTranslation();
  const nb = d.neighbors as { id: number; core: boolean; xy: number[] }[];
  const pt = d.point as number[];
  const ring = useMemo(() => (step >= 2 ? new Set(nb.map((x) => x.id)) : null), [nb, step]);
  const acc = col("--acc-t");
  const eps = d.eps as number;
  const exact = !!d.plane_is_model_space;
  const extra = () => step >= 1 && exact ? [{ name: "eps", segs: circleSegs(pt[0], pt[1], eps, eps), style: { color: acc, width: 1.5 } }] : [];
  const label = d.label as number;
  return (
    <div className="dec-body" style={{ gridTemplateColumns: "minmax(0, 1.4fr) minmax(0, 1fr)" }}>
      <PointPlot pid={pid} onPick={onPick} extra={extra} ringSet={ring} title={t("decide.where")} hint={exact ? t("decide.db.circle") : t("decide.db.noCircle")} />
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <div className={`dcard ${blk(step, [1, 2])}${step <= 2 ? " cur" : ""}`}>
          <span style={{ fontSize: 14, fontWeight: 600 }}>{t("decide.db.count")}</span>
          <div className="mono" style={{ fontSize: 34, fontWeight: 600, marginTop: 6 }}>{step >= 2 ? d.count as number : "…"}</div>
          <span className="hint">{t("decide.db.need", { m: d.min_samples, eps: fmt(eps, 2) })}</span>
        </div>
        <div className={`dcard ${blk(step, 3)}${step === 3 ? " cur" : ""}`}>
          <span style={{ fontSize: 14, fontWeight: 600 }}>{t(`decide.db.${d.status}`)}</span>
          <p className="hint" style={{ margin: "6px 0 0", fontSize: 13 }}>{t(`decide.db.${d.status}Text`)}</p>
        </div>
        <Decision step={step} at={4} text={t("decide.db.result")} cls={Math.max(0, label)} label={label >= 0 ? t("decide.db.cluster", { i: label + 1 }) : t("decide.db.noiseLbl")} />
        <span className="hint">{t("decide.db.trainingOnly")}</span>
      </div>
    </div>
  );
}

function PcaVis({ d, step, onPick, pid }: VP) {
  const { t } = useTranslation();
  const feats = d.features as string[];
  const comps = d.components as number[][];
  const xc = d.centered as number[];
  const sc = d.scores as number[];
  return (
    <div className="dec-body" style={{ gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1fr)" }}>
      <PointPlot pid={pid} onPick={onPick} title={t("decide.pca.where")} heat={false} />
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <div className={`dcard ${blk(step, 1)}${step === 1 ? " cur" : ""}`}>
          <span style={{ fontSize: 14, fontWeight: 600 }}>{t("decide.pca.center")}</span>
          <div className="mono" style={{ fontSize: 12, color: "var(--ink2)", marginTop: 6 }}>{feats.slice(0, 6).map((f, i) => `${f} − mean = ${fmt(xc[i], 2)}`).join(" · ")}</div>
        </div>
        {[0, 1].filter((i) => comps[i]).map((i) => (
          <div key={i} className={`dcard ${blk(step, i + 2)}${step === i + 2 ? " cur" : ""}`} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span style={{ fontSize: 14, fontWeight: 600 }}>{t("decide.pca.dot", { i: i + 1 })}</span>
            {feats.slice(0, 6).map((f, j) => (
              <div className="wf" key={f}><span>{f} × {fmt(comps[i][j], 2)}</span>
                <div style={{ position: "relative", height: 10 }}><motion.div style={{ position: "absolute", left: comps[i][j] * xc[j] >= 0 ? "50%" : `${50 - Math.min(50, Math.abs(comps[i][j] * xc[j]) * 20)}%`, width: `${Math.min(50, Math.abs(comps[i][j] * xc[j]) * 20)}%`, height: "100%", background: "var(--acc)", borderRadius: 2 }} initial={{ opacity: 0 }} animate={{ opacity: step >= i + 2 ? 1 : 0.2 }} /></div>
                <span style={{ textAlign: "right" }}>{fmt(comps[i][j] * xc[j], 2)}</span></div>
            ))}
            <div className="wf" style={{ color: "var(--ink)", fontWeight: 600, borderTop: "1px solid var(--line2)", paddingTop: 4 }}><span>PC{i + 1}</span><span /><span style={{ textAlign: "right" }}>{fmt(sc[i], 2)}</span></div>
          </div>
        ))}
      </div>
    </div>
  );
}
