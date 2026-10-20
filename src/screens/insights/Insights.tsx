import { motion } from "motion/react";
import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { useNav } from "../../app/nav";
import { useUi } from "../../app/ui";
import { Scatter } from "../../charts/Scatter";
import { engine } from "../../engine/client";
import { useSession } from "../../engine/store";
import type { ND } from "../../engine/types";
import { Icon } from "../../ui/Icon";
import { fmt, Info, Toggle } from "../../ui/controls";

interface Feat { name: string; mean: number; std: number; min: number; max: number; q1: number; median: number; q3: number; iqr: number; whisker_lo: number; whisker_hi: number; bins: number[]; hist: number[][] }
interface StatsData {
  features: Feat[]; classes: string[]; n: number; balance?: number[];
  categorical: { name: string; counts: Record<string, number> }[];
  pair: { features: string[]; ids: ND<Int32Array>; values: ND<Float32Array>; cls: ND<Int32Array> };
  corr?: { features: string[]; matrix: number[][] };
  spread?: { wide: string; narrow: string; ratio: number | null };
  outliers: { id: number; values: Record<string, number | null>; why: { feature: string; dir: "low" | "high" }[] }[];
  outlier_count: number;
}

export function Insights() {
  const { t } = useTranslation();
  const ds = useSession((s) => s.dataset);
  const [st, setSt] = useState<StatsData | null>(null);
  const hover = useUi((s) => s.insightHover);
  const showOut = useUi((s) => s.showOutliers);
  const go = useNav((s) => s.go);
  useEffect(() => {
    if (!ds) return;
    void engine.request<{ stats: StatsData }>({ type: "dataset_stats" }).then((r) => setSt(r.stats)).catch(() => undefined);
  }, [ds?.version, ds?.n_rows, ds?.name]);
  const idToIdx = useMemo(() => {
    const m = new Map<number, number>();
    if (st) st.pair.ids.data.forEach((id, i) => m.set(id, i));
    return m;
  }, [st]);
  const outIdx = useMemo(() => (st && showOut ? new Set(st.outliers.map((o) => idToIdx.get(o.id)!).filter((x) => x != null)) : null), [st, showOut, idToIdx]);
  if (!ds) {
    return (
      <div className="ins" style={{ gridTemplateColumns: "1fr" }}>
        <div className="solid" style={{ display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 12 }}>
          <span style={{ fontSize: 18, fontWeight: 600 }}>{t("states.noData")}</span>
          <button className="btn pri" onClick={() => go("data")}>{t("states.choosePreset")}</button>
        </div>
      </div>
    );
  }
  if (!st) return <div className="ins"><span className="hint">…</span></div>;
  const feats = st.pair.features;
  const k = feats.length;
  const balanced = st.balance && st.balance.length > 1 ? Math.max(...st.balance) / Math.max(1, Math.min(...st.balance)) < 1.5 : true;
  const maxBal = Math.max(1, ...(st.balance ?? [1]));
  const globalLo = Math.min(...st.features.map((f) => f.min));
  const globalHi = Math.max(...st.features.map((f) => f.max));
  const pc = (v: number) => ((v - globalLo) / (globalHi - globalLo || 1)) * 100;
  const hovered = hover != null ? idToIdx.get(hover) ?? null : null;
  const scale = ds.scale;
  return (
    <div className="ins" data-testid="insights">
      <div className="col">
        <section className="solid icard">
          <div className="ch"><span className="ct">{t("ins.stats")}<Info label={t("ins.statsAbout")}>{t("ins.statsTip")}</Info></span></div>
          <div>
            <div className="stt hd"><div>{t("ins.feature")}</div><div>{t("ins.mean")}</div><div>{t("ins.std")}</div><div>{t("ins.min")}</div><div>{t("ins.max")}</div></div>
            {st.features.map((f) => (
              <div className="stt" key={f.name}><div title={f.name}>{f.name}</div>
                {[f.mean, f.std, f.min, f.max].map((v, i) => <div key={i} title={String(v)}>{statNum(v, f.std)}</div>)}</div>
            ))}
            {st.categorical.map((c) => (
              <div className="hint" key={c.name} style={{ padding: "6px 0" }}><span className="mono">{c.name}</span>: {Object.entries(c.counts).map(([v, n]) => `${v} ${n}`).join(" · ")}</div>
            ))}
          </div>
          <span className="hint">{t("ins.statsHint")}</span>
        </section>
        {st.balance && (
          <section className="solid icard">
            <div className="ch"><span className="ct">{t("ins.balance")}<Info label={t("ins.balanceAbout")}>{t("ins.balanceTip")}</Info></span>
              <span className="tag" style={{ color: balanced ? "var(--acc-t)" : "var(--warn)" }}>{balanced ? t("ins.balanced") : t("ins.unbalanced")}</span></div>
            {st.classes.map((c, i) => (
              <div key={c} style={{ display: "grid", gridTemplateColumns: "70px 1fr 40px", gap: 10, alignItems: "center" }}>
                <span style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13 }}><span className={`dot k${i % 5} s${i % 4}`} />{c}</span>
                <div style={{ height: 12, borderRadius: 4, background: "var(--chip)" }}>
                  <motion.div className={`k${i % 5}`} style={{ width: `${((st.balance![i] ?? 0) / maxBal) * 100}%`, height: "100%", borderRadius: 4, transformOrigin: "left" }} initial={{ scaleX: 0 }} animate={{ scaleX: 1 }} transition={{ type: "spring", stiffness: 300, damping: 30 }} />
                </div>
                <span className="mono" style={{ textAlign: "right" }}>{st.balance![i]}</span>
              </div>
            ))}
            <span className="hint">{t("ins.balanceHint")}</span>
          </section>
        )}
        <section className="solid icard">
          <div className="ch"><span className="ct">{t("ins.outliers")}<Info label={t("ins.outliersAbout")}>{t("ins.outliersTip")}</Info></span>
            <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: "var(--ink2)" }}>{t("ins.showOnPlots")}<Toggle checked={showOut} label={t("ins.showOnPlots")} onChange={(v) => useUi.setState({ showOutliers: v })} /></label></div>
          <span className="hint">{t("ins.outliersHint")}</span>
          {st.outliers.length === 0 && <span className="hint">{t("ins.noOutliers")}</span>}
          {st.outliers.slice(0, 12).map((o) => (
            <div key={o.id} className={`olr${hover === o.id ? " hov" : ""}`} onMouseEnter={() => useUi.setState({ insightHover: o.id })} onMouseLeave={() => useUi.setState({ insightHover: null })}>
              <span className="mono" style={{ fontSize: 12.5 }}>#{o.id} · {Object.entries(o.values).map(([n, v]) => `${n} ${fmt(v)}`).join(" · ")}</span>
              <button className="btn sm gh" onClick={() => engine.send({ type: "remove_rows", ids: [o.id] })}>{t("ins.remove")}</button>
              <span className="hint">{o.why.map((w) => t(w.dir === "high" ? "ins.high" : "ins.low", { f: w.feature })).join(", ")}</span>
            </div>
          ))}
          {st.outliers.length > 1 && <button className="btn sm" style={{ alignSelf: "flex-start" }} onClick={() => engine.send({ type: "remove_rows", ids: st.outliers.map((o) => o.id) })}>{t("ins.removeAll", { count: st.outlier_count })}</button>}
        </section>
      </div>

      <section className="solid icard" style={{ minHeight: 0 }}>
        <div className="ch">
          <span className="ct">{t("ins.pair")}<Info label={t("ins.pairAbout")}>{t("ins.pairTip")}</Info></span>
          <div style={{ display: "flex", alignItems: "center", gap: 12, fontSize: 13, color: "var(--ink2)" }}>
            {st.classes.map((c, i) => <span key={c} style={{ display: "flex", alignItems: "center", gap: 6 }}><span className={`dot k${i % 5} s${i % 4}`} />{c}</span>)}
            <span style={{ display: "flex", alignItems: "center", gap: 6 }}><span style={{ width: 9, height: 9, borderRadius: "50%", boxShadow: "0 0 0 2px var(--warn)" }} />{t("ins.outlier")}</span>
          </div>
        </div>
        <span className="hint">{t("ins.pairHint")}{st.features.length > 4 ? " " + t("ins.pairFirst4") : ""}</span>
        {k === 0 ? <span className="hint">{t("ins.noNumeric")}</span> : (
          <div className="pair" style={{ gridTemplateColumns: `26px repeat(${k}, minmax(0, 1fr))`, gridTemplateRows: `repeat(${k}, minmax(0, 1fr)) 22px` }}>
            {feats.map((fy, r) => (
              <PairRow key={fy} r={r} feats={feats} st={st} hovered={hovered} outIdx={outIdx} />
            ))}
            <div />
            {feats.map((f) => <div key={f} className="pax">{f}</div>)}
          </div>
        )}
      </section>

      <div className="col">
        <section className="solid icard">
          <div className="ch"><span className="ct">{t("ins.spread")}<Info label={t("ins.boxAbout")}>{t("ins.boxTip")}</Info></span><span className="hint">{t("ins.boxHalf")}</span></div>
          {st.features.slice(0, 6).map((f) => {
            const outs = st.outliers.filter((o) => o.why.some((w) => w.feature === f.name)).map((o) => o.values[f.name]).filter((v): v is number => v != null);
            return (
              <div key={f.name} style={{ display: "flex", flexDirection: "column", gap: 6, padding: "6px 0 10px", borderBottom: "1px solid var(--line)" }}>
                <div className="ch"><span className="mono" style={{ fontWeight: 600 }}>{f.name}</span><span className="mono hint">IQR {fmt(f.iqr)} · {t("ins.range")} {fmt(f.max - f.min)}</span></div>
                <div className="box">
                  <div style={{ position: "absolute", left: `${pc(f.whisker_lo)}%`, width: `${pc(f.whisker_hi) - pc(f.whisker_lo)}%`, top: 14, height: 2, background: "var(--ink3)" }} />
                  <div style={{ position: "absolute", left: `${pc(f.q1)}%`, width: `${Math.max(0.5, pc(f.q3) - pc(f.q1))}%`, top: 4, height: 22, borderRadius: 4, background: "var(--acc-soft)", border: "1.5px solid var(--acc-t)" }} />
                  <div style={{ position: "absolute", left: `${pc(f.median)}%`, top: 4, width: 2, height: 22, background: "var(--ink)" }} />
                  <div style={{ position: "absolute", left: `${pc(f.whisker_lo)}%`, top: 8, width: 2, height: 14, background: "var(--ink3)" }} />
                  <div style={{ position: "absolute", left: `${pc(f.whisker_hi)}%`, top: 8, width: 2, height: 14, background: "var(--ink3)" }} />
                  {outs.map((v, i) => <div key={i} style={{ position: "absolute", left: `${pc(v)}%`, top: 15, width: 9, height: 9, transform: "translate(-50%, -50%)", borderRadius: "50%", boxShadow: "0 0 0 2px var(--warn)" }} />)}
                </div>
                <div className="ch mono hint"><span>{fmt(f.min)}</span><span>{t("ins.median")} {fmt(f.median)}</span><span>{fmt(f.max)}</span></div>
              </div>
            );
          })}
          <span className="hint">{t("ins.sameScale", { lo: fmt(globalLo, 1), hi: fmt(globalHi, 1) })}</span>
        </section>
        <section className="glass icard">
          <span className="ct">{t("ins.meaning")}</span>
          {st.spread && st.spread.ratio != null && (
            <p style={{ margin: 0, fontSize: 13, lineHeight: 1.5, color: "var(--ink2)" }} dangerouslySetInnerHTML={{ __html: t("ins.spreadText", { wide: st.spread.wide, narrow: st.spread.narrow, r: fmt(st.spread.ratio, 1) }) }} />
          )}
          <label style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, fontSize: 13 }}>
            {t("ins.scale")}<Toggle checked={scale} label={t("ins.scale")} onChange={(v) => engine.send({ type: "set_scale", scale: v })} />
          </label>
          <span className="hint">{t("ins.scaleHint")}</span>
          {st.corr && st.corr.features.length === 2 && (
            <>
              <div className="ch" style={{ paddingTop: 8, borderTop: "1px solid var(--line)" }}><span style={{ fontSize: 13, color: "var(--ink2)" }}>{t("ins.corr2", { a: st.corr.features[0], b: st.corr.features[1] })}</span><span className="mono" style={{ fontWeight: 600 }}>{fmt(st.corr.matrix[0][1])}</span></div>
              <span className="hint">{t("ins.corrHint")}</span>
            </>
          )}
          {st.corr && st.corr.features.length > 2 && (
            <div style={{ paddingTop: 8, borderTop: "1px solid var(--line)", display: "flex", flexDirection: "column", gap: 6 }}>
              <span style={{ fontSize: 13, color: "var(--ink2)" }}>{t("ins.corr")}</span>
              <div className="corr" style={{ gridTemplateColumns: `60px repeat(${Math.min(6, st.corr.features.length)}, minmax(0, 1fr))` }}>
                <div />
                {st.corr.features.slice(0, 6).map((f) => <div key={f} className="hint" style={{ overflow: "hidden" }} title={f}>{f.slice(0, 6)}</div>)}
                {st.corr.matrix.slice(0, 6).map((row, i) => (
                  <Fragment key={i} label={st.corr!.features[i]} row={row.slice(0, 6)} />
                ))}
              </div>
              <span className="hint">{t("ins.corrHint")}</span>
            </div>
          )}
        </section>
        <button className="btn sm" style={{ alignSelf: "flex-start" }} onClick={() => go("data", { dataTab: "table" })}><Icon name="table" />{t("ins.editTable")}</button>
      </div>
    </div>
  );
}

/**
 * Stats cell: decimals follow the feature's spread, so a mean of −0.0027 next to a spread of 0.5
 * reads "0.00" instead of a wide "−2.7e-3" (the exact value is in the cell's title).
 */
function statNum(v: number, spread: number): string {
  if (!Number.isFinite(v) || Math.abs(v) >= 1000) return fmt(v);
  const d = Math.min(4, Math.max(2, 1 - Math.floor(Math.log10(spread || 1))));
  const s = v.toFixed(d);
  return (/^-0\.0+$/.test(s) ? s.slice(1) : s).replace("-", "−");
}

function Fragment({ label, row }: { label: string; row: number[] }) {
  return (
    <>
      <div className="hint" style={{ justifyContent: "flex-start", overflow: "hidden" }} title={label}>{label.slice(0, 8)}</div>
      {row.map((v, j) => (
        <div key={j} style={{ background: v >= 0 ? `color-mix(in srgb, var(--c1) ${Math.abs(v) * 60}%, transparent)` : `color-mix(in srgb, var(--c0) ${Math.abs(v) * 60}%, transparent)`, color: "var(--ink)" }}>{fmt(v, 2)}</div>
      ))}
    </>
  );
}

function PairRow({ r, feats, st, hovered, outIdx }: { r: number; feats: string[]; st: StatsData; hovered: number | null; outIdx: Set<number> | null }) {
  const k = feats.length;
  const n = st.pair.ids.data.length;
  const vals = st.pair.values.data;
  const cols = st.pair.values.shape[1] ?? k;
  return (
    <>
      <div className="pax" style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}>{feats[r]}</div>
      {feats.map((_, c) => (r === c ? <HistCell key={c} f={st.features.find((x) => x.name === feats[r])!} classes={st.classes.length || 1} /> : (
        <ScatterCell key={c} vals={vals} n={n} cols={cols} cx={c} cy={r} cls={st.pair.cls.data} hovered={hovered} outIdx={outIdx} ids={st.pair.ids.data} single={!st.classes.length} />
      )))}
    </>
  );
}

function HistCell({ f, classes }: { f: Feat; classes: number }) {
  const bins = f.hist[0]?.length ?? 0;
  const tot = Array.from({ length: bins }, (_, i) => f.hist.reduce((a, h) => a + h[i], 0));
  const mx = Math.max(1, ...tot);
  return (
    <div className="pcell">
      <div className="hist">
        {Array.from({ length: bins }, (_, i) => (
          <div key={i}>
            {f.hist.map((h, ci) => <motion.i key={ci} className={`k${(classes > 1 ? ci : 0) % 5}`} style={{ height: `${(h[i] / mx) * 92}%` }} initial={{ scaleY: 0 }} animate={{ scaleY: 1 }} transition={{ type: "spring", stiffness: 300, damping: 30, delay: i * 0.01 }} />)}
          </div>
        ))}
      </div>
    </div>
  );
}

function ScatterCell({ vals, n, cols, cx, cy, cls, hovered, outIdx, ids, single }: { vals: Float32Array; n: number; cols: number; cx: number; cy: number; cls: Int32Array; hovered: number | null; outIdx: Set<number> | null; ids: Int32Array; single: boolean }) {
  const xy = useMemo(() => {
    const a = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) { a[i * 2] = vals[i * cols + cx]; a[i * 2 + 1] = vals[i * cols + cy]; }
    return a;
  }, [vals, n, cols, cx, cy]);
  const lab = useMemo(() => Float32Array.from(cls), [cls]);
  return (
    <div className="pcell">
      <Scatter xy={xy} labels={lab} single={single} size={4.5} highlight={hovered} rings={outIdx} interactive
        onHover={(i) => useUi.setState({ insightHover: i == null ? null : ids[i] })} />
    </div>
  );
}
