// Rule-based data generator with a live preview (scatter + first rows). Same seed = same data.
import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { create } from "zustand";
import { Scatter } from "../../charts/Scatter";
import { engine } from "../../engine/client";
import { toast } from "../../engine/store";
import type { PreviewMsg } from "../../engine/types";
import { Icon } from "../../ui/Icon";
import { fmt, Seg, Slider } from "../../ui/controls";
import { afterDataChosen } from "./DataScreen";

interface NumCol { name: string; type: "numeric"; dist: "uniform" | "normal"; min: number; max: number; mean: number; std: number }
interface CatCol { name: string; type: "category"; values: { v: string; w: number }[] }
type Col = NumCol | CatCol;
interface Target { name: string; source: "random" | "rule"; rule: "line" | "circle" | "xor"; on: [string, string]; noise: number; classes: string[] }
interface Gen { rows: number; mode: "random" | "rules"; seed: number; columns: Col[]; target: Target }

const useGen = create<Gen & { set: (p: Partial<Gen>) => void }>((set) => ({
  rows: 500,
  mode: "rules",
  seed: 42,
  columns: [
    { name: "x1", type: "numeric", dist: "normal", min: -3, max: 3, mean: 0, std: 1 },
    { name: "x2", type: "numeric", dist: "uniform", min: -3, max: 3, mean: 0, std: 1 },
    { name: "color", type: "category", values: [{ v: "red", w: 40 }, { v: "green", w: 35 }, { v: "blue", w: 25 }] },
  ],
  target: { name: "label", source: "rule", rule: "line", on: ["x1", "x2"], noise: 0.1, classes: ["A", "B"] },
  set: (p) => set(p),
}));

function spec(g: Gen) {
  return { rows: g.rows, mode: g.mode, seed: g.seed, columns: g.columns, target: g.target, name: "Generated" };
}

export function Generator() {
  const { t } = useTranslation();
  const g = useGen();
  const [pv, setPv] = useState<PreviewMsg | null>(null);
  const timer = useRef<number>();
  useEffect(() => {
    clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      void engine.request<PreviewMsg>({ type: "preview_generator", generator: spec(g) }).then(setPv).catch(() => undefined);
    }, 80) as unknown as number;
  }, [g.rows, g.mode, g.seed, g.columns, g.target]);
  const nums = g.columns.filter((c): c is NumCol => c.type === "numeric");
  const setCol = (i: number, c: Col) => g.set({ columns: g.columns.map((x, j) => (j === i ? c : x)) });
  const generate = async () => {
    try {
      await engine.request({ type: "load_dataset", spec: { kind: "generator", generator: spec(g) } });
      afterDataChosen();
    } catch (e) {
      const err = e as { message?: { en: string; ru: string } };
      toast("gen", err.message ?? { en: "Couldn't generate.", ru: "Не удалось сгенерировать." });
    }
  };
  const bal = (pv?.extra.balance as number[] | undefined) ?? [];
  const total = bal.reduce((a, b) => a + b, 0) || 1;
  const axes = (pv?.extra.axes as string[] | undefined) ?? [];
  const ruleLine = useMemo(() => ruleBorder(g, axes), [g, axes]);
  return (
    <div className="gen" data-testid="generator">
      <aside className="glass" style={{ display: "flex", flexDirection: "column", minHeight: 0 }}>
        <div style={{ flex: 1, minHeight: 0, overflow: "auto", display: "flex", flexDirection: "column", gap: 14, padding: "16px 16px 12px" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <div className="prow"><label className="lbl" htmlFor="g-rows" style={{ color: "var(--ink2)", fontSize: 13 }}>{t("gen.rows")}</label><span className="mono" style={{ fontWeight: 600 }}>{g.rows}</span></div>
            <Slider id="g-rows" label={t("gen.rows")} value={g.rows} lo={20} hi={50000} log integer onChange={(v) => g.set({ rows: v })} />
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <span className="lbl" style={{ color: "var(--ink2)", fontSize: 13 }}>{t("gen.fill")}</span>
            <Seg value={g.mode} ariaLabel={t("gen.fill")} style={{ alignSelf: "flex-start" }} onChange={(v) => g.set({ mode: v })}
              options={[{ value: "random", label: t("gen.random") }, { value: "rules", label: t("gen.rules") }]} />
            <span className="hint">{t("gen.fillHint")}</span>
          </div>
          <div className="prow">
            <span className="lbl" style={{ color: "var(--ink2)", fontSize: 13 }}>{t("gen.columns", { count: g.columns.length + 1 })}</span>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span className="hint">{t("gen.seed")}</span>
              <input className="fld" style={{ width: 70, height: 28, textAlign: "right" }} value={g.seed} aria-label={t("gen.seed")}
                onChange={(e) => { const n = parseInt(e.target.value, 10); if (Number.isFinite(n)) g.set({ seed: Math.max(0, n) }); }} />
              <button className="btn ico sm" aria-label={t("gen.newSeed")} onClick={() => g.set({ seed: Math.floor(Math.random() * 10000) })}><Icon name="shuffle" /></button>
            </div>
          </div>
          {g.columns.map((c, i) => (
            <div className="gcol" key={i}>
              <div className="prow">
                <input className="fld" style={{ width: 120, height: 28, background: "transparent", border: "1px solid transparent", fontWeight: 600 }} value={c.name} aria-label={t("gen.colName")}
                  onChange={(e) => setCol(i, { ...c, name: e.target.value || c.name })} />
                <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  <Seg size="sm" value={c.type} ariaLabel={t("table.type")} onChange={(v) => setCol(i, v === "numeric"
                    ? { name: c.name, type: "numeric", dist: "normal", min: -3, max: 3, mean: 0, std: 1 }
                    : { name: c.name, type: "category", values: [{ v: "a", w: 50 }, { v: "b", w: 50 }] })}
                    options={[{ value: "numeric", label: "123" }, { value: "category", label: "Aa" }]} />
                  <button className="btn gh ico sm" aria-label={t("gen.removeCol", { name: c.name })} disabled={nums.length <= 2 && c.type === "numeric"}
                    onClick={() => g.set({ columns: g.columns.filter((_, j) => j !== i) })}><Icon name="trash" /></button>
                </div>
              </div>
              {c.type === "numeric" ? <NumEditor c={c} disabled={g.mode === "random"} onChange={(n) => setCol(i, n)} /> : <CatEditor c={c} disabled={g.mode === "random"} onChange={(n) => setCol(i, n)} />}
            </div>
          ))}
          <TargetEditor g={g} nums={nums.map((n) => n.name)} />
          <button className="btn sm" style={{ alignSelf: "flex-start" }} onClick={() => {
            let k = g.columns.length + 1, name = `x${k}`;
            while (g.columns.some((c) => c.name === name)) name = `x${++k}`;
            g.set({ columns: [...g.columns, { name, type: "numeric", dist: "normal", min: -3, max: 3, mean: 0, std: 1 }] });
          }}><Icon name="plus" />{t("gen.addColumn")}</button>
        </div>
        <div style={{ padding: "12px 16px", borderTop: "1px solid var(--line)" }}>
          <button className="btn pri" style={{ width: "100%", height: 40 }} onClick={() => void generate()} data-testid="generate"><Icon name="reset" />{t("gen.generate", { n: g.rows })}</button>
        </div>
      </aside>
      <section className="solid" style={{ display: "flex", flexDirection: "column", gap: 14, minHeight: 0, padding: "16px 18px" }}>
        <div className="prow">
          <h2 style={{ fontSize: 15, fontWeight: 600 }}>{t("gen.preview")}</h2>
          <div style={{ display: "flex", alignItems: "center", gap: 14, fontSize: 13, color: "var(--ink2)" }}>
            {(pv?.classes ?? []).map((c, i) => <span key={c} style={{ display: "flex", alignItems: "center", gap: 6 }}><span className={`dot k${i % 5} s${i % 4}`} />{c} · {Math.round(((bal[i] ?? 0) / total) * 100)}%</span>)}
            {bal.length > 1 && <span className="hint">{Math.max(...bal) / Math.max(1, Math.min(...bal)) < 1.5 ? t("gen.balanced") : t("gen.unbalanced")}</span>}
          </div>
        </div>
        <div style={{ position: "relative", flex: 1, minHeight: 0, margin: "0 0 22px 30px", borderLeft: "1px solid var(--line2)", borderBottom: "1px solid var(--line2)" }}>
          <Scatter xy={pv?.xy.data ?? null} labels={pv?.label.data ?? null} size={6} line={ruleLine} interactive ariaLabel={t("gen.preview")} />
          <span className="hint mono" style={{ position: "absolute", right: 0, top: "calc(100% + 4px)" }}>{axes[0] ?? "x1"} →</span>
          <span className="hint mono" style={{ position: "absolute", left: -30, top: 0 }}>{axes[1] ?? "x2"}</span>
          {g.target.source === "rule" && g.mode === "rules" && <span className="hint" style={{ position: "absolute", right: 8, top: 8, padding: "4px 8px", borderRadius: 6, background: "var(--solid2)" }}>{t("gen.dashed")}</span>}
        </div>
        <div className="prev-table">
          <div className="tr hd" style={{ gridTemplateColumns: `40px repeat(${pv?.columns.length ?? 1}, minmax(0, 1fr))` }}><div>#</div>{pv?.columns.map((c) => <div key={c}>{c}</div>)}</div>
          {pv?.rows.map((r, i) => (
            <div className="tr" key={i} style={{ gridTemplateColumns: `40px repeat(${pv.columns.length}, minmax(0, 1fr))` }}>
              <div style={{ color: "var(--ink3)" }}>{i + 1}</div>
              {r.slice(1).map((v, j) => {
                const isT = pv.columns[j] === g.target.name;
                const ci = isT ? pv.classes.indexOf(String(v)) : -1;
                return <div key={j} style={isT ? { display: "flex", alignItems: "center", gap: 6 } : undefined}>{isT && ci >= 0 && <span className={`dot k${ci % 5} s${ci % 4}`} />}{typeof v === "number" ? fmt(v) : String(v)}</div>;
              })}
            </div>
          ))}
          <div className="hint" style={{ padding: "7px 10px" }}>{t("gen.more", { n: Math.max(0, ((pv?.extra.n as number) ?? 0) - 5) })}</div>
        </div>
      </section>
    </div>
  );
}

function NumEditor({ c, onChange, disabled }: { c: NumCol; onChange: (c: NumCol) => void; disabled: boolean }) {
  const { t } = useTranslation();
  const field = (k: "min" | "max" | "mean" | "std", label: string) => (
    <label>{label}<input className="fld" defaultValue={String(c[k]).replace("-", "−")} disabled={disabled} aria-label={`${c.name} ${label}`}
      onBlur={(e) => { const v = Number(e.target.value.replace("−", "-").replace(",", ".")); if (Number.isFinite(v)) onChange({ ...c, [k]: k === "std" ? Math.abs(v) : v }); }}
      onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()} /></label>
  );
  const bars = c.dist === "normal" ? [8, 22, 52, 86, 100, 84, 50, 20, 8] : [70, 64, 72, 68, 66, 71, 65, 69, 67];
  return (
    <>
      <Seg size="sm" value={c.dist} ariaLabel={t("gen.dist")} style={{ alignSelf: "flex-start" }} onChange={(v) => onChange({ ...c, dist: v })}
        options={[{ value: "uniform", label: t("gen.uniform"), disabled }, { value: "normal", label: t("gen.normal"), disabled }]} />
      <div className="f2" key={c.dist}>
        {c.dist === "normal" ? <>{field("mean", t("gen.mean"))}{field("std", t("gen.std"))}</> : <>{field("min", t("gen.min"))}{field("max", t("gen.max"))}</>}
        <div className="hb" aria-hidden="true">{bars.map((h, i) => <i key={i} style={{ height: "100%", transform: `scaleY(${h / 100})` }} />)}</div>
      </div>
      <span className="hint">{t(c.dist === "normal" ? "gen.normalHint" : "gen.uniformHint")}</span>
    </>
  );
}

function CatEditor({ c, onChange, disabled }: { c: CatCol; onChange: (c: CatCol) => void; disabled: boolean }) {
  const { t } = useTranslation();
  const sum = c.values.reduce((a, b) => a + b.w, 0) || 1;
  return (
    <>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {c.values.map((v, i) => (
          <div key={i} style={{ display: "grid", gridTemplateColumns: "84px 1fr 56px 22px", gap: 10, alignItems: "center" }}>
            <input className="fld" style={{ height: 26, width: "100%" }} defaultValue={v.v} aria-label={t("gen.valueName")} disabled={disabled}
              onBlur={(e) => onChange({ ...c, values: c.values.map((x, j) => (j === i ? { ...x, v: e.target.value || x.v } : x)) })} />
            <div style={{ height: 6, borderRadius: 3, background: "var(--chip2)" }}><div style={{ width: `${(v.w / sum) * 100}%`, height: "100%", borderRadius: 3, background: "var(--ink3)", transition: "width .2s" }} /></div>
            <input className="fld" style={{ height: 26, width: "100%", textAlign: "right" }} defaultValue={String(v.w)} aria-label={t("gen.weight")} disabled={disabled}
              onBlur={(e) => { const w = Number(e.target.value); if (Number.isFinite(w) && w >= 0) onChange({ ...c, values: c.values.map((x, j) => (j === i ? { ...x, w } : x)) }); }} />
            <button className="btn gh ico sm" style={{ width: 22, height: 22, minHeight: 22 }} aria-label={t("gen.removeValue")} disabled={c.values.length <= 2 || disabled}
              onClick={() => onChange({ ...c, values: c.values.filter((_, j) => j !== i) })}><Icon name="close" size={12} /></button>
          </div>
        ))}
      </div>
      <button className="btn sm gh" style={{ alignSelf: "flex-start", paddingLeft: 0 }} disabled={disabled}
        onClick={() => onChange({ ...c, values: [...c.values, { v: `v${c.values.length + 1}`, w: 20 }] })}><Icon name="plus" />{t("gen.value")}</button>
    </>
  );
}

function TargetEditor({ g, nums }: { g: Gen & { set: (p: Partial<Gen>) => void }; nums: string[] }) {
  const { t } = useTranslation();
  const tg = g.target;
  const set = (p: Partial<Target>) => g.set({ target: { ...tg, ...p } });
  const [a, b] = [tg.on[0] && nums.includes(tg.on[0]) ? tg.on[0] : nums[0], tg.on[1] && nums.includes(tg.on[1]) ? tg.on[1] : nums[1]];
  const ruleText = tg.rule === "line" ? t("gen.ruleLine", { a, b }) : tg.rule === "circle" ? t("gen.ruleCircle", { a, b }) : t("gen.ruleXor", { a, b });
  return (
    <div className="gcol gtarget">
      <div className="prow">
        <span className="mono" style={{ fontWeight: 600, display: "flex", alignItems: "center", gap: 8 }}>{tg.name}
          <span className="tag" style={{ height: 20, background: "var(--acc)", color: "var(--acc-ink)", borderColor: "transparent", fontFamily: "var(--font)" }}>{t("table.target")}</span></span>
        <span className="tag">{t("gen.targetTag", { classes: tg.classes.join(" / ") })}</span>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        <span className="lbl">{t("gen.labelsFrom")}</span>
        <Seg size="sm" value={tg.source} ariaLabel={t("gen.labelsFrom")} style={{ alignSelf: "flex-start" }} onChange={(v) => set({ source: v })}
          options={[{ value: "random", label: t("gen.random") }, { value: "rule", label: t("gen.ruleOn", { a, b }) }]} />
      </div>
      {tg.source === "rule" && (
        <>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <span className="lbl">{t("gen.shape")}</span>
            <Seg size="sm" value={tg.rule} ariaLabel={t("gen.shape")} style={{ alignSelf: "flex-start" }} onChange={(v) => set({ rule: v })}
              options={[{ value: "line", label: t("gen.line") }, { value: "circle", label: t("gen.circle") }, { value: "xor", label: "XOR" }]} />
          </div>
          <span className="mono" style={{ fontSize: 13, padding: "8px 10px", borderRadius: 8, background: "var(--solid)", border: "1px solid var(--line)" }}>{ruleText}</span>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <div className="prow"><label className="lbl" htmlFor="g-noise">{t("gen.noise")}</label><span className="mono" style={{ fontWeight: 600 }}>{Math.round(tg.noise * 100)}%</span></div>
            <Slider id="g-noise" label={t("gen.noise")} value={tg.noise} lo={0} hi={0.5} step={0.01} onChange={(v) => set({ noise: v })} />
            <span className="hint">{t("gen.noiseHint")}</span>
          </div>
        </>
      )}
    </div>
  );
}

/** The rule's own border (user-defined geometry, not a model) drawn dashed on the preview. */
function ruleBorder(g: Gen, axes: string[]): [number, number, number, number] | null {
  if (g.mode !== "rules" || g.target.source !== "rule" || g.target.rule !== "line" || axes.length < 2) return null;
  const ca = g.columns.find((c) => c.name === axes[0]) as NumCol | undefined;
  const cb = g.columns.find((c) => c.name === axes[1]) as NumCol | undefined;
  if (!ca || !cb || ca.type !== "numeric" || cb.type !== "numeric") return null;
  const cen = (c: NumCol): [number, number] => (c.dist === "normal" ? [c.mean, Math.max(1e-9, c.std)] : [(c.min + c.max) / 2, Math.max(1e-9, (c.max - c.min) / 2)]);
  const [ma, sa] = cen(ca), [mb, sb] = cen(cb);
  // (a - ma)/sa + (b - mb)/sb = 0  ->  b = mb - sb (a - ma)/sa
  const x0 = ma - 3 * sa, x1 = ma + 3 * sa;
  return [x0, mb - (sb * (x0 - ma)) / sa, x1, mb - (sb * (x1 - ma)) / sa];
}
