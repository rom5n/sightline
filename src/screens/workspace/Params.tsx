import { AnimatePresence, motion } from "motion/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useNav } from "../../app/nav";
import { datasetLabel } from "../../app/Shell";
import { engine } from "../../engine/client";
import { currentParams, resetParams, setParams, trainPhase, useSession } from "../../engine/store";
import type { ParamSchema, ParamValue } from "../../engine/types";
import { useBi } from "../../i18n";
import { Icon } from "../../ui/Icon";
import { fmtParam, SPRING, Toggle } from "../../ui/controls";
import { DatasetThumb } from "../data/Thumb";
import { Control, isEnabled, isVisible, ParamRow } from "./ParamControls";

function useValues() {
  const model = useSession((s) => s.model);
  const byModel = useSession((s) => (s.model ? s.paramsByModel[s.model] : undefined));
  const schema = useSession((s) => (s.model ? s.models[s.model] : undefined));
  return useMemo(() => currentParams(), [model, byModel, schema]);
}

/** Debounce live param changes while a slider drags; expensive models only retrain on release. */
function useParamSetter() {
  const schema = useSession((s) => (s.model ? s.models[s.model] : undefined));
  const timer = useRef<number>();
  const pending = useRef<Record<string, ParamValue>>({});
  return (name: string, v: ParamValue, commit: boolean) => {
    const live = !schema?.expensive;
    pending.current[name] = v;
    clearTimeout(timer.current);
    const flush = () => {
      const patch = pending.current;
      pending.current = {};
      setParams(patch, true);
    };
    if (commit) flush();
    else if (live) timer.current = setTimeout(flush, 60) as unknown as number;
    else useSession.setState((s) => ({ paramsByModel: { ...s.paramsByModel, [s.model!]: { ...(s.paramsByModel[s.model!] ?? {}), [name]: v } } }));
  };
}

export function ParamsAside({ onAdvanced, onResetLayout }: { onAdvanced: () => void; onResetLayout: () => void }) {
  const { t } = useTranslation();
  const schema = useSession((s) => (s.model ? s.models[s.model] : undefined));
  const ds = useSession((s) => s.dataset);
  const phase = useSession(trainPhase);
  const values = useValues();
  const set = useParamSetter();
  const go = useNav((s) => s.go);
  const locked = !!schema?.lockWhileTraining && (phase === "training" || phase === "paused");
  const tipKey = (p: ParamSchema) => `paramTips.${schema?.name}.${p.name}`;
  const changeHint = !schema ? "" : locked ? t("params.locked") : schema.lossKind === "depth" || !schema.iterative ? t("params.refit") : schema.task === "clustering" ? t("params.restart") : t("params.retrain");
  return (
    <aside className="glass aside" aria-label={t("params.title")} data-guide="params">
      <div style={{ display: "flex", flexDirection: "column", gap: 10, paddingBottom: 16 }}>
        <div className="prow">
          <span className="lbl">{t("params.dataset")}</span>
          <button className="btn sm gh" style={{ color: "var(--acc-t)" }} onClick={() => go("data")}>{t("params.change")}</button>
        </div>
        {ds ? (
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <DatasetThumb />
            <div style={{ display: "flex", flexDirection: "column", minWidth: 0 }}>
              <span style={{ fontSize: 14, fontWeight: 600 }}>{datasetLabel(t, ds.name, ds.preset)}</span>
              <span className="hint">{datasetLine(t, ds, schema?.task)}</span>
            </div>
          </div>
        ) : (
          <span className="hint" style={{ fontSize: 13 }}>{t("params.noData")}</span>
        )}
      </div>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", flexWrap: "wrap", gap: "2px 8px", padding: "14px 0 4px", borderTop: "1px solid var(--line)" }}>
        <h2 style={{ fontSize: 15, fontWeight: 600 }}>{t("params.main")}</h2>
        <span className="hint" style={{ minWidth: 0 }}>{changeHint}</span>
      </div>
      {schema?.main.filter((p) => isVisible(p, values)).map((p) => (
        <ParamRow key={p.name} p={p} value={values[p.name] ?? p.initial} values={values} locked={locked} classes={ds?.classes}
          tip={t(tipKey(p), { defaultValue: "" }) || null} onChange={(v, c) => set(p.name, v, c)} />
      ))}
      <div style={{ marginTop: "auto", display: "flex", flexDirection: "column", gap: 8, paddingTop: 14, borderTop: "1px solid var(--line)" }}>
        <button className="btn" style={{ justifyContent: "space-between", width: "100%" }} onClick={onAdvanced} disabled={!schema || !schema.advanced.length} data-testid="advanced-btn">
          <span style={{ display: "flex", alignItems: "center", gap: 8 }}><Icon name="sliders" />{t("params.advanced")}</span>
          <span className="hint">{t("params.more", { count: schema?.advanced.length ?? 0 })}</span>
        </button>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          <button className="btn gh sm" onClick={() => resetParams()} disabled={locked}>{t("params.reset")}</button>
          <button className="btn gh sm" onClick={onResetLayout}><Icon name="layout" />{t("params.resetLayout")}</button>
        </div>
      </div>
    </aside>
  );
}

export function datasetLine(t: (k: string, o?: Record<string, unknown>) => string, ds: NonNullable<ReturnType<typeof useSession.getState>["dataset"]>, task?: string) {
  const parts = [t("ds.rows", { count: ds.n_rows }), t("ds.features", { count: ds.n_features })];
  if (task === "clustering" || task === "reduction") parts.push(t("ds.targetIgnored"));
  else if (ds.task === "classification") parts.push(t("ds.classes", { count: ds.classes.length }));
  if ((task === "classification" || task === "regression") && ds.n_test) parts.push(t("ds.heldOut"));
  return parts.join(" · ");
}

// ------------------------------------------------------------------ Advanced drawer

export function AdvancedDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { t } = useTranslation();
  const bi = useBi();
  const schema = useSession((s) => (s.model ? s.models[s.model] : undefined));
  const ds = useSession((s) => s.dataset);
  const committed = useValues();
  const [draft, setDraft] = useState<Record<string, ParamValue>>({});
  const [seed, setSeed] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const [closed, setClosed] = useState<Record<string, boolean>>({ repro: true, other: true });
  const panel = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (open) {
      setDraft({});
      setSeed(null);
      setQ("");
      setTimeout(() => panel.current?.querySelector<HTMLElement>("input,button")?.focus(), 50);
    }
  }, [open]);
  if (!schema) return null;
  const values = { ...committed, ...draft };
  const changed = schema.advanced.filter((p) => JSON.stringify(values[p.name]) !== JSON.stringify(p.default) && !p.disabled);
  const filtered = schema.advanced.filter((p) => !q || (p.name + bi(p.label) + bi(p.hint)).toLowerCase().includes(q.toLowerCase()));
  const sections: string[] = [];
  for (const p of filtered) if (!sections.includes(p.section ?? "other")) sections.push(p.section ?? "other");
  const apply = () => {
    if (Object.keys(draft).length) setParams(draft, true);
    const n = seed != null ? parseInt(seed, 10) : NaN;
    if (Number.isFinite(n) && n >= 0 && n !== ds?.split_seed) engine.send({ type: "set_split_seed", seed: n });
    onClose();
  };
  const resetAll = () => {
    const d: Record<string, ParamValue> = {};
    for (const p of schema.advanced) if (!p.disabled) d[p.name] = p.initial;
    setDraft(d);
  };
  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div className="adv-dim" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.25 }} onClick={onClose} />
          <motion.aside ref={panel} role="dialog" aria-modal="true" aria-label={t("adv.title")} className="adv" data-testid="advanced"
            initial={{ x: 520, opacity: 0.6 }} animate={{ x: 0, opacity: 1 }} exit={{ x: 520, opacity: 0 }} transition={SPRING}
            onKeyDown={(e) => e.key === "Escape" && onClose()}>
            <div style={{ display: "flex", flexDirection: "column", gap: 12, padding: "20px 22px 14px", borderBottom: "1px solid var(--line)" }}>
              <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 12 }}>
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  <h2 style={{ fontSize: 18, fontWeight: 600 }}>{t("adv.title")}</h2>
                  <span className="mono" style={{ fontSize: 13, color: "var(--ink2)" }}>{t("adv.sub", { model: schema.name, count: schema.advanced.length })}</span>
                </div>
                <button className="btn ico gh" aria-label={t("adv.close")} onClick={onClose}><Icon name="close" /></button>
              </div>
              <p style={{ margin: 0, fontSize: 13, color: "var(--ink2)", lineHeight: 1.5 }}>{t("adv.intro")}</p>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <label style={{ flex: 1, display: "flex", alignItems: "center", gap: 8, height: 34, padding: "0 12px", borderRadius: 10, background: "var(--chip)", border: "1px solid var(--line)" }}>
                  <Icon name="search" style={{ color: "var(--ink3)" }} />
                  <input type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder={t("adv.filter")} aria-label={t("adv.filterLabel")}
                    style={{ all: "unset", flex: 1, font: "400 13px var(--font)", color: "var(--ink)" }} />
                </label>
                <span className="tag" style={{ color: "var(--ink)" }}>{changed.length > 0 && <span className="ch" />}{t("adv.changed", { count: changed.length })}</span>
              </div>
            </div>
            <div style={{ flex: 1, minHeight: 0, overflow: "auto", padding: "0 22px 12px" }}>
              {sections.map((sec) => {
                const items = filtered.filter((p) => (p.section ?? "other") === sec);
                const isClosed = !!closed[sec] && !q;
                return (
                  <div key={sec}>
                    <button className="sec" aria-expanded={!isClosed} onClick={() => setClosed((c) => ({ ...c, [sec]: !isClosed }))}>
                      <h3>{t(`adv.sec.${sec}`)}</h3>
                      <span className="hint" style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        {isClosed ? items.map((p) => p.name).join(" · ") : t("adv.settings", { count: items.length })}
                        <Icon name={isClosed ? "chevDown" : "chevUp"} />
                      </span>
                    </button>
                    {!isClosed && items.map((p) => (
                      <AdvRow key={p.name} p={p} values={values} classes={ds?.classes} onChange={(v) => setDraft((d) => ({ ...d, [p.name]: v }))} />
                    ))}
                  </div>
                );
              })}
              {!filtered.length && <p className="hint" style={{ padding: 20 }}>{t("adv.none")}</p>}
              {(schema.task === "classification" || schema.task === "regression") && !q && (
                <div>
                  <div className="sec" role="heading" aria-level={3}><h3>{t("adv.sec.split")}</h3></div>
                  <div className="row" style={{ borderTop: 0 }} data-param="split_seed">
                    <span className="nm">{t("adv.splitSeed")} <span className="k">train_test_split(random_state)</span></span>
                    <input className="fld" aria-label={t("adv.splitSeed")} value={seed ?? String(ds?.split_seed ?? 42)} onChange={(e) => setSeed(e.target.value.replace(/[^0-9]/g, ""))} />
                    <span className="hint">{t("adv.splitHint")}</span>
                  </div>
                </div>
              )}
            </div>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, padding: "14px 22px", borderTop: "1px solid var(--line)" }}>
              <button className="btn gh sm" onClick={resetAll}>{t("adv.resetAll")}</button>
              <div style={{ display: "flex", gap: 8 }}>
                <button className="btn" onClick={onClose}>{t("common.cancel")}</button>
                <button className="btn pri" onClick={apply} data-testid="adv-apply">{t("adv.apply")}</button>
              </div>
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}

function AdvRow({ p, values, onChange, classes }: { p: ParamSchema; values: Record<string, ParamValue>; onChange: (v: ParamValue) => void; classes?: string[] }) {
  const bi = useBi();
  const { t } = useTranslation();
  const value = values[p.name] ?? p.initial;
  const enabled = isEnabled(p, values);
  const changed = !p.disabled && JSON.stringify(value) !== JSON.stringify(p.default);
  const tag = p.disabled ? bi(p.disabled) : !enabled && p.onlyTag ? bi(p.onlyTag) : null;
  return (
    <div className={`row${enabled ? "" : " off"}`} data-param={p.name}>
      <span className="nm">{changed && <span className="ch" aria-label={t("adv.changed")} />}{bi(p.label)} <span className="k">{p.name}</span></span>
      <div className="ctlx">
        {p.kind === "bool" ? (
          <Toggle checked={!!value} disabled={!enabled} label={p.name} onChange={(v) => onChange(v)} />
        ) : p.kind === "fixed" ? (
          <input className="fld" disabled value={fmtParam(value)} aria-label={p.name} />
        ) : (
          <Control p={p} value={value} enabled={enabled} id={`adv-${p.name}`} classes={classes} compact onChange={(v) => onChange(v)} />
        )}
      </div>
      <span className="hint">{tag && <span className="tag" style={{ height: 20, marginRight: 6 }}>{tag}</span>}{bi(p.hint)}</span>
    </div>
  );
}
