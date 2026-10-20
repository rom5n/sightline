import { motion, useReducedMotion } from "motion/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { PRESET_ORDER, presetFit } from "../../app/catalog";
import { useSettings } from "../../app/settings";
import { Scatter } from "../../charts/Scatter";
import { engine } from "../../engine/client";
import { loadPreset, useSession } from "../../engine/store";
import type { PresetInfo, PreviewMsg } from "../../engine/types";
import { Icon } from "../../ui/Icon";
import { fmt, Slider } from "../../ui/controls";
import { afterDataChosen } from "./DataScreen";

const memo: Record<string, Record<string, number>> = {};

export function Presets() {
  const presets = useSession((s) => s.presets);
  const ready = useSession((s) => s.ready);
  const byId = useMemo(() => Object.fromEntries(presets.map((p) => [p.id, p])), [presets]);
  if (!ready || !presets.length) return <div className="presets"><span className="hint">…</span></div>;
  return (
    <div className="presets" data-testid="presets">
      {PRESET_ORDER.map((id) => byId[id] && <PresetCard key={id} p={byId[id]} />)}
    </div>
  );
}

function PresetCard({ p }: { p: PresetInfo }) {
  const { t } = useTranslation();
  const reduce = useReducedMotion();
  const ds = useSession((s) => s.dataset);
  const model = useSession((s) => s.model);
  const inUse = ds?.kind === "preset" && ds.preset === p.id;
  const defaults = Object.fromEntries(Object.entries(p.settings).map(([k, v]) => [k, v.default]));
  const [vals, setVals] = useState<Record<string, number>>(() => memo[p.id] ?? (inUse ? { ...defaults, ...ds!.settings } : defaults));
  const [pv, setPv] = useState<PreviewMsg | null>(null);
  const timer = useRef<number>();
  const loadTimer = useRef<number>();
  useEffect(() => {
    memo[p.id] = vals;
    clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      void engine.request<PreviewMsg>({ type: "preview_preset", preset: p.id, settings: vals }).then(setPv).catch(() => undefined);
    }, 40) as unknown as number;
  }, [vals, p.id]);
  const fit = presetFit(model, p.id);
  const fitColor = fit.level === "good" ? "var(--acc-t)" : "var(--warn)";
  const use = async () => {
    await loadPreset(p.id, vals);
    useSettings.getState().pushRecentData({ name: p.id, preset: p.id, settings: vals, rows: vals.n_samples ?? 0, at: Date.now() });
    afterDataChosen();
  };
  const change = (k: string, v: number) => {
    const next = { ...vals, [k]: v };
    setVals(next);
    if (inUse) {
      clearTimeout(loadTimer.current);
      loadTimer.current = setTimeout(() => void loadPreset(p.id, next), 120) as unknown as number;
    }
  };
  const isReg = p.task === "regression";
  const line = isReg ? ((pv?.extra.line as [number, number, number, number] | undefined) ?? null) : null;
  return (
    <motion.article className={`glass pcard${inUse ? " sel" : ""}`} data-testid={`preset-${p.id}`} whileHover={reduce ? undefined : { y: -2 }} transition={{ duration: 0.15 }}>
      <div className="thumb"><Scatter xy={pv?.xy.data ?? null} labels={pv?.label.data ?? null} single={isReg} size={5} line={line} interactive ariaLabel={t(`presets.${p.id}.name`)} /></div>
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 10 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 0 }}>
          <span style={{ fontSize: 15, fontWeight: 600 }}>{t(`presets.${p.id}.name`)}</span>
          <span className="hint">{t(`presets.${p.id}.desc`)}</span>
        </div>
        <span className="tag">{t(`presets.${p.id}.task`)}</span>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        {Object.entries(p.settings).map(([k, s]) => (
          <div className="set" key={k}>
            <label htmlFor={`${p.id}-${k}`}>{k}</label>
            <Slider id={`${p.id}-${k}`} label={k} value={vals[k] ?? s.default} lo={s.min} hi={s.max} step={s.step} integer={s.step >= 1}
              onChange={(v) => change(k, v)} />
            <span className="v">{s.step >= 1 ? Math.round(vals[k] ?? s.default) : fmt(vals[k] ?? s.default, 2)}</span>
          </div>
        ))}
      </div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, paddingTop: 8, borderTop: "1px solid var(--line)" }}>
        <span className="fit" style={{ color: fitColor }}>{t(`fitBadge.${fit.key}`)}</span>
        {inUse ? (
          <button className="btn sm" style={{ color: "var(--acc-t)", borderColor: "var(--acc-t)" }} onClick={() => void use()}><Icon name="check" />{t("data.inUse")}</button>
        ) : (
          <button className="btn sm pri" onClick={() => void use()} disabled={fit.level === "wrong" && !!model} title={fit.level === "wrong" ? t(`fitBadge.${fit.key}`) : undefined}
            data-testid={`use-${p.id}`}>{t("data.use")}</button>
        )}
      </div>
    </motion.article>
  );
}
