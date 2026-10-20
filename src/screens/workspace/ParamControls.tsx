// One control per parameter kind, shared by the main panel and the Advanced drawer.
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useBi } from "../../i18n";
import type { ParamSchema, ParamValue } from "../../engine/types";
import { Icon } from "../../ui/Icon";
import { fmtParam, Info, roundSig, Seg, Slider, Toggle } from "../../ui/controls";

export function fmtNum(v: number, kind: "float" | "int" | string): string {
  if (!Number.isFinite(v)) return String(v);
  if (kind === "int") return String(Math.round(v));
  const a = Math.abs(v);
  if (a !== 0 && a < 1e-4) {
    const e = v.toExponential(0);
    return e.replace("e-", "e-").replace("e+", "e");
  }
  if (a >= 1e5) return v.toExponential(0).replace("e+", "e");
  if (Number.isInteger(v)) return v.toFixed(1);
  return String(roundSig(v, 3));
}

export function optLabel(o: ParamValue, opts: ParamValue[] | undefined): string {
  const quote = !!opts?.some((x) => x === null || typeof x === "number" || typeof x === "boolean");
  if (typeof o === "string") return quote ? `'${o}'` : o;
  return fmtParam(o);
}

export function isEnabled(p: ParamSchema, values: Record<string, ParamValue>): boolean {
  if (p.disabled) return false;
  if (!p.enabledIf) return true;
  return Object.entries(p.enabledIf).every(([k, allowed]) => allowed.some((a) => a === values[k]));
}

export function isVisible(p: ParamSchema, values: Record<string, ParamValue>): boolean {
  if (!p.visibleIf) return true;
  return Object.entries(p.visibleIf).every(([k, allowed]) => allowed.some((a) => a === values[k]));
}

function parseNum(s: string): number | null {
  const t = s.trim().replace("−", "-").replace(",", ".");
  if (!t) return null;
  const v = Number(t);
  return Number.isFinite(v) ? v : null;
}

/** Inline-editable value chip (click to type, Enter to apply). */
export function ValueChip({ value, kind, onCommit, disabled, label, lo, hi }: {
  value: ParamValue; kind: string; onCommit: (v: number) => void; disabled?: boolean; label: string; lo?: number; hi?: number;
}) {
  const [txt, setTxt] = useState<string | null>(null);
  const shown = typeof value === "number" ? fmtNum(value, kind) : fmtParam(value);
  const n = txt != null ? parseNum(txt) : null;
  const bad = txt != null && (n == null || (kind === "int" && !Number.isInteger(n)));
  const commit = () => {
    if (txt != null && n != null && !bad) onCommit(kind === "int" ? Math.round(n) : n);
    setTxt(null);
  };
  return (
    <input className={`val${bad ? " bad" : ""}`} aria-label={label} disabled={disabled} value={txt ?? shown} size={Math.max(3, (txt ?? shown).length)}
      onFocus={(e) => { setTxt(shown); requestAnimationFrame(() => e.target.select()); }}
      onChange={(e) => setTxt(e.target.value)} onBlur={commit}
      onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); if (e.key === "Escape") { setTxt(null); (e.target as HTMLInputElement).blur(); } }}
      title={lo != null && hi != null ? `${lo} … ${hi}` : undefined} />
  );
}

export interface ParamProps {
  p: ParamSchema;
  value: ParamValue;
  values: Record<string, ParamValue>;
  onChange: (v: ParamValue, commit: boolean) => void;
  locked?: boolean;
  classes?: string[];
  tip?: string | null;
  compact?: boolean; // Advanced drawer: text fields instead of sliders
}

export function ParamRow({ p, value, values, onChange, locked, classes, tip, compact }: ParamProps) {
  const { t } = useTranslation();
  const bi = useBi();
  const enabled = isEnabled(p, values) && !locked;
  const id = `p-${p.name}`;
  const label = bi(p.label);
  const info = (
    <Info label={t("params.about", { name: p.name })}>
      <b>{label} — <span className="mono">{p.name}</span></b>
      <br />
      {tip || bi(p.hint)}
      <br />
      <span style={{ color: "var(--ink3)" }}>{t("params.default", { v: optLabel(p.default, p.options) })}</span>
    </Info>
  );
  return (
    <div className={`param${enabled ? "" : " off"}`} data-param={p.name}>
      <div className="prow">
        <label className="pn" htmlFor={id}>{label} <span className="pk">{p.name}</span>{info}</label>
        {(p.kind === "float" || p.kind === "int") && !compact && (
          <ValueChip value={value} kind={p.kind} label={p.name} disabled={!enabled} lo={p.lo} hi={p.hi}
            onCommit={(v) => onChange(clamp(v, p), true)} />
        )}
        {p.kind === "bool" && <Toggle id={id} label={p.name} checked={!!value} disabled={!enabled} onChange={(v) => onChange(v, true)} />}
      </div>
      <Control p={p} value={value} onChange={onChange} enabled={enabled} id={id} classes={classes} compact={compact} />
      {p.onlyTag && !isEnabled(p, values) && !p.disabled && <span className="tag" style={{ height: 20, alignSelf: "flex-start" }}>{bi(p.onlyTag)}</span>}
      {p.disabled && <span className="tag" style={{ height: 20, alignSelf: "flex-start" }}>{bi(p.disabled)}</span>}
      <div className="hint">{bi(p.hint)}</div>
    </div>
  );
}

function clamp(v: number, p: ParamSchema) {
  let x = v;
  if (p.lo != null) x = Math.max(p.lo, x);
  if (p.hi != null) x = Math.min(p.hi, x);
  return x;
}

export function Control({ p, value, onChange, enabled, id, classes, compact }: {
  p: ParamSchema; value: ParamValue; onChange: (v: ParamValue, commit: boolean) => void; enabled: boolean; id: string; classes?: string[]; compact?: boolean;
}) {
  const { t } = useTranslation();
  const bi = useBi();
  const [local, setLocal] = useState<ParamValue>(value);
  const dragging = useRef(false);
  useEffect(() => {
    if (!dragging.current) setLocal(value);
  }, [value]);

  if (p.kind === "bool" || p.kind === "fixed") return null;

  if ((p.kind === "float" || p.kind === "int") && p.lo != null && p.hi != null) {
    const isNone = local === null;
    if (compact) {
      return <NumField p={p} value={value} enabled={enabled} onChange={(v) => onChange(v, true)} id={id} />;
    }
    const v = typeof local === "number" ? local : (typeof p.initial === "number" ? p.initial : (p.lo + p.hi) / 2);
    return (
      <>
        <Slider id={id} label={p.name} value={v} lo={p.lo} hi={p.hi} log={p.log} step={p.step} integer={p.kind === "int"} disabled={!enabled || isNone}
          onChange={(x) => { dragging.current = true; setLocal(x); onChange(x, false); }}
          onCommit={(x) => { dragging.current = false; onChange(x, true); }} />
        {p.log && !p.noneLabel && (
          <div className="prow hint mono" style={{ marginTop: -4 }}><span>{Number.isInteger(p.lo) ? p.lo : fmtNum(p.lo, p.kind)}</span><span>{Number.isInteger(p.hi) ? p.hi : fmtNum(p.hi, p.kind)}</span></div>
        )}
        {p.noneLabel && (
          <label className="prow hint" style={{ justifyContent: "flex-start", gap: 8 }}>
            <Toggle checked={isNone} disabled={!enabled} label={bi(p.noneLabel)}
              onChange={(on) => onChange(on ? null : (typeof p.initial === "number" ? p.initial : p.default ?? Math.round((p.lo! + p.hi!) / 4)) as ParamValue, true)} />
            {bi(p.noneLabel)}
          </label>
        )}
      </>
    );
  }

  if (p.kind === "enum") {
    const opts = p.options ?? [];
    return (
      <Seg size="sm" mono ariaLabel={p.name} value={value as never} style={{ alignSelf: "flex-start", flexWrap: "wrap" }}
        options={opts.map((o) => ({ value: o as never, label: optLabel(o, opts), disabled: !enabled }))}
        onChange={(v) => onChange(v as ParamValue, true)} />
    );
  }

  if (p.kind === "mixed") {
    const opts = p.options ?? [];
    const isOpt = opts.some((o) => o === value);
    const numVal = typeof value === "number" && !isOpt ? value : null;
    return (
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
        <Seg size="sm" mono ariaLabel={p.name} value={(isOpt ? value : "__n") as never}
          options={[...opts.map((o) => ({ value: o as never, label: optLabel(o, opts), disabled: !enabled })),
            ...(p.lo != null ? [{ value: "__n" as never, label: numVal != null ? fmtNum(numVal, p.step && p.step >= 1 ? "int" : "float") : t("params.custom"), disabled: !enabled }] : [])]}
          onChange={(v) => {
            if (v === ("__n" as never)) {
              const start = typeof p.initial === "number" ? p.initial : p.lo != null && p.hi != null ? (p.log ? Math.sqrt(p.lo * p.hi) : p.step && p.step >= 1 ? Math.round((p.lo + p.hi) / 4) : (p.lo + p.hi) / 2) : 1;
              onChange(numVal ?? roundSig(start, 2), true);
            } else onChange(v as ParamValue, true);
          }} />
        {numVal != null && p.lo != null && (
          <NumField p={p} value={numVal} enabled={enabled} onChange={(v) => onChange(v, true)} id={id} small />
        )}
      </div>
    );
  }

  if (p.kind === "layers") {
    const layers = Array.isArray(value) ? (value as number[]) : [100];
    const set = (arr: number[]) => onChange(arr, true);
    return (
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
        {layers.map((n, i) => (
          <span key={i} className="lyr">
            {n}
            <button aria-label={t("params.fewer", { i: i + 1 })} disabled={!enabled}
              onClick={() => set(n <= 1 ? layers.filter((_, j) => j !== i).concat(layers.length === 1 ? [1] : []) : layers.map((x, j) => (j === i ? Math.max(1, x > 16 ? Math.round(x / 2) : x - 1) : x)))}>
              <Icon name="minus" size={14} />
            </button>
            <button aria-label={t("params.more", { i: i + 1 })} disabled={!enabled}
              onClick={() => set(layers.map((x, j) => (j === i ? Math.min(256, x >= 16 ? x * 2 : x + 1) : x)))}>
              <Icon name="plus" size={14} />
            </button>
          </span>
        ))}
        {layers.length < 4 && (
          <button className="btn sm" disabled={!enabled} onClick={() => set([...layers, layers[layers.length - 1] ?? 8])}>
            <Icon name="plus" />{t("params.layer")}
          </button>
        )}
      </div>
    );
  }

  if (p.kind === "priors") {
    const k = classes?.length ?? 2;
    const arr = Array.isArray(value) ? (value as number[]) : null;
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <label className="prow hint" style={{ justifyContent: "flex-start", gap: 8 }}>
          <Toggle checked={arr === null} disabled={!enabled} label={bi(p.noneLabel)}
            onChange={(on) => onChange(on ? null : Array.from({ length: k }, () => Number((1 / k).toFixed(3))), true)} />
          {bi(p.noneLabel)}
        </label>
        {arr && (
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            {Array.from({ length: k }, (_, i) => (
              <label key={i} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
                <span className={`dot k${i % 5} s${i % 4}`} />{classes?.[i] ?? i}
                <input className="fld" style={{ width: 64 }} defaultValue={String(arr[i] ?? 0)} aria-label={`prior ${classes?.[i] ?? i}`}
                  onBlur={(e) => {
                    const n = parseNum(e.target.value);
                    if (n == null || n < 0) return;
                    const next = Array.from({ length: k }, (_, j) => (j === i ? n : arr[j] ?? 0));
                    onChange(next, true);
                  }} />
              </label>
            ))}
            <span className="hint">{t("params.priorsNorm")}</span>
          </div>
        )}
      </div>
    );
  }

  return <NumField p={p} value={value} enabled={enabled} onChange={(v) => onChange(v, true)} id={id} />;
}

function NumField({ p, value, enabled, onChange, id, small }: { p: ParamSchema; value: ParamValue; enabled: boolean; onChange: (v: ParamValue) => void; id: string; small?: boolean }) {
  const shown = value === null ? "None" : typeof value === "number" ? fmtNum(value, p.kind === "int" || (p.step ?? 0) >= 1 ? "int" : "float") : fmtParam(value);
  const [txt, setTxt] = useState<string | null>(null);
  const parsed = txt == null ? null : txt.trim() === "None" && p.noneLabel ? "none" : parseNum(txt);
  const bad = txt != null && parsed == null;
  return (
    <input id={id} className={`fld${bad ? " bad" : ""}`} style={{ width: small ? 90 : 110, alignSelf: small ? undefined : "flex-start", textAlign: "right" }} aria-label={p.name}
      disabled={!enabled} value={txt ?? shown} onFocus={() => setTxt(shown)} onChange={(e) => setTxt(e.target.value)}
      onBlur={() => {
        if (txt != null && !bad) onChange(parsed === "none" ? null : clamp(parsed as number, p));
        setTxt(null);
      }}
      onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()} />
  );
}
