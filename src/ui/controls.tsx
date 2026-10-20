// Small, reusable controls that follow the design (seg, toggle, slider, info tooltip, numbers).
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useEffect, useId, useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { create } from "zustand";
import { Icon } from "./Icon";

export const SPRING = { type: "spring", stiffness: 300, damping: 30 } as const;
export const MICRO = { duration: 0.15, ease: [0.2, 0.8, 0.2, 1] } as const;

// ------------------------------------------------------------------ segmented control

export interface SegOption<T> { value: T; label: ReactNode; title?: string; disabled?: boolean }

export function Seg<T extends string | number | boolean | null>({
  value, options, onChange, size, mono, ariaLabel, multi, values, style,
}: {
  value?: T; options: SegOption<T>[]; onChange: (v: T) => void; size?: "sm"; mono?: boolean; ariaLabel?: string;
  multi?: boolean; values?: T[]; style?: CSSProperties;
}) {
  const id = useId();
  const reduce = useReducedMotion();
  return (
    <div className={`seg${size ? " " + size : ""}${mono ? " mono" : ""}`} role={multi ? "group" : "radiogroup"} aria-label={ariaLabel} style={style}>
      {options.map((o, i) => {
        const on = multi ? !!values?.includes(o.value) : o.value === value;
        return (
          <button
            key={String(o.value) + i}
            type="button"
            className={on ? "on" : ""}
            role={multi ? "checkbox" : "radio"}
            aria-checked={on}
            title={o.title}
            disabled={o.disabled}
            onClick={() => onChange(o.value)}
            onKeyDown={(e) => {
              if (multi) return;
              if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
              const enabled = options.filter((x) => !x.disabled);
              const idx = enabled.findIndex((x) => x.value === value);
              const next = enabled[(idx + (e.key === "ArrowRight" ? 1 : enabled.length - 1)) % enabled.length];
              if (next) onChange(next.value);
              e.preventDefault();
            }}
          >
            {on && (multi ? <span className="seg-ind" /> : <motion.span className="seg-ind" layoutId={reduce ? undefined : `seg-${id}`} transition={SPRING} />)}
            <span className="seg-lbl">{o.label}</span>
          </button>
        );
      })}
    </div>
  );
}

// ------------------------------------------------------------------ toggle

export function Toggle({ checked, onChange, id, label, disabled }: { checked: boolean; onChange: (v: boolean) => void; id?: string; label?: string; disabled?: boolean }) {
  return (
    <span className="tg">
      <input id={id} type="checkbox" role="switch" aria-label={label} checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      <span />
    </span>
  );
}

// ------------------------------------------------------------------ slider (linear or log)

export function Slider({
  value, lo, hi, log, step, onChange, onCommit, id, label, disabled, integer,
}: {
  value: number; lo: number; hi: number; log?: boolean; step?: number; onChange: (v: number) => void; onCommit?: (v: number) => void;
  id?: string; label?: string; disabled?: boolean; integer?: boolean;
}) {
  const N = 1000;
  const toPos = (v: number) => {
    const c = Math.min(hi, Math.max(lo, v));
    return log ? (Math.log(c) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) : (c - lo) / (hi - lo || 1);
  };
  const fromPos = (p: number) => {
    let v = log ? Math.exp(Math.log(lo) + p * (Math.log(hi) - Math.log(lo))) : lo + p * (hi - lo);
    if (integer) v = Math.round(log ? roundSig(v, 2) : v);
    else if (log) v = roundSig(v, 2);
    else if (step) v = Math.round(v / step) * step;
    return Number(v.toFixed(10));
  };
  const pos = toPos(value);
  return (
    <input
      id={id}
      className="rng"
      type="range"
      min={0}
      max={N}
      step={1}
      value={Math.round(pos * N)}
      aria-label={label}
      aria-valuetext={String(value)}
      disabled={disabled}
      style={{ ["--p" as string]: `${pos * 100}%` }}
      onChange={(e) => onChange(fromPos(Number(e.target.value) / N))}
      onPointerUp={(e) => onCommit?.(fromPos(Number((e.target as HTMLInputElement).value) / N))}
      onKeyUp={(e) => onCommit?.(fromPos(Number((e.target as HTMLInputElement).value) / N))}
    />
  );
}

export function roundSig(v: number, d: number) {
  if (v === 0) return 0;
  const p = Math.pow(10, d - Math.ceil(Math.log10(Math.abs(v))));
  return Math.round(v * p) / p;
}

// ------------------------------------------------------------------ tooltips (300 ms delay)

interface TipState { rect: DOMRect | null; content: ReactNode; key: string | null; interactive: boolean }
const useTip = create<TipState>(() => ({ rect: null, content: null, key: null, interactive: false }));
let tipTimer: number | undefined;
let hideTimer: number | undefined;

export function showTip(el: HTMLElement, content: ReactNode, key: string, delay = 300, interactive = false) {
  clearTimeout(tipTimer);
  clearTimeout(hideTimer);
  tipTimer = setTimeout(() => useTip.setState({ rect: el.getBoundingClientRect(), content, key, interactive }), delay) as unknown as number;
}
export function hideTip(delay = 80) {
  clearTimeout(tipTimer);
  clearTimeout(hideTimer);
  hideTimer = setTimeout(() => useTip.setState({ rect: null, key: null }), delay) as unknown as number;
}

export function tipHandlers(content: ReactNode, key: string) {
  return {
    onMouseEnter: (e: React.MouseEvent<HTMLElement>) => showTip(e.currentTarget, content, key),
    onMouseLeave: () => hideTip(),
    onFocus: (e: React.FocusEvent<HTMLElement>) => showTip(e.currentTarget, content, key, 0),
    onBlur: () => hideTip(0),
  };
}

export function Info({ label, children, style }: { label: string; children: ReactNode; style?: CSSProperties }) {
  const key = useId();
  const open = useTip((s) => s.key === key);
  return (
    <button type="button" className="inf" aria-label={label} data-open={open} style={style} {...tipHandlers(children, key)}
      onKeyDown={(e) => e.key === "Escape" && hideTip(0)}>
      <Icon name="info" />
    </button>
  );
}

export function TooltipLayer() {
  const { rect, content, key, interactive } = useTip();
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ left: number; top: number }>({ left: -9999, top: -9999 });
  useLayoutEffect(() => {
    if (!rect || !ref.current) return;
    const tw = ref.current.offsetWidth;
    const th = ref.current.offsetHeight;
    const vw = innerWidth;
    const vh = innerHeight;
    let top = rect.top - th - 10;
    if (top < 8) top = rect.bottom + 10;
    if (top + th > vh - 8) top = Math.max(8, vh - th - 8);
    let left = rect.left + rect.width / 2 - tw / 2;
    left = Math.max(8, Math.min(vw - tw - 8, left));
    setPos({ left, top });
  }, [rect, key]);
  useEffect(() => {
    const esc = (e: KeyboardEvent) => e.key === "Escape" && hideTip(0);
    addEventListener("keydown", esc);
    return () => removeEventListener("keydown", esc);
  }, []);
  return createPortal(
    <AnimatePresence>
      {rect && (
        <motion.div
          key={key}
          ref={ref}
          role="tooltip"
          className={`tip${interactive ? " interactive" : ""}`}
          style={{ left: pos.left, top: pos.top }}
          initial={{ opacity: 0, scale: 0.97 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0 }}
          transition={MICRO}
          onMouseEnter={() => interactive && clearTimeout(hideTimer)}
          onMouseLeave={() => interactive && hideTip()}
        >
          {content}
        </motion.div>
      )}
    </AnimatePresence>,
    document.body,
  );
}

// ------------------------------------------------------------------ numbers that count

export function useTween(target: number | null, ms = 450): number | null {
  const [v, setV] = useState(target);
  const from = useRef(target);
  const reduce = useReducedMotion();
  useEffect(() => {
    if (target == null || from.current == null || reduce) {
      from.current = target;
      setV(target);
      return;
    }
    const a = from.current;
    const b = target;
    const t0 = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - t0) / ms);
      const e = 1 - Math.pow(1 - t, 3);
      const x = a + (b - a) * e;
      from.current = x;
      setV(x);
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, ms, reduce]);
  return v;
}

export function Ticker({ value, digits = 2, style, className }: { value: number | null; digits?: number; style?: CSSProperties; className?: string }) {
  const v = useTween(value);
  return <span className={`mono ${className ?? ""}`} style={style}>{v == null ? "—" : fmt(v, digits)}</span>;
}

/** +/− chip that flashes briefly whenever the value changes. */
export function DeltaChip({ value, digits = 2 }: { value: number | null; digits?: number }) {
  const prev = useRef<number | null>(null);
  const [delta, setDelta] = useState<{ d: number; k: number } | null>(null);
  useEffect(() => {
    if (value != null && prev.current != null) {
      const d = value - prev.current;
      if (Math.abs(d) >= Math.pow(10, -digits) / 2) setDelta({ d, k: performance.now() });
    }
    prev.current = value;
  }, [value, digits]);
  useEffect(() => {
    if (!delta) return;
    const t = setTimeout(() => setDelta(null), 1400);
    return () => clearTimeout(t);
  }, [delta]);
  return (
    <AnimatePresence mode="popLayout" initial={false}>
      {delta && (
        <motion.span key={delta.k} className="tag mono" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}
          transition={MICRO} style={{ height: 20, marginLeft: "auto", color: delta.d >= 0 ? "var(--acc-t)" : "var(--warn)" }}>
          {delta.d >= 0 ? "+" : "−"}{Math.abs(delta.d).toFixed(digits)}
        </motion.span>
      )}
    </AnimatePresence>
  );
}

export function fmt(v: number | null | undefined, digits = 2): string {
  if (v == null || !Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  let s: string;
  if (a !== 0 && (a < Math.pow(10, -digits) || a >= 1e6)) s = v.toExponential(1);
  else if (a >= 1000) s = Math.round(v).toLocaleString("en-US").replace(/,/g, " ");
  else s = v.toFixed(digits);
  return s.replace("-", "−");
}

/** Short value label for params: 0.0001, 1e-9, 100, 'balanced', None. */
export function fmtParam(v: unknown): string {
  if (v === null || v === undefined) return "None";
  if (typeof v === "boolean") return v ? "True" : "False";
  if (typeof v === "number") {
    if (Number.isInteger(v)) return String(v);
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(0).replace("e-", "e-").replace("+", "");
    return String(roundSig(v, 3));
  }
  if (Array.isArray(v)) return `(${v.join(", ")}${v.length === 1 ? "," : ""})`;
  return `'${String(v)}'`;
}
