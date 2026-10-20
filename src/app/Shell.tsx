import { AnimatePresence, motion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { setLang, togglePseudo } from "../i18n";
import { chapterOf } from "../handbook/chapters";
import { useSession } from "../engine/store";
import { engine } from "../engine/client";
import { Icon } from "../ui/Icon";
import { MICRO, Seg, SPRING } from "../ui/controls";
import { BY_NAME } from "./catalog";
import { useNav, type Screen } from "./nav";
import { resolvedTheme, useSettings, type Theme } from "./settings";
import { isTauri, win } from "./tauri";
import { perfStats } from "./perf";

// ------------------------------------------------------------------ theme

export function applyTheme(t: Theme, animate = true) {
  const next = resolvedTheme(t);
  const root = document.documentElement;
  if (root.dataset.theme === next) return;
  const doc = document as Document & { startViewTransition?: (cb: () => void) => unknown };
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (animate && doc.startViewTransition && !reduce) doc.startViewTransition(() => (root.dataset.theme = next));
  else root.dataset.theme = next;
  if (isTauri) void win().then((w) => w.setTheme?.(next)).catch(() => undefined);
}

export function ThemeButton() {
  const { t } = useTranslation();
  const theme = useSettings((s) => s.theme);
  const order: Theme[] = ["dark", "light", "system"];
  const next = order[(order.indexOf(theme) + 1) % 3];
  const icon = theme === "system" ? "system" : theme === "light" ? "sun" : "moon";
  return (
    <button className="btn ico" aria-label={t("nav.theme", { current: t(`theme.${theme}`), next: t(`theme.${next}`) })}
      title={t("nav.theme", { current: t(`theme.${theme}`), next: t(`theme.${next}`) })}
      onClick={() => { useSettings.getState().set({ theme: next }); applyTheme(next); }}>
      <Icon name={icon} />
    </button>
  );
}

export function LangSeg() {
  const { t } = useTranslation();
  const lang = useSettings((s) => s.lang);
  return <Seg ariaLabel={t("nav.language")} value={lang} onChange={(v) => setLang(v)} options={[{ value: "en", label: "EN" }, { value: "ru", label: "RU" }]} />;
}

// ------------------------------------------------------------------ title bar

export function TitleBar() {
  const { t } = useTranslation();
  const screen = useNav((s) => s.screen);
  const model = useSession((s) => s.model);
  const dataset = useSession((s) => s.dataset);
  const conn = useSession((s) => s.conn);
  const ready = useSession((s) => s.ready);
  const workerReady = useSession((s) => s.workerReady);
  const queued = useSession((s) => !!s.status?.queued);
  const [maxed, setMaxed] = useState(false);
  useEffect(() => {
    if (!isTauri) return;
    let off: (() => void) | undefined;
    void win().then(async (w) => {
      setMaxed(await w.isMaximized());
      off = await w.onResized(async () => setMaxed(await w.isMaximized()));
    });
    return () => off?.();
  }, []);
  const sub = screen !== "home" && model ? `${model}${dataset ? " · " + datasetLabel(t, dataset.name, dataset.preset) : ""}` : "";
  const starting = conn !== "open" || !ready || !workerReady;
  return (
    <header className="titlebar" data-tauri-drag-region>
      <div className="tb-left" data-tauri-drag-region>
        <span style={{ color: "var(--acc-t)", display: "inline-flex" }}><Icon name="logo" /></span>
        <b>Sightline</b>
        {sub && <span className="sub">{sub}</span>}
      </div>
      <div className="tb-right">
        <AnimatePresence>
          {starting && (
            <motion.span className={`engine-pill${queued ? " attn" : ""}`} role="status" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={MICRO}>
              <i />{conn !== "open" || !ready ? t("engine.starting") : t("engine.warming")}
            </motion.span>
          )}
        </AnimatePresence>
        <button className="wb" aria-label={t("win.minimize")} onClick={() => void win().then((w) => w.minimize())}><Icon name="min" /></button>
        <button className="wb" aria-label={maxed ? t("win.restore") : t("win.maximize")} onClick={() => void win().then((w) => w.toggleMaximize())}><Icon name={maxed ? "restore" : "max"} /></button>
        <button className="wb x" aria-label={t("win.close")} onClick={() => void win().then((w) => w.close())}><Icon name="close" /></button>
      </div>
    </header>
  );
}

export function datasetLabel(t: (k: string) => string, name: string, preset: string | null | undefined) {
  return preset ? t(`presets.${preset}.name`) : name;
}

// ------------------------------------------------------------------ nav

export function TopNav({ onHelp }: { onHelp: () => void }) {
  const { t } = useTranslation();
  const { screen, go, standaloneData } = useNav();
  const model = useSession((s) => s.model);
  const entry = model ? BY_NAME[model] : null;
  const home = screen === "home" || screen === "handbook" || (screen === "data" && standaloneData) || !model;
  return (
    <nav className={`topnav${home ? " home" : ""}`} aria-label={t("nav.main")}>
      {home ? (
        <div className="left">
          <Seg ariaLabel={t("nav.sections")} value={screen === "data" ? "data" : screen === "handbook" ? "handbook" : "home"}
            onChange={(v) => (v === "home" ? go("home") : v === "handbook" ? go("handbook") : go("data", { standalone: true, dataTab: "presets" }))}
            options={[{ value: "home", label: t("nav.models") }, { value: "data", label: t("nav.datasets") }, { value: "handbook", label: t("nav.handbook") }]} />
        </div>
      ) : (
        <>
          <div className="left">
            <button className="btn gh ico" aria-label={screen === "decide" ? t("nav.backToWorkspace") : t("nav.backToModels")}
              onClick={() => (screen === "decide" ? go("workspace") : go("home"))}><Icon name="back" /></button>
            <div className="mname">
              <span className="mono">{model}</span>
              <span className="lbl">{entry ? [...new Set([t(`family.${entry.family}`), t(`task.${entry.task}`)])].join(" · ") : ""}</span>
            </div>
          </div>
          <div className="center">
            <Seg ariaLabel={t("nav.views")} value={screen as Screen}
              onChange={(v) => go(v)}
              options={[
                { value: "workspace", label: t("nav.workspace") },
                { value: "data", label: t("nav.data") },
                { value: "insights", label: t("nav.insights") },
                { value: "decide", label: t("nav.decide") },
              ]} />
          </div>
        </>
      )}
      <div className="right">
        {!home && model && (
          <button className="btn ico" aria-label={t("nav.openChapter", { model })} title={t("nav.openChapter", { model })}
            onClick={() => go("handbook", { chapter: chapterOf(model) })} data-testid="open-chapter"><Icon name="book" /></button>
        )}
        <LangSeg />
        <ThemeButton />
        <button className="btn ico" aria-label={t("nav.help")} onClick={onHelp}><Icon name="help" /></button>
      </div>
    </nav>
  );
}

// ------------------------------------------------------------------ help

export function Help({ open, onClose, onGuided }: { open: boolean; onClose: () => void; onGuided: () => void }) {
  const { t } = useTranslation();
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (open) ref.current?.focus();
  }, [open]);
  const sk = useSession((s) => s.sklearn);
  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div className="modal-dim" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.25 }} onClick={onClose} />
          <motion.div ref={ref} tabIndex={-1} role="dialog" aria-modal="true" aria-label={t("help.title")} className="help"
            initial={{ opacity: 0, y: 12, x: "-50%" }} animate={{ opacity: 1, y: 0, x: "-50%" }} exit={{ opacity: 0, y: 8, x: "-50%" }} transition={SPRING}
            onKeyDown={(e) => e.key === "Escape" && onClose()}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <h2 style={{ fontSize: 18, fontWeight: 600 }}>{t("help.title")}</h2>
              <button className="btn ico gh" aria-label={t("common.close")} onClick={onClose}><Icon name="close" /></button>
            </div>
            <p style={{ margin: 0, fontSize: 14, color: "var(--ink2)", lineHeight: 1.5 }}>{t("help.intro")}</p>
            <div className="keys">
              <kbd>{t("keys.space")}</kbd><span>{t("help.kSpace")}</span>
              <kbd>.</kbd><span>{t("help.kStep")}</span>
              <kbd>R</kbd><span>{t("help.kReset")}</span>
              <kbd>F11</kbd><span>{t("help.kMax")}</span>
              <kbd>← →</kbd><span>{t("help.kSteps")}</span>
              <kbd>{t("keys.altClick")}</kbd><span>{t("help.kDelete")}</span>
              <kbd>Ctrl + Shift + P</kbd><span>{t("help.kPerf")}</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10, paddingTop: 8, borderTop: "1px solid var(--line)" }}>
              <span className="hint mono">scikit-learn {sk || "…"}</span>
              <button className="btn pri sm" onClick={() => { onClose(); onGuided(); }}>{t("help.guided")}</button>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

// ------------------------------------------------------------------ toasts

export function Toasts() {
  const toasts = useSession((s) => s.toasts);
  const { i18n } = useTranslation();
  const ru = i18n.language.startsWith("ru");
  return (
    <div className="toasts" aria-live="polite">
      <AnimatePresence>
        {toasts.map((x) => (
          <motion.div key={x.id} className="toast" role="alert" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 6 }} transition={SPRING}>
            <Icon name="warn" />{ru ? x.message.ru : x.message.en}
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}

// ------------------------------------------------------------------ perf overlay (Ctrl+Shift+P)


export function PerfOverlay() {
  const { t } = useTranslation();
  const on = useNav((s) => s.perfOverlay);
  const [, force] = useState(0);
  const fpsHist = useRef<number[]>([]);
  useEffect(() => {
    if (!on) return;
    let raf = 0;
    let last = performance.now();
    let frames = 0;
    let acc = 0;
    const loop = (now: number) => {
      frames++;
      acc += now - last;
      last = now;
      if (acc >= 500) {
        fpsHist.current.push((frames * 1000) / acc);
        if (fpsHist.current.length > 60) fpsHist.current.shift();
        frames = 0;
        acc = 0;
        force((x) => x + 1);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    let obs: PerformanceObserver | null = null;
    try {
      obs = new PerformanceObserver((l) => {
        for (const e of l.getEntries()) {
          perfStats.longTasks++;
          perfStats.longest = Math.max(perfStats.longest, e.duration);
        }
      });
      obs.observe({ type: "longtask", buffered: false });
    } catch { /* not supported */ }
    return () => { cancelAnimationFrame(raf); obs?.disconnect(); };
  }, [on]);
  const s = useSession.getState();
  const fps = fpsHist.current.at(-1) ?? 0;
  const mem = (performance as Performance & { memory?: { usedJSHeapSize: number } }).memory?.usedJSHeapSize;
  if (!on) return null;
  const h = fpsHist.current;
  const pts = h.map((v, i) => `${(i / 59) * 200},${28 - Math.min(28, (v / 130) * 28)}`).join(" ");
  return (
    <div className="perf" role="status" aria-label={t("perf.label")} data-testid="perf">
      <div className="row"><span>{t("perf.fps")}</span><b data-testid="perf-fps">{fps.toFixed(0)}</b></div>
      <div className="row"><span>{t("perf.render")}</span><b>{perfStats.renderMs.toFixed(2)} ms</b></div>
      <div className="row"><span>{t("perf.engineFps")}</span><b>{engine.framesPerSec}</b></div>
      <div className="row"><span>{t("perf.grid")}</span><b>{s.latest?.viz.grid ? `${s.latest.viz.grid.w}×${s.latest.viz.grid.h} · ${s.latest.viz_ms.toFixed(1)} ms` : "—"}</b></div>
      <div className="row"><span>{t("perf.paramFrame")}</span><b>{s.paramLatency != null ? `${s.paramLatency.toFixed(0)} ms` : "—"}</b></div>
      <div className="row"><span>{t("perf.longTasks")}</span><b>{perfStats.longTasks} · {t("perf.max")} {perfStats.longest.toFixed(0)} ms</b></div>
      <div className="row"><span>{t("perf.points")}</span><b>{s.dataset?.n_rows ?? 0}</b></div>
      {mem != null && <div className="row"><span>{t("perf.heap")}</span><b>{(mem / 1048576).toFixed(0)} MB</b></div>}
      <svg className="spark" viewBox="0 0 200 28" preserveAspectRatio="none"><polyline points={pts} fill="none" stroke="var(--acc-t)" strokeWidth="1.5" vectorEffect="non-scaling-stroke" /></svg>
    </div>
  );
}

export function usePerfHotkey() {
  useEffect(() => {
    const f = (e: KeyboardEvent) => {
      if (import.meta.env.DEV && e.ctrlKey && e.shiftKey && (e.key === "L" || e.key === "l")) {
        e.preventDefault();
        togglePseudo();
        return;
      }
      if (e.ctrlKey && e.shiftKey && (e.key === "P" || e.key === "p")) {
        e.preventDefault();
        useNav.setState((s) => ({ perfOverlay: !s.perfOverlay }));
      }
    };
    addEventListener("keydown", f);
    return () => removeEventListener("keydown", f);
  }, []);
}
