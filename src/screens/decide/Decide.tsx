// "How it decides": the same Back / Next stepper for every model; the centre visual
// changes with the family. Everything shown comes from the engine's explain() data.
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useNav } from "../../app/nav";
import { useUi } from "../../app/ui";
import { explain, useSession } from "../../engine/store";
import { Icon } from "../../ui/Icon";
import { fmt } from "../../ui/controls";
import { buildSteps, type Step } from "./steps";
import { DecideVisual } from "./visuals";

const AUTOPLAY_MS = 1600;

export function Decide() {
  const { t } = useTranslation();
  const ds = useSession((s) => s.dataset);
  const latest = useSession((s) => s.latest);
  const schema = useSession((s) => (s.model ? s.models[s.model] : undefined));
  const ex = useSession((s) => s.explain);
  const pointSel = useUi((s) => s.decidePoint);
  const go = useNav((s) => s.go);
  const [step, setStep] = useState(1);
  const [auto, setAuto] = useState(false);
  const reduce = useReducedMotion();

  const defaultPoint = useMemo(() => {
    if (!ds || !ds.ids.data.length) return null;
    const i = ds.test.data.findIndex((x) => x === 1);
    return ds.ids.data[i >= 0 ? i : 0];
  }, [ds]);
  const pid = pointSel ?? defaultPoint;

  useEffect(() => {
    if (pid != null && latest) void explain(pid);
  }, [pid, latest?.job, latest?.final, ds?.version]);

  const data = ex && ex.point_id === pid ? (ex.data as Record<string, unknown>) : null;
  const family = ex?.family ?? schema?.explainFamily ?? "";
  const steps: Step[] = useMemo(() => (data ? buildSteps(family, data, t, ds?.classes ?? [], schema?.name ?? "") : []), [data, family, t, ds?.classes, schema?.name]);
  const n = steps.length;
  useEffect(() => setStep((s) => Math.min(Math.max(1, s), Math.max(1, n))), [n]);
  useEffect(() => {
    setStep(1);
  }, [pid]);
  useEffect(() => {
    if (!auto || !n) return;
    const id = setInterval(() => setStep((s) => (s >= n ? 1 : s + 1)), AUTOPLAY_MS);
    return () => clearInterval(id);
  }, [auto, n]);
  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === "INPUT") return;
      if (e.key === "ArrowRight") setStep((s) => Math.min(n, s + 1));
      if (e.key === "ArrowLeft") setStep((s) => Math.max(1, s - 1));
    };
    addEventListener("keydown", k);
    return () => removeEventListener("keydown", k);
  }, [n]);

  if (!latest) {
    return (
      <div className="dec" style={{ gridTemplateColumns: "1fr" }}>
        <section className="solid" style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 12 }} data-testid="decide-untrained">
          <Icon name="path" size={36} style={{ color: "var(--ink3)" }} />
          <span style={{ fontSize: 18, fontWeight: 600 }}>{t("decide.trainFirst")}</span>
          <span style={{ color: "var(--ink2)" }}>{t("decide.trainFirstBody")}</span>
          <button className="btn pri" onClick={() => go("workspace")}>{t("decide.toWorkspace")}</button>
        </section>
      </div>
    );
  }

  const idx = ds && pid != null ? ds.ids.data.indexOf(pid) : -1;
  const cls = ds && idx >= 0 && ds.task === "classification" ? ds.label.data[idx] : null;
  const raw = (data?.raw as Record<string, number | string> | undefined) ?? {};
  const truth = cls != null ? t("decide.trueClass", { cls: ds!.classes[cls] }) : ds?.task === "regression" && idx >= 0 ? t("decide.trueValue", { v: fmt(ds.label.data[idx]) }) : t("decide.noLabel");
  const pickAnother = () => {
    if (!ds) return;
    const n2 = ds.ids.data.length;
    let j = Math.floor(Math.random() * n2);
    if (ds.ids.data[j] === pid) j = (j + 1) % n2;
    useUi.setState({ decidePoint: ds.ids.data[j] });
  };
  return (
    <div className="dec" data-testid="decide">
      <section className="solid dec-main">
        <div className="dec-bar">
          <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap", minWidth: 0 }}>
            <span className="tag" style={{ height: 30, padding: "0 12px", fontSize: 13, color: "var(--ink)" }}>
              {cls != null ? <span className={`dot k${cls % 5} s${cls % 4}`} /> : <span className="dot" style={{ background: "transparent", boxShadow: "inset 0 0 0 2px var(--ink)" }} />}
              {t("tip.point", { id: pid })}
            </span>
            <span className="mono" style={{ fontSize: 13, color: "var(--ink2)" }} data-testid="decide-point">
              {Object.entries(raw).slice(0, 4).map(([k, v]) => `${k} = ${typeof v === "string" ? `"${v}"` : fmt(v)}`).join(" · ")}{Object.keys(raw).length > 4 ? " · …" : ""} · {truth}
            </span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span className="hint">{t("decide.clickHint")}</span>
            <button className="btn sm" onClick={pickAnother}><Icon name="crosshair" />{t("decide.another")}</button>
          </div>
        </div>
        <div style={{ flex: 1, minHeight: 0, position: "relative" }}>
          {data ? <DecideVisual family={family} data={data} step={step} n={n} onPick={(id) => useUi.setState({ decidePoint: id })} pid={pid} />
            : <span className="hint" style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>{t("decide.loading")}</span>}
        </div>
      </section>

      <aside className="glass" style={{ display: "flex", flexDirection: "column", minHeight: 0, padding: 18 }} aria-label={t("decide.title")}>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", padding: "0 4px 12px" }}>
          <h2 style={{ fontSize: 17, fontWeight: 600 }}>{t("decide.title")}</h2>
          <span className="mono hint" data-testid="step-of">{t("decide.stepOf", { s: step, n })}</span>
        </div>
        <div style={{ flex: 1, minHeight: 0, overflow: "auto", display: "flex", flexDirection: "column", gap: 4 }} role="list">
          {steps.map((s, i) => {
            const k = i + 1;
            const state = k < step ? "done" : k === step ? "cur" : "todo";
            return (
              <button key={i} className={`st ${state}`} onClick={() => setStep(k)} role="listitem" aria-current={k === step ? "step" : undefined}>
                <span className="num">{k}</span><span className="tt">{s.title}</span><span className="sh">{s.short}</span>
                <AnimatePresence initial={false}>
                  {k === step && (
                    <motion.span className="det" initial={reduce ? { opacity: 0 } : { opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.18 }}>{s.detail}</motion.span>
                  )}
                </AnimatePresence>
              </button>
            );
          })}
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 12, paddingTop: 14, borderTop: "1px solid var(--line)" }}>
          <div style={{ display: "flex", gap: 4 }} aria-hidden="true">
            {steps.map((_, i) => <div className="pg" key={i}><motion.i initial={false} animate={{ scaleX: i + 1 <= step ? 1 : 0 }} transition={{ type: "spring", stiffness: 300, damping: 30 }} /></div>)}
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn" style={{ flex: 1 }} onClick={() => setStep((s) => Math.max(1, s - 1))} disabled={step <= 1} data-testid="back"><Icon name="arrowL" />{t("decide.back")}</button>
            <button className="btn pri" style={{ flex: 1.4 }} onClick={() => setStep((s) => Math.min(n, s + 1))} disabled={step >= n} data-testid="next">{t("decide.next")}<Icon name="arrowR" /></button>
          </div>
          <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, color: "var(--ink2)" }}>
            <input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} style={{ accentColor: "var(--acc)" }} />{t("decide.autoplay")}
          </label>
        </div>
      </aside>
    </div>
  );
}

export function usePrevious<T>(v: T): T | undefined {
  const r = useRef<T>();
  useEffect(() => { r.current = v; });
  return r.current;
}
