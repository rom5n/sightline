import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { lazy, Suspense, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useSession } from "../engine/store";
import { TooltipLayer } from "../ui/controls";
import { Home, startGuidedRun } from "../screens/Home";
import { Workspace } from "../screens/workspace/Workspace";
import { DataScreen } from "../screens/data/DataScreen";
import { Insights } from "../screens/insights/Insights";
import { Decide } from "../screens/decide/Decide";
// KaTeX and the chapter texts load only when the Handbook opens
const Handbook = lazy(() => import("../screens/handbook/Handbook").then((m) => ({ default: m.Handbook })));
import { Guide } from "../screens/Guide";
import { useNav } from "./nav";
import { useSettings } from "./settings";
import { applyTheme, Help, PerfOverlay, Toasts, TitleBar, TopNav, usePerfHotkey } from "./Shell";
import { BY_NAME } from "./catalog";
import * as P from "../screens/workspace/panels";

export function App() {
  const screen = useNav((s) => s.screen);
  const model = useSession((s) => s.model);
  const reduce = useReducedMotion();
  const [help, setHelp] = useState(false);
  const theme = useSettings((s) => s.theme);
  usePerfHotkey();
  useRecentTracker();
  useEffect(() => {
    applyTheme(theme, false);
    if (theme !== "system") return;
    const mq = matchMedia("(prefers-color-scheme: light)");
    const f = () => applyTheme("system");
    mq.addEventListener("change", f);
    return () => mq.removeEventListener("change", f);
  }, [theme]);
  const key = screen === "home" ? "home" : `${screen}`;
  return (
    <div className="app app-bg">
      <TitleBar />
      <TopNav onHelp={() => setHelp(true)} />
      <div className="screen">
        <AnimatePresence mode="wait" initial={false}>
          <motion.div key={key} style={{ height: "100%" }}
            initial={reduce ? { opacity: 0 } : { opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            transition={{ duration: 0.18, ease: [0.2, 0.8, 0.2, 1] }}>
            {screen === "home" && <Home />}
            {screen === "workspace" && model && <Workspace />}
            {screen === "data" && <DataScreen />}
            {screen === "insights" && <Insights />}
            {screen === "decide" && model && <Decide />}
            {screen === "handbook" && <Suspense fallback={null}><Handbook /></Suspense>}
          </motion.div>
        </AnimatePresence>
      </div>
      <TooltipLayer />
      <Toasts />
      <PerfOverlay />
      <Guide />
      <Help open={help} onClose={() => setHelp(false)} onGuided={() => void startGuidedRun()} />
    </div>
  );
}

/** Records "Recently used" when a training run finishes. */
function useRecentTracker() {
  useEffect(() => useSession.subscribe((s, p) => {
    if (s.done && s.done !== p.done && s.model && s.dataset) {
      const m = BY_NAME[s.model];
      const f = s.latest;
      const metric = f ? (m?.task === "clustering" || m?.task === "reduction" ? f.metric_train : f.metric_test ?? f.metric_train) : null;
      useSettings.getState().pushRecent({
        model: s.model, dataset: s.dataset.name, rows: s.dataset.n_rows, metric: metric ?? null, metricName: m?.metric ?? "accuracy",
        at: Date.now(), preset: s.dataset.preset ?? null, settings: s.dataset.settings ?? {},
      });
    }
  }), []);
}

/** A single panel in its own window (pop-out). */
export function PopoutApp({ panel }: { panel: string }) {
  const { t } = useTranslation();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const map: Record<string, { body: React.ComponentType<any>; head?: React.ComponentType<any> }> = {
    viz: { body: P.VizPanel as never, head: P.VizHeader as never },
    loss: { body: P.LossPanel as never, head: P.LossHeader as never },
    metric: { body: P.MetricPanel as never, head: P.MetricHeader as never },
    network: { body: P.NetworkPanel as never, head: P.NetworkHeader as never },
    regions: { body: P.RegionsPanel as never, head: P.RegionsHeader as never },
    controls: { body: P.ControlsPanel as never },
  };
  const Comp = map[panel] ?? null;
  const fakeApi = { api: { id: panel } } as Record<string, unknown>;
  return (
    <div className="app app-bg no-nav">
      <TitleBar />
      <div className="screen" style={{ padding: "0 12px 12px" }}>
        <div className={`solid popout ${panel}`} style={{ height: "100%", display: "flex", flexDirection: "column", overflow: "hidden", borderRadius: 16 }}>
          {Comp?.head && <Comp.head {...fakeApi} />}
          <div style={{ flex: 1, minHeight: 0, position: "relative", display: "flex", flexDirection: "column" }}>{Comp ? <Comp.body {...fakeApi} /> : <span className="hint">{t("common.loading")}</span>}</div>
        </div>
      </div>
      <TooltipLayer />
    </div>
  );
}
