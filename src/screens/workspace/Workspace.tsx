import { DockviewReact, type DockviewReadyEvent, type DockviewTheme } from "dockview-react";
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useNav } from "../../app/nav";
import { useSettings } from "../../app/settings";
import { pause, resetModel, resume, step, train, trainPhase, useSession } from "../../engine/store";
import { animateFrom, bindApi, capture, loadLayout, maximizeHovered, onLayoutChange, resetLayout, trackPointerPanel } from "./dock";
import {
  ControlsPanel, HiddenHeader, LossHeader, LossPanel, MetricHeader, MetricPanel, NetworkHeader, NetworkPanel,
  RegionsHeader, RegionsPanel, VizHeader, VizPanel,
} from "./panels";
import { AdvancedDrawer, ParamsAside } from "./Params";

const THEME: DockviewTheme = { name: "sightline", className: "dv-sightline", gap: 12 };

const components = { viz: VizPanel, controls: ControlsPanel, loss: LossPanel, metric: MetricPanel, network: NetworkPanel, regions: RegionsPanel };
const tabComponents = { vizHeader: VizHeader, lossHeader: LossHeader, metricHeader: MetricHeader, networkHeader: NetworkHeader, regionsHeader: RegionsHeader, hidden: HiddenHeader };

export function Workspace() {
  const model = useSession((s) => s.model);
  const lang = useSettings((s) => s.lang);
  const { i18n } = useTranslation();
  const [adv, setAdv] = useState(false);
  const before = useRef<ReturnType<typeof capture> | null>(null);

  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || (e.target as HTMLElement).isContentEditable || adv) return;
      if (e.key === "F11") {
        e.preventDefault();
        maximizeHovered();
        return;
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const s = useSession.getState();
      const phase = trainPhase(s);
      if (e.key === " " && (e.target as HTMLElement).tagName !== "BUTTON") {
        e.preventDefault();
        if (phase === "training") pause();
        else if (phase === "paused") resume();
        else if (phase !== "nodata" && phase !== "incompatible") train();
      } else if (e.key === ".") step();
      else if (e.key === "r" || e.key === "R") resetModel();
    };
    addEventListener("keydown", key);
    addEventListener("pointermove", trackPointerPanel, { passive: true });
    return () => {
      removeEventListener("keydown", key);
      removeEventListener("pointermove", trackPointerPanel);
    };
  }, [adv]);

  const onReady = (e: DockviewReadyEvent) => {
    bindApi(e.api, model);
    loadLayout(e.api, model ?? "LogisticRegression", i18n.language);
    e.api.onDidLayoutChange(() => onLayoutChange());
    e.api.onWillDrop(() => {
      before.current = capture();
    });
    e.api.onDidDrop(() => {
      if (before.current) animateFrom(before.current);
      before.current = null;
    });
  };

  return (
    <div className="ws" data-testid="workspace">
      <div className="ws-dock">
        <DockviewReact key={model ?? "none"} components={components} tabComponents={tabComponents} onReady={onReady} theme={THEME}
          singleTabMode="fullwidth" disableFloatingGroups={false} floatingGroupDragHandle="tabbar" />
      </div>
      <ParamsAside onAdvanced={() => setAdv(true)} onResetLayout={() => resetLayout(lang)} />
      <AdvancedDrawer open={adv} onClose={() => setAdv(false)} />
    </div>
  );
}

export function useWorkspaceTitle() {
  return useNav((s) => s.screen);
}
