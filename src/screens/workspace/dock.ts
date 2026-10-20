// Layout management around dockview: default layouts that match the design, per-model
// persistence, maximize (F11), pop-out windows, and a spring "snap" (FLIP on transform).
import type { DockviewApi } from "dockview-react";
import { animate } from "motion";
import { useSettings } from "../../app/settings";
import { openPopout, isTauri } from "../../app/tauri";

let api: DockviewApi | null = null;
let model: string | null = null;
const maxSubs = new Map<string, Set<(on: boolean) => void>>();
const popped = new Set<string>();

export const PANEL_TITLES: Record<string, string> = { viz: "viz", controls: "controls", loss: "loss", metric: "metric", network: "network", regions: "regions" };

function reduce() {
  return matchMedia("(prefers-reduced-motion: reduce)").matches;
}

type Rects = Map<HTMLElement, DOMRect>;

export function capture(): Rects {
  const r: Rects = new Map();
  if (api) for (const g of api.groups) r.set(g.element, g.element.getBoundingClientRect());
  return r;
}

/** FLIP: animate group transforms from their old rects to identity with a spring. */
export function animateFrom(before: Rects) {
  if (!api || reduce()) return;
  requestAnimationFrame(() => {
    if (!api) return;
    for (const g of api.groups) {
      const el = g.element;
      const a = before.get(el);
      if (!a) continue;
      const b = el.getBoundingClientRect();
      if (!b.width || !b.height) continue;
      const dx = a.left - b.left, dy = a.top - b.top, sx = a.width / b.width, sy = a.height / b.height;
      if (Math.abs(dx) + Math.abs(dy) < 1 && Math.abs(sx - 1) + Math.abs(sy - 1) < 0.01) continue;
      el.style.transformOrigin = "0 0";
      void animate(el, { transform: [`translate(${dx}px, ${dy}px) scale(${sx}, ${sy})`, "translate(0px, 0px) scale(1, 1)"] },
        { type: "spring", stiffness: 300, damping: 30 }).then(() => { el.style.transform = ""; });
    }
  });
}

/** Run a layout change with a spring snap. */
export function snap(mutate: () => void) {
  const before = capture();
  mutate();
  animateFrom(before);
}

export function bindApi(a: DockviewApi | null, m: string | null) {
  api = a;
  model = m;
}

function markGroups() {
  if (!api) return;
  for (const g of api.groups) {
    const id = g.activePanel?.id ?? "";
    g.element.dataset.kind = id;
    if (id === "controls") {
      g.header.hidden = true;
      g.locked = true;
    }
  }
  for (const [id, subs] of maxSubs) {
    const p = api.getPanel(id);
    const on = !!p && p.api.isMaximized();
    subs.forEach((f) => f(on));
  }
}

export function buildDefault(a: DockviewApi, m: string, lang: string) {
  a.clear();
  a.addPanel({ id: "viz", component: "viz", tabComponent: "vizHeader", title: "viz" });
  a.addPanel({ id: "controls", component: "controls", tabComponent: "hidden", title: "controls", position: { referencePanel: "viz", direction: "below" } });
  a.addPanel({ id: "loss", component: "loss", tabComponent: "lossHeader", title: "loss", position: { referencePanel: "controls", direction: "right" } });
  a.addPanel({ id: "metric", component: "metric", tabComponent: "metricHeader", title: "metric", position: { referencePanel: "loss", direction: "right" } });
  const ctl = a.getPanel("controls")!;
  ctl.group.api.setSize({ height: 200, width: lang === "ru" ? 290 : 270 });
  a.getPanel("metric")!.group.api.setSize({ width: 240 });
  if (m === "MLPClassifier") {
    a.addPanel({ id: "network", component: "network", tabComponent: "networkHeader", title: "network", position: { referencePanel: "viz", direction: "right" } });
    const w = a.getPanel("viz")!.group.element.getBoundingClientRect().width;
    a.getPanel("network")!.group.api.setSize({ width: Math.round(w / 2) });
  }
  if (m === "DecisionTreeClassifier") {
    const viz = a.getPanel("viz")!;
    const r = viz.group.element.getBoundingClientRect();
    a.addPanel({ id: "regions", component: "regions", tabComponent: "regionsHeader", title: "regions", floating: { width: 236, height: 206, x: Math.max(0, r.width - 252), y: 52 } });
  }
  markGroups();
}

export function loadLayout(a: DockviewApi, m: string, lang: string) {
  const saved = useSettings.getState().layouts[m];
  try {
    if (saved) {
      a.fromJSON(saved as never);
      const need = ["viz", "controls", "loss", "metric", ...(m === "MLPClassifier" ? ["network"] : []), ...(m === "DecisionTreeClassifier" ? ["regions"] : [])];
      if (need.every((id) => a.getPanel(id))) {
        markGroups();
        return;
      }
    }
  } catch {
    /* fall through to default */
  }
  buildDefault(a, m, lang);
}

let saveTimer: number | undefined;
export function onLayoutChange() {
  markGroups();
  if (!api || !model) return;
  clearTimeout(saveTimer);
  const m = model;
  const a = api;
  saveTimer = setTimeout(() => {
    if (popped.size) return; // don't persist a layout with a panel temporarily popped out
    const s = useSettings.getState();
    s.set({ layouts: { ...s.layouts, [m]: a.toJSON() } });
  }, 400) as unknown as number;
}

export function resetLayout(lang: string) {
  if (!api || !model) return;
  const s = useSettings.getState();
  const layouts = { ...s.layouts };
  delete layouts[model];
  s.set({ layouts });
  const a = api;
  const m = model;
  snap(() => buildDefault(a, m, lang));
}

export const panelActions = {
  onMax(id: string, f: (on: boolean) => void) {
    let s = maxSubs.get(id);
    if (!s) maxSubs.set(id, (s = new Set()));
    s.add(f);
    return () => {
      s!.delete(f);
    };
  },
  toggleMax(id: string) {
    const p = api?.getPanel(id);
    if (!p || !api) return;
    const a = api;
    snap(() => {
      if (p.api.isMaximized()) a.exitMaximizedGroup();
      else if (p.group.api.location.type === "grid") p.api.maximize();
    });
    setTimeout(markGroups, 0);
  },
  mlpView(v: "both" | "boundary" | "network") {
    if (!api) return;
    const a = api;
    snap(() => {
      if (a.hasMaximizedGroup()) a.exitMaximizedGroup();
      if (v === "boundary") a.getPanel("viz")?.api.maximize();
      if (v === "network") a.getPanel("network")?.api.maximize();
    });
    setTimeout(markGroups, 0);
  },
  async popout(id: string) {
    if (!api || !model) return;
    const p = api.getPanel(id);
    if (!p) return;
    const layout = api.toJSON();
    popped.add(id);
    const a = api;
    const m = model;
    snap(() => p.api.close());
    await openPopout(`pop-${id}`, id, `Sightline — ${id}`);
    const restore = () => {
      if (!popped.delete(id)) return;
      if (api !== a || model !== m) return;
      snap(() => {
        try {
          a.fromJSON(layout as never);
        } catch {
          buildDefault(a, m, useSettings.getState().lang);
        }
      });
      markGroups();
    };
    const bc = new BroadcastChannel("sightline");
    bc.onmessage = (e) => {
      if (e.data?.kind === "popout-closed" && e.data.panel === id) {
        bc.close();
        restore();
      }
    };
    if (isTauri) {
      const { WebviewWindow } = await import("@tauri-apps/api/webviewWindow");
      const w = await WebviewWindow.getByLabel(`pop-${id}`);
      void w?.once("tauri://destroyed", () => { bc.close(); restore(); });
    }
  },
  isPopped(id: string) {
    return popped.has(id);
  },
};

/** F11 maximizes the panel under the pointer (or the visualization). */
let pointerPanel: string | null = null;
export function trackPointerPanel(e: PointerEvent) {
  const g = (e.target as HTMLElement).closest?.(".dv-groupview") as HTMLElement | null;
  pointerPanel = g?.dataset.kind || pointerPanel;
}
export function maximizeHovered() {
  panelActions.toggleMax(pointerPanel && pointerPanel !== "controls" ? pointerPanel : "viz");
}
