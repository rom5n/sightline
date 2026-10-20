import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import type { IDockviewPanelHeaderProps, IDockviewPanelProps } from "dockview-react";
import { BY_NAME } from "../../app/catalog";
import { useNav } from "../../app/nav";
import { useUi } from "../../app/ui";
import { LineChart, type Series } from "../../charts/LineChart";
import { cancel, currentParams, loadPreset, pause, resetModel, resume, selectModel, setSpeed, step, train, trainPhase, useSession, type TrainPhase } from "../../engine/store";
import type { FrameMsg, LossKind, ModelSchema, TreeData } from "../../engine/types";
import { NetworkView } from "../../gl/NetworkView";
import { PlotView, type PlotExtra } from "../../gl/PlotView";
import { Icon } from "../../ui/Icon";
import { DeltaChip, fmt, Info, Seg, Ticker } from "../../ui/controls";
import { panelActions } from "./dock";
import { TreeView } from "./TreeView";

// ------------------------------------------------------------------ shared hooks

export function useModel(): { schema: ModelSchema | undefined; name: string | null } {
  const name = useSession((s) => s.model);
  const schema = useSession((s) => (s.model ? s.models[s.model] : undefined));
  return { schema, name };
}

/** Frame shown right now: the scrubbed one, or the latest. */
export function useFrame(): FrameMsg | null {
  return useSession((s) => (s.scrub != null ? s.frames[s.scrub] ?? s.latest : s.latest));
}

function PanelButtons({ id }: { id: string }) {
  const { t } = useTranslation();
  const [max, setMax] = useState(false);
  useEffect(() => panelActions.onMax(id, setMax), [id]);
  if (location.hash.includes("popout=")) return null; // already in its own window
  return (
    <span className="pbtns" onPointerDown={(e) => e.stopPropagation()} draggable={false}>
      <button className="btn gh ico sm" aria-label={max ? t("panel.restore") : t("panel.maximize")} title={`${max ? t("panel.restore") : t("panel.maximize")} (F11)`}
        onClick={() => panelActions.toggleMax(id)}><Icon name={max ? "shrink" : "expand"} size={14} /></button>
      <button className="btn gh ico sm" aria-label={t("panel.popout")} title={t("panel.popout")} onClick={() => panelActions.popout(id)}><Icon name="popout" size={14} /></button>
    </span>
  );
}

function StatusTag() {
  const { t } = useTranslation();
  const phase = useSession(trainPhase);
  const latest = useSession((s) => s.latest);
  const done = useSession((s) => s.done);
  const status = useSession((s) => s.status);
  const { schema } = useModel();
  if (!schema) return null;
  const unit = t(`unit.${schema.stepUnit}`);
  const unitNom = t(`unit.${schema.stepUnit}Nom`); // "эпоха 12" (nominative) vs "на эпохе 12"
  if (phase === "training" && latest) return <span className="tag" style={{ color: "var(--acc-t)" }}><span className="dot" style={{ width: 7, height: 7, background: "var(--acc)" }} />{t("status.training", { unitNom, n: latest.iteration })}</span>;
  if (phase === "paused" && latest) return <span className="tag" style={{ color: "var(--warn)" }}><Icon name="pauseTag" size={12} />{t("status.paused", { unit, n: latest.iteration })}</span>;
  if (phase === "fitting") return <span className="tag" style={{ color: "var(--ink2)" }}>{t("status.fitting", { p: Math.round((status?.progress ?? 0) * 100) })}</span>;
  if (phase === "starting") return <span className="tag">{t("engine.restarting")}</span>;
  if (phase === "done" && done) {
    let txt = "";
    if (schema.name === "DecisionTreeClassifier" && latest?.viz.tree) txt = t("status.treeFitted", { d: latest.viz.tree.depth, l: latest.viz.tree.leaves });
    else if (done.reason === "converged") txt = t("status.converged", { unit, n: done.iteration });
    else if (done.reason === "early_stop") txt = t("status.earlyStop", { unit, n: done.iteration });
    else if (done.reason === "max_iter") txt = t("status.maxIter", { n: done.iteration });
    else if (schema.iterative && done.iteration > 1) txt = t("status.doneAt", { n: done.iteration, unit: t(`unit.${schema.stepUnit}Plural`) });
    else txt = t("status.fitted");
    return <span className="tag" style={{ color: done.reason === "max_iter" ? "var(--warn)" : "var(--acc-t)" }}><Icon name="check" size={12} />{txt}</span>;
  }
  return null;
}

function Legend({ addable }: { addable: boolean }) {
  const { t } = useTranslation();
  const ds = useSession((s) => s.dataset);
  const add = useUi((s) => s.addClass);
  if (!ds || ds.task !== "classification" || !ds.classes.length) return null;
  return (
    <div className="legend" role={addable ? "radiogroup" : undefined} aria-label={addable ? t("viz.addClass") : t("viz.classes")}>
      {ds.classes.map((c, i) => (
        addable ? (
          <button key={c} className={add === i ? "on" : ""} role="radio" aria-checked={add === i} title={t("viz.addAs", { cls: c })} onClick={() => useUi.setState({ addClass: i })}>
            <span className={`dot k${i % 5} s${i % 4}`} />{t("viz.class", { cls: c })}
          </button>
        ) : (
          <span key={c} style={{ display: "flex", alignItems: "center", gap: 6 }}><span className={`dot k${i % 5} s${i % 4}`} />{t("viz.class", { cls: c })}</span>
        )
      ))}
    </div>
  );
}

// ------------------------------------------------------------------ viz panel

function vizTitleKey(schema: ModelSchema | undefined, mode: string | undefined): string {
  if (!schema) return "viz.boundary";
  if (schema.name === "DecisionTreeClassifier") return "viz.tree";
  if (schema.name === "MLPClassifier") return "viz.boundaryNet";
  if (schema.name === "KMeans") return "viz.kmeans";
  if (schema.name === "DBSCAN") return "viz.dbscan";
  if (schema.name === "PCA") return "viz.pca";
  if (schema.task === "regression") return mode === "xy" ? "viz.regLine" : "viz.regSurface";
  return "viz.boundary";
}

export function VizHeader(props: IDockviewPanelHeaderProps) {
  const { t } = useTranslation();
  const { schema } = useModel();
  const ds = useSession((s) => s.dataset);
  const ui = useUi();
  const go = useNav((s) => s.go);
  const latest = useSession((s) => s.latest);
  const mode = ds?.plane?.mode;
  const trained = !!latest;
  return (
    <div className="ph" data-testid="viz-header">
      <div className="ph-l">
        <h2>{t(vizTitleKey(schema, mode))}</h2>
        <Info label={t("viz.aboutLabel")}><b>{t(vizTitleKey(schema, mode))}</b><br />{t(`${vizTitleKey(schema, mode)}Tip`)}</Info>
        <StatusTag />
        {mode === "pca" && schema?.name !== "PCA" && (
          <span className="tag" title={t("viz.pcaPlaneTip")}>{t("viz.pcaPlane", { n: ds?.n_features, p: Math.round(((ds?.plane?.explained?.[0] ?? 0) + (ds?.plane?.explained?.[1] ?? 0)) * 100) })}</span>
        )}
        {schema?.task === "clustering" && schema.name === "KMeans" && <span className="hint">{t("viz.kmeansNote")}</span>}
        {schema?.name.startsWith("CatBoost") && <CatNote />}
      </div>
      <div className="ph-r" onPointerDown={(e) => e.stopPropagation()}>
        {schema?.task === "classification" && <Legend addable={!!ds?.plane?.editable} />}
        {schema?.task === "classification" && schema.name !== "DecisionTreeClassifier" && schema.name !== "MLPClassifier" && (
          <Seg size="sm" ariaLabel={t("viz.layers")} value={ui.heatLayer} onChange={(v) => useUi.setState({ heatLayer: v })}
            options={[{ value: "proba", label: t("viz.probability") }, { value: "border", label: t("viz.borderOnly") }]} />
        )}
        {schema?.name === "DecisionTreeClassifier" && (
          <Seg size="sm" ariaLabel={t("viz.view")} value={ui.treeView} onChange={(v) => useUi.setState({ treeView: v })}
            options={[{ value: "tree", label: t("viz.treeTab") }, { value: "regions", label: t("viz.regionsTab") }]} />
        )}
        {schema?.name === "MLPClassifier" && (
          <Seg size="sm" ariaLabel={t("viz.layout")} value={ui.mlpView} onChange={(v) => { useUi.setState({ mlpView: v }); panelActions.mlpView(v); }}
            options={[{ value: "both", label: t("viz.both") }, { value: "boundary", label: t("viz.boundaryTab") }, { value: "network", label: t("viz.networkTab") }]} />
        )}
        {schema?.name === "KMeans" && (
          <Seg size="sm" multi ariaLabel={t("viz.layers")} values={[...ui.kmLayers, ...(ui.pointsOnly ? ["points" as const] : [])] as never}
            onChange={(v) => {
              if (v === ("points" as never)) useUi.setState({ pointsOnly: !ui.pointsOnly });
              else useUi.setState({ pointsOnly: false, kmLayers: ui.kmLayers.includes(v as never) ? ui.kmLayers.filter((x) => x !== v) : [...ui.kmLayers, v as never] });
            }}
            options={[{ value: "regions" as never, label: t("viz.regions") }, { value: "paths" as never, label: t("viz.paths") }, { value: "points" as never, label: t("viz.pointsOnly") }]} />
        )}
        <button className="btn sm" onClick={() => go("decide")} disabled={!trained} title={trained ? undefined : t("viz.trainFirst")}><Icon name="path" />{t("nav.decide")}</button>
        <PanelButtons id={props.api.id} />
      </div>
    </div>
  );
}

/** CatBoost's distinctive feature, said where it matters: which columns it reads as categories. */
function CatNote() {
  const { t } = useTranslation();
  const ds = useSession((s) => s.dataset);
  const target = ds?.table.target;
  const cats = (ds?.table.columns ?? []).filter((c) => c.type === "category" && c.name !== target).map((c) => c.name);
  return cats.length ? (
    <span className="tag" title={t("cb.catTip", { cols: cats.join(", ") })} data-testid="cat-note">
      <Icon name="check" size={12} style={{ color: "var(--acc-t)" }} />{t("cb.catTag", { cols: cats.join(", ") })}
    </span>
  ) : (
    <span className="hint opt" title={t("cb.noCatsTip")} data-testid="cat-note">{t("cb.noCats")}</span>
  );
}

export function VizPanel(_p: IDockviewPanelProps) {
  const { t } = useTranslation();
  const { schema } = useModel();
  const phase = useSession(trainPhase);
  const ds = useSession((s) => s.dataset);
  const ui = useUi();
  const latest = useFrame();
  const treeNode = useUi((s) => s.treeNode);
  if (phase === "nodata") return <NoDataState />;
  if (phase === "incompatible") return <IncompatibleState />;
  const isTree = schema?.name === "DecisionTreeClassifier";
  const tree = latest?.viz.tree;
  return (
    <div className="pbody" data-guide="viz">
      {isTree && ui.treeView === "tree" ? (
        tree ? <TreeView tree={tree} classes={ds?.classes ?? []} selected={treeNode} onSelect={(id) => useUi.setState({ treeNode: id })} criterion={String(currentParams().criterion ?? "gini")} testId="tree-view" />
          : <PlotView variant="main" editable tooltip ghost testId="plot-main" />
      ) : (
        <PlotView variant="main" editable tooltip ghost testId="plot-main" extra={isTree ? () => regionLines(tree, treeNode) : undefined}>
          {phase === "idle" && !latest && <div className="cta-chip"><Icon name="play" style={{ color: "var(--acc-t)" }} /><span dangerouslySetInnerHTML={{ __html: t("states.pressTrain") }} /></div>}
          {ds?.plane?.editable && latest && <span className="edit-hint">{t("viz.editHint")}</span>}
          <ShadeLegend />
        </PlotView>
      )}
    </div>
  );
}

function ShadeLegend() {
  const { t } = useTranslation();
  const { schema } = useModel();
  const latest = useSession((s) => s.latest);
  const ds = useSession((s) => s.dataset);
  const layer = useUi((s) => s.heatLayer);
  if (!latest || !schema) return null;
  const v = latest.viz;
  if (v.kind === "proba" && ds) {
    const k = ds.classes.length;
    const grad = k === 2 ? "linear-gradient(90deg, var(--c0), transparent 50%, var(--c1))" : `linear-gradient(90deg, ${ds.classes.map((_, i) => `var(--c${i % 5})`).join(", ")})`;
    return (
      <div className="shade">
        {layer === "proba" && <span className="lbl">{v.score ? t("viz.shadeScore") : t("viz.shade")}</span>}
        {layer === "proba" && <div style={{ display: "flex", alignItems: "center", gap: 8 }}><span className="mono hint">{ds.classes[0]}</span><div className="bar" style={{ background: grad }} /><span className="mono hint">{ds.classes[k - 1]}</span></div>}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}><span className="ln-key" /><span className="hint">{t("viz.borderNow")}</span><span className="ln-key dsh" /><span className="hint">{t("viz.tenAgo")}</span></div>
      </div>
    );
  }
  if (v.kind === "clusters") {
    return (
      <div className="shade" style={{ gap: 8 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}><div className="ctr" style={{ position: "static", width: 20, height: 20, fontSize: 12 }}>1</div><span className="hint">{t("viz.centerKey")}</span></div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}><span className="ln-key dsh" /><span className="hint">{t("viz.pathKey")}</span></div>
      </div>
    );
  }
  if (v.kind === "dbscan") {
    return <div className="shade"><div style={{ display: "flex", alignItems: "center", gap: 8 }}><span style={{ width: 9, height: 9, borderRadius: "50%", boxShadow: "inset 0 0 0 1.5px var(--ink3)" }} /><span className="hint">{t("viz.noiseKey")}</span></div></div>;
  }
  if (v.kind === "curve") return <div className="shade"><div style={{ display: "flex", alignItems: "center", gap: 8 }}><span className="ln-key" style={{ borderColor: "var(--acc-t)" }} /><span className="hint">{t("viz.lineKey")}</span></div></div>;
  if (v.kind === "value") return <div className="shade"><span className="lbl">{t("viz.valueKey")}</span><div style={{ display: "flex", alignItems: "center", gap: 8 }}><span className="mono hint">{fmt(v.range?.[0], 1)}</span><div className="bar" style={{ background: "linear-gradient(90deg, var(--c0), var(--c1))", opacity: 0.6 }} /><span className="mono hint">{fmt(v.range?.[1], 1)}</span></div></div>;
  if (v.kind === "projection") return <div className="shade"><span className="hint">{t("viz.pcaKey")}</span></div>;
  return null;
}

/** Axis-aligned region of a tree node, as rectangle edges, when the plane is the raw feature plane. */
export function regionLines(tree: TreeData | undefined, nodeId: number | null): PlotExtra[] {
  const ds = useSession.getState().dataset;
  if (!tree || nodeId == null || !ds?.plane || ds.plane.mode !== "raw") return [];
  const b = ds.plane.bounds;
  let [x0, x1, y0, y1] = [b[0] - 1e3, b[1] + 1e3, b[2] - 1e3, b[3] + 1e3];
  const parent = new Map<number, { id: number; left: boolean }>();
  for (const n of tree.nodes) {
    if (n.left != null) parent.set(n.left, { id: n.id, left: true });
    if (n.right != null) parent.set(n.right, { id: n.id, left: false });
  }
  let cur = nodeId;
  const byId = new Map(tree.nodes.map((n) => [n.id, n]));
  while (parent.has(cur)) {
    const p = parent.get(cur)!;
    const n = byId.get(p.id)!;
    const thr = n.threshold ?? 0;
    if (n.feature === 0) { if (p.left) x1 = Math.min(x1, thr); else x0 = Math.max(x0, thr); }
    else if (p.left) y1 = Math.min(y1, thr); else y0 = Math.max(y0, thr);
    cur = p.id;
  }
  const acc = getComputedStyle(document.documentElement).getPropertyValue("--acc-t").trim();
  const hex = acc.startsWith("#") ? acc : "#C4EC6A";
  const c: [number, number, number, number] = [parseInt(hex.slice(1, 3), 16) / 255, parseInt(hex.slice(3, 5), 16) / 255, parseInt(hex.slice(5, 7), 16) / 255, 1];
  return [{ name: "region", segs: new Float32Array([x0, y0, x1, y0, x1, y0, x1, y1, x1, y1, x0, y1, x0, y1, x0, y0]), style: { color: c, width: 2.5 } }];
}

function NoDataState() {
  const { t } = useTranslation();
  const go = useNav((s) => s.go);
  const { name } = useModel();
  const suggestion = name && BY_NAME[name]?.task === "regression" ? "linear" : name && BY_NAME[name]?.task === "clustering" ? "blobs" : "moons";
  return (
    <div className="pbody">
      <div className="vstate dashed" data-testid="no-data">
        <div className="inner">
          <Icon name="table" size={40} style={{ color: "var(--ink3)", strokeWidth: 1.3 }} />
          <span style={{ fontSize: 18, fontWeight: 600 }}>{t("states.noData")}</span>
          <span style={{ fontSize: 14, color: "var(--ink2)", lineHeight: 1.5 }}>{t("states.noDataBody")}</span>
          <button className="btn pri" style={{ marginTop: 4 }} onClick={() => go("data", { dataTab: "presets" })}>{t("states.choosePreset")}</button>
          <div style={{ display: "flex", gap: 6 }}>
            <button className="btn sm" onClick={() => go("data", { dataTab: "generator" })}>{t("states.generate")}</button>
            <button className="btn sm" onClick={() => go("data", { dataTab: "table" })}>{t("states.buildTable")}</button>
          </div>
          <button className="linkbtn" onClick={() => void loadPreset(suggestion)}>{t("states.quickPreset", { preset: t(`presets.${suggestion}.name`) })}</button>
        </div>
      </div>
    </div>
  );
}

function IncompatibleState() {
  const { t } = useTranslation();
  const ds = useSession((s) => s.dataset)!;
  const { name } = useModel();
  const go = useNav((s) => s.go);
  const c = ds.compat;
  const p = c.params as Record<string, unknown>;
  const values = Array.isArray(p.values) ? (p.values as string[]).join(", ") : "";
  return (
    <div className="pbody">
      <div className="incompat" data-testid="incompatible">
        <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
          <Icon name="warn" size={22} style={{ color: "var(--warn)", marginTop: 2 }} />
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <span style={{ fontSize: 16, fontWeight: 600, lineHeight: 1.35 }} dangerouslySetInnerHTML={{ __html: t(`compat.${c.code}.title`, { model: name, ...p, values }) }} />
            <span style={{ fontSize: 13, color: "var(--ink2)", lineHeight: 1.5 }} dangerouslySetInnerHTML={{ __html: t(`compat.${c.code}.body`, { model: name, ...p, values }) }} />
          </div>
        </div>
        <div style={{ marginTop: 12, display: "flex", flexDirection: "column" }}>
          <span className="lbl" style={{ paddingBottom: 4 }}>{t("compat.fixes", { count: c.fixes.length })}</span>
          {c.fixes.map((f, i) => (
            <div className="fix" key={i}>
              <span className="n">{i + 1}</span>
              <span style={{ fontSize: 13 }} dangerouslySetInnerHTML={{ __html: t(`fix.${f.kind}.text`, { model: f.model, column: f.column ? t("fix.colHint", { c: f.column }) : "", preset: f.preset ? t(`presets.${f.preset}.name`) : "" }) }} />
              <span />
              <button className={`btn sm${i === 0 ? " pri" : ""}`} style={{ justifySelf: "start" }} onClick={() => {
                if (f.kind === "switch_model" && f.model) selectModel(f.model);
                else if (f.kind === "preset" && f.preset) void loadPreset(f.preset);
                else go("data", { dataTab: "table" });
              }}>{t(`fix.${f.kind}.btn`)}</button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ controls

const SPEEDS = [0.5, 1, 2, 4];

export function ControlsPanel(_p: IDockviewPanelProps) {
  const { t } = useTranslation();
  const { schema } = useModel();
  const phase = useSession(trainPhase);
  const latest = useSession((s) => s.latest);
  const frames = useSession((s) => s.frames);
  const scrub = useSession((s) => s.scrub);
  const speed = useSession((s) => s.speed);
  const status = useSession((s) => s.status);
  const done = useSession((s) => s.done);
  const reduce = useReducedMotion();
  if (!schema) return null;
  const iterative = schema.iterative;
  const canTrain = phase !== "nodata" && phase !== "incompatible";
  const unit = t(`unit.${schema.stepUnit}`);
  const isTree = schema.name === "DecisionTreeClassifier";
  const onTrain = () => {
    if (phase === "training") pause();
    else if (phase === "paused") resume();
    else train();
  };
  let label = t("ctl.train");
  if (phase === "training") label = t("ctl.training");
  else if (phase === "paused") label = t("ctl.continue");
  else if (phase === "done") label = t("ctl.again");
  else if (phase === "fitting") label = t("ctl.fitting");
  const total = latest?.total ?? 0;
  const it = latest?.iteration ?? 0;
  const frac = total ? it / total : 0;
  const stoppedEarly = done && (done.reason === "converged" || done.reason === "early_stop") && it < total;
  const showScrub = phase === "done" && frames.length > 1;
  return (
    <div className="ctl" data-testid="controls">
      <button className="btn pri train-btn" disabled={!canTrain || phase === "starting"} aria-pressed={phase === "training"} onClick={onTrain} data-testid="train-btn"
        title={!canTrain ? (phase === "nodata" ? t("ctl.addData") : t("ctl.fixData")) : undefined}>
        <AnimatePresence mode="wait" initial={false}>
          <motion.span key={label} style={{ display: "inline-flex", alignItems: "center", gap: 8, minWidth: 0, maxWidth: "100%" }} initial={reduce ? { opacity: 0 } : { opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={reduce ? { opacity: 0 } : { opacity: 0, y: -6 }} transition={{ duration: 0.15 }}>
            {phase === "training" ? <span className="live" /> : <Icon name="play" fill />}<span className="bl" style={{ minWidth: 0, overflow: "hidden", textOverflow: "ellipsis" }}>{label}</span>
          </motion.span>
        </AnimatePresence>
        {phase === "training" && !reduce && <span className="pulse" />}
      </button>
      {phase === "fitting" ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <div className="prow"><span className="lbl">{t("ctl.fitProgress")}</span><span className="mono" style={{ fontSize: 12, color: "var(--ink2)" }}>{Math.round((status?.progress ?? 0) * 100)}%</span></div>
          <div className="prog" role="progressbar" aria-valuenow={Math.round((status?.progress ?? 0) * 100)} aria-valuemin={0} aria-valuemax={100}>
            <i style={{ transform: `scaleX(${status?.progress ?? 0})`, transition: "transform .15s" }} />
          </div>
          <button className="btn sm" onClick={cancel} data-testid="cancel-btn"><Icon name="close" />{t("ctl.cancel")}</button>
        </div>
      ) : (
        <>
          <div className="grid3">
            <button className={`btn sm${phase === "training" ? " act" : ""}`} disabled={!iterative || isTree || phase !== "training"} onClick={pause}
              title={!iterative || isTree ? t(isTree ? "ctl.noPauseTree" : "ctl.noPause") : t("ctl.pause")}><Icon name="pause" /><span className="bl">{t("ctl.pause")}</span></button>
            <button className="btn sm" disabled={!iterative || !canTrain || phase === "training"} onClick={step} data-testid="step-btn"
              title={!iterative ? t("ctl.noStep") : phase === "training" ? t("ctl.pauseFirst", { unit }) : isTree ? t("ctl.stepTree") : t("ctl.stepTitle", { unit })}>
              <Icon name="step" /><span className="bl">{isTree ? t("ctl.plusLevel") : t("ctl.step")}</span>
            </button>
            <button className="btn sm" disabled={!latest && phase !== "training"} onClick={resetModel} data-testid="reset-btn" title={t("ctl.reset")}><Icon name="reset" /><span className="bl">{t("ctl.reset")}</span></button>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 2 }}>
            <div className="prow">
              <span className="lbl">{showScrub ? t("ctl.replay") : t(isTree ? "ctl.growth" : `ctl.unit.${schema.stepUnit}`)}</span>
              <span className="mono" style={{ fontSize: 12, color: "var(--ink2)" }} data-testid="iter">
                {showScrub && scrub != null ? `${frames[scrub]?.iteration ?? 0} / ${total}` : !latest ? "—" : isTree ? t("ctl.depthOf", { d: it, n: total }) : stoppedEarly ? t("ctl.stoppedEarly", { n: it }) : !iterative ? t("ctl.oneStep") : `${it} / ${total}`}
              </span>
            </div>
            {showScrub ? (
              <input className="rng scrub" type="range" min={0} max={frames.length - 1} value={scrub ?? frames.length - 1} aria-label={t("ctl.scrub")}
                style={{ ["--p" as string]: `${((scrub ?? frames.length - 1) / (frames.length - 1)) * 100}%` }}
                onChange={(e) => { const v = Number(e.target.value); useSession.setState({ scrub: v >= frames.length - 1 ? null : v }); }} />
            ) : (
              <div className="prog"><i style={{ transform: `scaleX(${stoppedEarly || (!iterative && latest) ? 1 : frac})`, transition: "transform .2s" }} /></div>
            )}
          </div>
          <div className="prow">
            <span className="lbl">{isTree ? t("ctl.replaySpeed") : t("ctl.speed")}</span>
            <Seg size="sm" ariaLabel={t("ctl.speed")} value={speed} onChange={setSpeed}
              options={SPEEDS.map((s) => ({ value: s, label: s === 0.5 ? "½×" : `${s}×`, disabled: !iterative }))} />
          </div>
        </>
      )}
      <Ripple />
    </div>
  );
}

/** Short accent sweep across the bottom panels when training converges. */
export function Ripple() {
  const done = useSession((s) => s.done);
  const reduce = useReducedMotion();
  const [key, setKey] = useState<number | null>(null);
  const last = useRef<number | null>(null);
  useEffect(() => {
    if (done && done.job !== last.current && (done.reason === "converged" || done.reason === "fitted" || done.reason === "early_stop" || done.reason === "max_iter")) {
      last.current = done.job;
      setKey(done.job);
      const tm = setTimeout(() => setKey(null), 900);
      return () => clearTimeout(tm);
    }
  }, [done]);
  if (key == null || reduce) return null;
  return (
    <div className="ripple" aria-hidden="true">
      <motion.i initial={{ x: "-100%", opacity: 0.9 }} animate={{ x: "260%", opacity: 0 }} transition={{ duration: 0.8, ease: "easeOut" }} />
    </div>
  );
}

// ------------------------------------------------------------------ loss panel

function lossTitle(kind: LossKind, name: string | null): { title: string; key: string } {
  switch (kind) {
    case "inertia": return { title: "loss.inertia", key: "loss.inertiaKey" };
    case "boost": return { title: "loss.boost", key: name === "AdaBoostClassifier" ? "loss.errorRate" : name === "CatBoostRegressor" ? "loss.rmse" : "loss.logloss" };
    case "trees": return { title: "loss.trees", key: "loss.treesKey" };
    case "depth": return { title: "loss.depth", key: "" };
    case "onestep": return { title: "loss.onestep", key: "" };
    case "onestep_reg": return { title: "loss.onestepReg", key: "" };
    case "knn": return { title: "loss.knn", key: "" };
    case "pca": return { title: "loss.pca", key: "" };
    default:
      return { title: "loss.loss", key: name === "Lasso" || name === "ElasticNet" ? "loss.objective" : name === "SGDClassifier" ? "loss.sgd" : name === "MLPClassifier" ? "loss.perEpoch" : "loss.logloss" };
  }
}

export function LossHeader(props: IDockviewPanelHeaderProps) {
  const { t } = useTranslation();
  const { schema, name } = useModel();
  const latest = useSession((s) => s.latest);
  if (!schema) return null;
  const { title, key } = lossTitle(schema.lossKind, name);
  const lossName = name === "SGDClassifier" ? String(currentParams().loss) : undefined;
  const live = ["loss", "inertia", "boost"].includes(schema.lossKind);
  return (
    <div className="ph sm" data-guide="loss">
      <div className="ph-l" style={{ gap: 6 }}>
        <h2>{t(title)}</h2>
        {key && <span className="pk">{t(key, { loss: lossName })}</span>}
        <Info label={t("loss.about")}>{t(`${title}Tip`)}</Info>
        {live && schema.lossKind !== "inertia" && <span className="hint opt">{t(schema.lossKind === "trees" ? "loss.higherBetter" : "loss.lowerBetter")}</span>}
        {schema.lossKind === "inertia" && <span className="hint opt">{t("loss.inertiaHint")}</span>}
      </div>
      <div className="ph-r" style={{ gap: 12 }}>
        {schema.lossKind === "loss" && name === "MLPClassifier" && latest?.loss != null && <span className="mono" style={{ fontSize: 12, color: "var(--ink)" }}>{fmt(latest.loss, 3)}</span>}
        {(schema.lossKind === "loss" || schema.lossKind === "boost") && (
          <span className="hint key" style={{ display: "flex", gap: 12 }}>
            <span style={{ display: "flex", alignItems: "center", gap: 6 }}><i className="k-acc" />{t("loss.train")}</span>
            <span style={{ display: "flex", alignItems: "center", gap: 6 }}><i className="k-ink" />{t(schema.lossKind === "boost" && !name?.startsWith("CatBoost") ? "loss.validation" : "loss.test")}</span>
          </span>
        )}
        <PanelButtons id={props.api.id} />
      </div>
    </div>
  );
}

export function LossPanel(_p: IDockviewPanelProps) {
  const { t } = useTranslation();
  const { schema } = useModel();
  const phase = useSession(trainPhase);
  const h = useSession((s) => s.history);
  const latest = useSession((s) => s.latest);
  const done = useSession((s) => s.done);
  const scrub = useSession((s) => s.scrub);
  const frames = useSession((s) => s.frames);
  if (!schema) return null;
  const k = schema.lossKind;
  if (!latest) {
    return (
      <div className="lp" style={{ paddingBottom: 12 }}>
        <div className="ghost" style={{ marginTop: 6 }}>
          {phase === "nodata" || phase === "incompatible" ? (
            <span className="hint">{phase === "nodata" ? t("states.lossNoData") : t("states.lossOff")}</span>
          ) : (
            <>
              <svg width="100%" height="28" viewBox="0 0 100 28" preserveAspectRatio="none" aria-hidden="true"><path d="M2 3C20 20 40 24 98 25" style={{ fill: "none", stroke: "var(--ink3)", strokeWidth: 1.5, strokeDasharray: "3 4", vectorEffect: "non-scaling-stroke" }} /></svg>
              <span className="hint">{t(`states.lossIdle.${k}`, { defaultValue: t("states.lossIdle.loss") })}</span>
            </>
          )}
        </div>
        <Ripple />
      </div>
    );
  }
  const cursor = scrub != null ? frames[scrub]?.iteration ?? null : null;
  const total = latest.total || 1;
  let body: ReactNode = null;
  if (k === "loss") {
    const series: Series[] = [{ xs: h.it, ys: h.loss, color: "acc" }];
    if (h.lossTest.some((v) => v != null)) series.push({ xs: h.it, ys: h.lossTest, color: "ink2", dash: true });
    const last = h.loss[h.loss.length - 1];
    body = <LineChart series={series} xMax={total} glow={phase === "training" || phase === "paused"} zeroBased xLabel={t(`unit.${schema.stepUnit}Plural`)}
      point={last != null ? { x: h.it[h.it.length - 1], y: last, label: fmt(last, 3) } : null} cursor={cursor} ariaLabel={t("loss.loss")} />;
  } else if (k === "inertia") {
    const conv = done?.reason === "converged" ? done.iteration : null;
    const last = h.loss[h.loss.length - 1];
    body = <LineChart series={[{ xs: h.it, ys: h.loss, color: "acc", dots: true }]} xMin={1} xMax={Math.max(conv ?? 0, h.it[h.it.length - 1] ?? 1, 2)} zeroBased
      markers={conv ? [{ x: conv, label: t("loss.convergedTol") }] : []} xLabel={t("unit.iterationPlural")}
      point={last != null ? { x: h.it[h.it.length - 1], y: last, label: fmt(last, 0) } : null} cursor={cursor} ariaLabel={t("loss.inertia")} />;
  } else if (k === "boost") {
    const best = latest.extra.best_stage as number | undefined;
    const es = latest.extra.early_stop as number | undefined;
    const marks = [];
    if (best) marks.push({ x: best, label: t("loss.bestStage", { n: best }) });
    if (es) marks.push({ x: es, label: t("loss.earlyStop", { n: es }) });
    body = <LineChart series={[{ xs: h.it, ys: h.loss, color: "acc" }, { xs: h.it, ys: h.lossTest, color: "ink2", dash: true }]} xMax={total} zeroBased
      markers={marks} xLabel={t(schema.stepUnit === "trees" ? "unit.treesPlural" : "unit.stagePlural")} glow cursor={cursor} ariaLabel={t("loss.boost")} />;
  } else if (k === "trees") {
    const oob = frames.map((f) => (f.extra.oob as number | null) ?? null);
    const series: Series[] = [{ xs: h.it, ys: h.mte, color: "acc" }];
    if (oob.some((v) => v != null)) series.push({ xs: frames.map((f) => f.iteration), ys: oob, color: "ink2", dash: true });
    const first = h.mte[0], last = h.mte[h.mte.length - 1];
    body = (
      <>
        <LineChart series={series} xMin={0} xMax={total} xLabel={t("unit.treesPlural")} glow
          point={last != null ? { x: h.it[h.it.length - 1], y: last, label: t("loss.nTrees", { n: h.it[h.it.length - 1], v: fmt(last) }) } : null}
          cursor={cursor} ariaLabel={t("loss.trees")} />
        <span className="hint">{first != null ? t("loss.firstTrees", { n: h.it[0], v: fmt(first) }) : ""}{oob.some((v) => v != null) ? " · " + t("loss.oobKey") : ""}</span>
      </>
    );
  } else if (k === "depth") {
    const by = (latest.extra.by_depth as { depth: number; train: number; test: number | null }[]) ?? [];
    const ms = latest.extra.fit_ms as number | undefined;
    body = (
      <div className="onestep">
        <div style={{ display: "flex", flexDirection: "column", gap: 6, width: "46%", minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}><Icon name="checkCircle" style={{ color: "var(--acc-t)" }} /><span style={{ fontSize: 13, fontWeight: 600 }}>{t("loss.onestep")}</span>{ms != null && <span className="hint">{t("loss.ms", { ms: ms < 1 ? "<1" : Math.round(ms) })}</span>}</div>
          <p>{t("loss.treeText")}</p>
          <span className="hint" dangerouslySetInnerHTML={{ __html: t("loss.treeReplay") }} />
        </div>
        <DepthBars by={by} cur={latest.iteration} />
      </div>
    );
  } else if (k === "onestep") body = <OneStepSummary />;
  else if (k === "onestep_reg") body = <RegSummary />;
  else if (k === "knn") {
    const curve = (latest.extra.curve as { k: number; test: number | null }[]) ?? [];
    const kk = latest.extra.k as number;
    const best = latest.extra.best_k as number | null;
    const cur = curve.find((c) => c.k === kk);
    body = (
      <>
        <span className="hint key"><span><i className="k-acc" />{t("loss.knnKey")}</span></span>
        <LineChart series={[{ xs: curve.map((c) => c.k), ys: curve.map((c) => c.test), color: "acc" }]} xMin={1} xMax={Math.max(2, curve.length)} xLabel=""
          markers={best ? [{ x: best, label: t("loss.bestK", { k: best }) }] : []}
          point={cur?.test != null ? { x: kk, y: cur.test, label: `k = ${kk} · ${fmt(cur.test)}` } : null} ariaLabel={t("loss.knn")} xTicks={[1, Math.round(curve.length / 2), curve.length]} />
      </>
    );
  } else if (k === "pca") body = <PcaBars />;
  return (
    <div className="lp">
      {body}
      <Ripple />
    </div>
  );
}

function DepthBars({ by, cur }: { by: { depth: number; train: number; test: number | null }[]; cur: number }) {
  const { t } = useTranslation();
  const shown = by.slice(-8);
  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 6, minWidth: 0 }}>
      <div className="prow"><span className="lbl">{t("loss.byDepth")}</span>
        <span className="hint" style={{ display: "flex", gap: 10 }}>
          <span style={{ display: "flex", alignItems: "center", gap: 5 }}><span className="dot" style={{ background: "var(--ink3)", borderRadius: 2 }} />{t("loss.train")}</span>
          <span style={{ display: "flex", alignItems: "center", gap: 5 }}><span className="dot" style={{ background: "var(--acc)", borderRadius: 2 }} />{t("loss.test")}</span>
        </span>
      </div>
      <div className="bars" style={{ gridTemplateColumns: `repeat(${Math.max(1, shown.length)}, minmax(0, 1fr))` }}>
        {shown.map((b) => (
          <div className="bcol" key={b.depth}>
            <span className="mono hint" style={{ color: b.depth === cur ? "var(--ink)" : undefined, visibility: shown.length > 5 && b.depth !== cur ? "hidden" : undefined }}>{fmt(b.test ?? b.train)}</span>
            <div className="bpair">
              <motion.i style={{ height: `${b.train * 100}%`, background: "var(--ink3)", opacity: 0.6 }} initial={{ scaleY: 0 }} animate={{ scaleY: 1 }} transition={{ type: "spring", stiffness: 300, damping: 30 }} />
              <motion.i style={{ height: `${(b.test ?? 0) * 100}%`, background: "var(--acc)" }} initial={{ scaleY: 0 }} animate={{ scaleY: 1 }} transition={{ type: "spring", stiffness: 300, damping: 30, delay: 0.05 }} />
            </div>
          </div>
        ))}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: `repeat(${Math.max(1, shown.length)}, minmax(0, 1fr))`, gap: 14, padding: "0 8px", textAlign: "center" }} className="hint">
        {shown.map((b) => <span key={b.depth}>{shown.length > 4 ? b.depth : t("loss.depthN", { d: b.depth })}</span>)}
      </div>
    </div>
  );
}

function OneStepSummary() {
  const { t } = useTranslation();
  const { name } = useModel();
  const done = useSession((s) => s.done);
  const ds = useSession((s) => s.dataset);
  const s = (done?.summary ?? {}) as Record<string, unknown>;
  const ms = s.fit_ms as number | undefined;
  const head = (
    <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 14, fontWeight: 600 }}>
      <Icon name="checkCircle" size={20} style={{ color: "var(--acc-t)" }} />{t("loss.doneIn", { ms: ms == null ? "—" : ms < 1 ? "<1" : Math.round(ms) })}
    </div>
  );
  if (!done) return <span className="hint">{t("loss.fitting")}</span>;
  if (name === "GaussianNB") {
    const pri = (s.priors as number[]) ?? [];
    const means = (s.means as number[][]) ?? [];
    const feats = ((s.features as string[]) ?? []).slice(0, 3);
    const cols = `60px repeat(${feats.length + 1}, minmax(0, 1fr))`;
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 8, overflow: "auto" }}>
        {head}
        <div className="tbl">
          <div className="r h" style={{ gridTemplateColumns: cols }}><span>{t("loss.class")}</span><span>{t("loss.prior")}</span>{feats.map((f) => <span key={f}>{t("loss.mean", { f })}</span>)}</div>
          {pri.map((p, i) => (
            <div className="r" key={i} style={{ gridTemplateColumns: cols }}>
              <span style={{ display: "flex", alignItems: "center", gap: 6 }}><span className={`dot k${i % 5} s${i % 4}`} style={{ width: 8, height: 8 }} />{ds?.classes[i]}</span>
              <span>{fmt(p)}</span>{feats.map((_, j) => <span key={j}>{fmt(means[i]?.[j])}</span>)}
            </div>
          ))}
        </div>
      </div>
    );
  }
  if (name === "SVC") {
    const ns = (s.n_support as number[]) ?? [];
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {head}
        <span style={{ fontSize: 13, color: "var(--ink2)" }} dangerouslySetInnerHTML={{ __html: t("loss.svcText", { sv: s.total, n: s.n_train }) }} />
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {ns.map((n, i) => <span className="tag" key={i}><span className={`dot k${i % 5} s${i % 4}`} />{ds?.classes[i]} · {n}</span>)}
        </div>
        <span className="hint">{t("loss.svcRing")}</span>
      </div>
    );
  }
  if (name === "DBSCAN") {
    const sizes = (s.sizes as number[]) ?? [];
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {head}
        <span style={{ fontSize: 13, color: "var(--ink2)" }} dangerouslySetInnerHTML={{ __html: t("loss.dbText", { c: s.clusters, noise: s.noise, core: s.core }) }} />
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {sizes.slice(0, 10).map((n, i) => <span className="tag" key={i}><span className={`dot k${i % 5} s${i % 4}`} />{t("loss.clusterN", { i: i + 1 })} · {n}</span>)}
          <span className="tag"><span className="dot" style={{ boxShadow: "inset 0 0 0 1.5px var(--ink3)" }} />{t("loss.noise")} · {String(s.noise)}</span>
        </div>
      </div>
    );
  }
  return head;
}

function RegSummary() {
  const { t } = useTranslation();
  const done = useSession((s) => s.done);
  const s = (done?.summary ?? {}) as Record<string, unknown>;
  const res = s.residuals as { counts: number[]; lo: number; hi: number } | undefined;
  if (!done || !res) return <span className="hint">{t("loss.fitting")}</span>;
  const mx = Math.max(1, ...res.counts);
  const mid = res.counts.length / 2;
  return (
    <div style={{ display: "flex", alignItems: "flex-end", gap: 16, height: "100%", paddingTop: 4 }}>
      <div style={{ display: "flex", flexDirection: "column" }}><span className="lbl">R²</span><Ticker value={(s.r2 as number) ?? null} style={{ fontSize: 34, fontWeight: 600, lineHeight: 1.1 }} /></div>
      <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 4, height: "100%", minWidth: 0 }}>
        <span className="lbl">{t("loss.residuals")}</span>
        <div style={{ flex: 1, display: "flex", alignItems: "flex-end", gap: 2, borderBottom: "1px solid var(--line2)", minHeight: 40 }}>
          {res.counts.map((c, i) => (
            <motion.i key={i} style={{ flex: 1, height: `${(c / mx) * 100}%`, background: Math.abs(i + 0.5 - mid) < 1.5 ? "var(--acc)" : "var(--ink3)", opacity: Math.abs(i + 0.5 - mid) < 1.5 ? 1 : 0.7, borderRadius: "2px 2px 0 0", transformOrigin: "bottom" }}
              initial={{ scaleY: 0 }} animate={{ scaleY: 1 }} transition={{ type: "spring", stiffness: 300, damping: 30, delay: i * 0.02 }} />
          ))}
        </div>
        <div className="mono hint" style={{ display: "flex", justifyContent: "space-between" }}><span>{fmt(res.lo, 1)}</span><span>0</span><span>+{fmt(res.hi, 1)}</span></div>
      </div>
    </div>
  );
}

function PcaBars() {
  const { t } = useTranslation();
  const done = useSession((s) => s.done);
  const s = (done?.summary ?? {}) as Record<string, unknown>;
  const ratio = ((s.ratio as number[]) ?? []).slice(0, 8);
  const kept = (s.kept as number) ?? 2;
  if (!done) return <span className="hint">{t("loss.fitting")}</span>;
  const share = ratio.slice(0, kept).reduce((a, b) => a + b, 0);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4, height: "100%" }}>
      <div className="bars" style={{ gridTemplateColumns: `repeat(${ratio.length}, minmax(0, 1fr))`, gap: 10 }}>
        {ratio.map((r, i) => (
          <div className="bcol" key={i}>
            <span className={`mono${i < kept ? "" : " hint"}`} style={{ fontSize: 12 }}>{Math.round(r * 100)}%</span>
            <motion.div style={{ width: "70%", height: `${r * 100}%`, background: i < kept ? "var(--acc)" : "var(--ink3)", opacity: i < kept ? 1 : 0.6, borderRadius: "3px 3px 0 0", transformOrigin: "bottom" }}
              initial={{ scaleY: 0 }} animate={{ scaleY: 1 }} transition={{ type: "spring", stiffness: 300, damping: 30, delay: i * 0.04 }} />
          </div>
        ))}
      </div>
      <div className="mono hint" style={{ display: "grid", gridTemplateColumns: `repeat(${ratio.length}, minmax(0, 1fr))`, textAlign: "center", gap: 10 }}>{ratio.map((_, i) => <span key={i}>PC{i + 1}</span>)}</div>
      <span className="hint" dangerouslySetInnerHTML={{ __html: t("loss.pcaText", { n: kept, p: Math.round(share * 100) }) }} />
    </div>
  );
}

// ------------------------------------------------------------------ metric

function metricInfo(task: string | undefined, name: string | null) {
  if (task === "regression") return "r2";
  if (name === "PCA") return "variance";
  if (task === "clustering") return "silhouette";
  return "accuracy";
}

export function MetricHeader(props: IDockviewPanelHeaderProps) {
  const { t } = useTranslation();
  const { schema, name } = useModel();
  const latest = useSession((s) => s.latest);
  const m = metricInfo(schema?.task, name);
  const v = latest ? (schema?.task === "clustering" || name === "PCA" ? latest.metric_train : latest.metric_test ?? latest.metric_train) : null;
  return (
    <div className="ph sm">
      <div className="ph-l" style={{ gap: 6 }}>
        <span className="lbl" style={{ color: "var(--ink2)", fontSize: 12 }}>{t(`metric.${m}Title`)}</span>
        <Info label={t(`metric.${m}About`)}><b>{t(`metric.${m}Title`)}</b> — {t(`metric.${m}Tip`)}</Info>
      </div>
      <div className="ph-r" style={{ gap: 8 }}>
        <DeltaChip value={schema?.iterative ? v : null} />
        <PanelButtons id={props.api.id} />
      </div>
    </div>
  );
}

export function MetricPanel(_p: IDockviewPanelProps) {
  const { t } = useTranslation();
  const { schema, name } = useModel();
  const latest = useFrame();
  const ds = useSession((s) => s.dataset);
  const frames = useSession((s) => s.frames);
  const phase = useSession(trainPhase);
  const m = metricInfo(schema?.task, name);
  const unsup = schema?.task === "clustering" || name === "PCA";
  const v = latest ? (unsup ? latest.metric_train : latest.metric_test ?? latest.metric_train) : null;
  let sub = "";
  if (!latest) sub = phase === "incompatible" ? "" : t(`metric.${m}After`);
  else if (m === "silhouette") sub = v == null ? t("metric.silNone") : v > 0.5 ? t("metric.silGood") : v > 0.25 ? t("metric.silOk") : t("metric.silWeak");
  else if (m === "variance") sub = t("metric.varSub", { n: (latest.extra.kept as number) ?? 2 });
  else if (ds?.n_test) {
    const prev = frames.length > 10 ? frames[frames.length - 11]?.metric_test : null;
    const trend = phase === "training" && prev != null && v != null ? (v > prev + 0.005 ? t("metric.rising") : v < prev - 0.005 ? t("metric.falling") : "") : "";
    sub = t(m === "r2" ? "metric.onTestR2" : "metric.onTest", { n: ds.n_test }) + (trend ? " · " + trend : "");
  }
  const foot: ReactNode[] = [];
  if (latest) {
    if (!unsup && latest.metric_train != null) foot.push(<span key="tr">{t("metric.train")} <span className="mono">{fmt(latest.metric_train)}</span></span>);
    if (schema?.lossKind === "loss" && latest.loss != null && name !== "MLPClassifier") foot.push(<span key="l">{t(name === "Lasso" || name === "ElasticNet" ? "metric.objective" : "metric.logloss")} <span className="mono">{fmt(latest.loss, 2)}</span></span>);
    if (name === "MLPClassifier" && frames.length > 10) foot.push(<span key="ago">{t("metric.tenAgo", { unit: t("unit.epochPlural") })} <span className="mono">{fmt(frames[frames.length - 11].metric_test)}</span></span>);
    if (name === "DecisionTreeClassifier" && latest.viz.tree) foot.push(<span key="lv">{t("metric.leaves")} <span className="mono">{latest.viz.tree.leaves}</span></span>);
    if (name === "KMeans") foot.push(<span key="in">{t("metric.inertia")} <span className="mono">{fmt(latest.loss, 0)}</span></span>, <span key="g">{t("metric.groups")} <span className="mono">{latest.viz.centers?.shape[0]}</span></span>);
    if (name === "DBSCAN") foot.push(<span key="c">{t("metric.clusters")} <span className="mono">{String(latest.extra.clusters)}</span></span>, <span key="n">{t("metric.noise")} <span className="mono">{String(latest.extra.noise)}</span></span>);
  }
  return (
    <div className="met" data-guide="metric" data-testid="metric">
      <Ticker value={v} className="big" />
      <div className="hint" style={{ fontSize: 13 }}>{sub}</div>
      <div className="foot">{foot}</div>
      <Ripple />
    </div>
  );
}

// ------------------------------------------------------------------ network (MLP) + regions (tree)

export function NetworkHeader(props: IDockviewPanelHeaderProps) {
  const { t } = useTranslation();
  const latest = useSession((s) => s.latest);
  const net = latest?.viz.network;
  const act = String(currentParams().activation ?? "relu");
  return (
    <div className="ph">
      <div className="ph-l"><h2>{t("net.title")}</h2><Info label={t("net.about")}>{t("net.tip")}</Info></div>
      <div className="ph-r">
        {net && <span className="hint mono">{net.sizes.join(" → ")} · {act}</span>}
        <PanelButtons id={props.api.id} />
      </div>
    </div>
  );
}

export function NetworkPanel(_p: IDockviewPanelProps) {
  const { t } = useTranslation();
  const latest = useFrame();
  const probe = useSession((s) => s.probe);
  const hover = useUi((s) => s.hoverId);
  const ds = useSession((s) => s.dataset);
  const net = latest?.viz.network ?? null;
  const acts = probe?.layers.map((l) => l.act) ?? null;
  const sizes = net?.sizes ?? [];
  const act = String(currentParams().activation ?? "relu");
  return (
    <div className="pbody" style={{ padding: "0 12px 10px" }}>
      {!net ? (
        <div className="ghost" style={{ margin: "6px 6px 12px" }}><span className="hint">{t("net.empty")}</span></div>
      ) : (
        <>
          <div className="nhead" style={{ gridTemplateColumns: `repeat(${sizes.length}, minmax(0, 1fr))` }}>
            {sizes.map((n, i) => (
              <span key={i} className="lbl">{i === 0 ? t("net.input") : i === sizes.length - 1 ? t("net.output") : t("net.hidden", { i })}<br />
                <span className="mono">{i === 0 ? n : i === sizes.length - 1 ? (latest?.viz.network?.out_activation ?? "") : `${n} · ${act}`}</span></span>
            ))}
          </div>
          <div className="netv"><NetworkView net={net} acts={acts} featureNames={ds?.plane?.mode === "raw" ? ds.feature_names : undefined} /></div>
          <div className="hint" style={{ display: "flex", flexWrap: "wrap", gap: "6px 14px" }}>
            <span style={{ display: "flex", alignItems: "center", gap: 6 }}><i className="k-acc" style={{ borderTopWidth: 1.5 }} />{t("net.pos")}</span>
            <span style={{ display: "flex", alignItems: "center", gap: 6 }}><i className="k-ink" />{t("net.neg")}</span>
            <span>{probe ? t("net.fill", { id: probe.row_id }) : hover == null ? t("net.hoverHint") : ""}</span>
          </div>
        </>
      )}
    </div>
  );
}

export function RegionsHeader(props: IDockviewPanelHeaderProps) {
  const { t } = useTranslation();
  return (
    <div className="ph sm" style={{ padding: "10px 10px 4px" }}>
      <div className="ph-l" style={{ gap: 6 }}><span className="lbl">{t("regions.title")}</span><Info label={t("regions.about")}>{t("regions.tip")}</Info></div>
      <PanelButtons id={props.api.id} />
    </div>
  );
}

export function RegionsPanel(_p: IDockviewPanelProps) {
  const { t } = useTranslation();
  const latest = useFrame();
  const treeNode = useUi((s) => s.treeNode);
  const extra = useMemo(() => () => regionLines(latest?.viz.tree, treeNode), [latest, treeNode]);
  return (
    <div className="pbody" style={{ padding: "0 10px 10px", gap: 6 }}>
      <div style={{ position: "relative", flex: 1, minHeight: 80 }}><PlotView variant="inset" axes={false} border={false} extra={extra} /></div>
      <span className="hint">{treeNode != null ? t("regions.selected", { id: treeNode }) : t("regions.cut")}</span>
    </div>
  );
}

export function HiddenHeader() {
  return null;
}

export type { TrainPhase };
