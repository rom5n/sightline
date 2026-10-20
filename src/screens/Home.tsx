import { motion, useReducedMotion } from "motion/react";
import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { BY_NAME, CATALOG, GROUPS, type Task } from "../app/catalog";
import { useNav } from "../app/nav";
import { useSettings, type Recent } from "../app/settings";
import { loadPreset, selectModel, setSpeed, train } from "../engine/store";
import { Icon } from "../ui/Icon";
import { fmt, Info, Seg } from "../ui/controls";
import { ModelPreview } from "./ModelPreview";
import { startGuide } from "./Guide";

export function openModel(name: string) {
  selectModel(name);
  useNav.getState().go("workspace");
}

export async function startGuidedRun() {
  useSettings.getState().set({ firstRunDone: true });
  openModel("LogisticRegression");
  await loadPreset("moons", { n_samples: 300, noise: 0.15 });
  setSpeed(1);
  train();
  startGuide();
}

function ago(t: (k: string, o?: Record<string, unknown>) => string, at: number) {
  const s = (Date.now() - at) / 1000;
  if (s < 90) return t("time.now");
  if (s < 3600) return t("time.min", { count: Math.round(s / 60) });
  if (s < 86400) return t("time.hour", { count: Math.round(s / 3600) });
  if (s < 172800) return t("time.yesterday");
  return t("time.day", { count: Math.round(s / 86400) });
}

export function Home() {
  const { t, i18n } = useTranslation();
  const [q, setQ] = useState("");
  const [task, setTask] = useState<"all" | Task>("all");
  const recent = useSettings((s) => s.recent);
  const firstRunDone = useSettings((s) => s.firstRunDone);
  const reduce = useReducedMotion();
  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return CATALOG.filter((m) => {
      if (task !== "all" && m.task !== task) return false;
      if (!needle) return true;
      const hay = [m.name, t(`models.${m.name}.desc`), t(`models.${m.name}.keywords`), t(`task.${m.task}`)].join(" ").toLowerCase();
      return hay.includes(needle);
    });
  }, [q, task, t, i18n.language]);

  const openRecent = async (r: Recent) => {
    openModel(r.model);
    if (r.preset) await loadPreset(r.preset, r.settings);
  };

  return (
    <div className="scroll" data-testid="home">
      <main className="home-main">
        <div className="home-head">
          <div style={{ display: "flex", flexDirection: "column", gap: 8, maxWidth: 640 }}>
            <h1>{t("home.title")}</h1>
            <p>{t("home.subtitle")}</p>
          </div>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 10 }}>
            <label className="glass srch">
              <Icon name="search" style={{ color: "var(--ink3)" }} />
              <input type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder={t("home.search")} aria-label={t("home.searchLabel")} />
            </label>
            <Seg ariaLabel={t("home.filter")} value={task} onChange={setTask} options={[
              { value: "all", label: t("home.all") },
              { value: "regression", label: t("task.regression") },
              { value: "classification", label: t("task.classification") },
              { value: "clustering", label: t("task.clustering") },
              { value: "reduction", label: t("task.reductionShort") },
            ]} />
          </div>
        </div>

        {!firstRunDone && (
          <section className="glass banner" data-testid="first-run">
            <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
              <div className="pvb" style={{ width: 72, height: 54 }}><ModelPreview pv="linb" small /></div>
              <div style={{ display: "flex", flexDirection: "column", gap: 4, maxWidth: 640 }}>
                <span style={{ fontSize: 17, fontWeight: 600 }}>{t("first.title")}</span>
                <span style={{ fontSize: 14, color: "var(--ink2)" }}>{t("first.body")}</span>
              </div>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="btn gh" onClick={() => useSettings.getState().set({ firstRunDone: true })}>{t("first.skip")}</button>
              <button className="btn pri" onClick={() => void startGuidedRun()}>{t("first.start")}</button>
            </div>
          </section>
        )}

        {recent.length > 0 && !q && task === "all" && (
          <section style={{ display: "flex", flexDirection: "column", gap: 12 }} aria-labelledby="recent-h">
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <Icon name="clock" style={{ color: "var(--ink3)" }} />
              <h2 id="recent-h" style={{ fontSize: 15, fontWeight: 600 }}>{t("home.recent")}</h2>
            </div>
            <div className="cards recent">
              {recent.slice(0, 3).map((r) => (
                <motion.button key={r.model} className="glass card" style={{ alignItems: "center" }} onClick={() => void openRecent(r)}
                  whileHover={reduce ? undefined : { y: -2 }} transition={{ duration: 0.15 }}>
                  <div className="pvb" style={{ width: 64, height: 48 }}><ModelPreview pv={BY_NAME[r.model]?.pv ?? "linb"} small /></div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 0, flex: 1 }}>
                    <span className="cname">{r.model}</span>
                    <span className="hint" style={{ fontSize: 13 }}>{r.preset ? t(`presets.${r.preset}.name`) : r.dataset} · {t("home.rows", { count: r.rows })}</span>
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 2 }}>
                    <span className="mono" style={{ fontSize: 15, fontWeight: 600 }}>{fmt(r.metric)}</span>
                    <span className="hint">{t(`metric.${r.metricName}`)} · {ago(t, r.at)}</span>
                  </div>
                </motion.button>
              ))}
            </div>
          </section>
        )}

        {GROUPS.map((g) => {
          const items = filtered.filter((m) => m.group === g);
          if (!items.length) return null;
          return (
            <section key={g} style={{ display: "flex", flexDirection: "column", gap: 14 }} aria-labelledby={`g-${g}`}>
              <div className="grp-head">
                <h2 id={`g-${g}`}>{t(`groups.${g}.title`)}</h2>
                <span className="tag mono">{CATALOG.filter((m) => m.group === g).length}</span>
                <span style={{ fontSize: 13, color: "var(--ink3)" }}>{t(`groups.${g}.desc`)}</span>
              </div>
              <div className="cards">
                {items.map((m) => (
                  <motion.div key={m.name} className="cardw" whileHover={reduce ? undefined : { y: -2 }} transition={{ duration: 0.15 }}>
                    <button className="glass card" onClick={() => openModel(m.name)} data-testid={`card-${m.name}`}>
                      <div className="pvb"><ModelPreview pv={m.pv} /></div>
                      <div style={{ display: "flex", flexDirection: "column", gap: 8, minWidth: 0, flex: 1, textAlign: "left" }}>
                        <span className="cname">{m.name.replace(/(Classifier|Regression)$/, "​$1")}</span>
                        <span className="tag" style={{ alignSelf: "flex-start" }}>{t(`task.${m.task}Long`)}</span>
                      </div>
                    </button>
                    {/* the description lives in a popup so the cards stay compact */}
                    <Info label={t("home.about", { name: m.name })}>
                      <div className="mtip" data-testid={`about-${m.name}`}>
                        <b className="mono">{m.name}</b>
                        <p>{t(`models.${m.name}.desc`)}</p>
                        <dl className="cfacts">
                          <dt>{t("home.useWhen")}</dt><dd>{t(`models.${m.name}.when`)}</dd>
                          <dt className="pro" aria-label={t("home.pro")}>+</dt><dd>{t(`models.${m.name}.pro`)}</dd>
                          <dt className="con" aria-label={t("home.con")}>−</dt><dd>{t(`models.${m.name}.con`)}</dd>
                        </dl>
                      </div>
                    </Info>
                  </motion.div>
                ))}
              </div>
            </section>
          );
        })}
        {!filtered.length && <div className="empty-search">{t("home.noMatch", { q })}</div>}
      </main>
    </div>
  );
}
