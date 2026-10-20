// Handbook: one chapter per model (plus an intro), in EN and RU, with KaTeX formulas, a
// parameter table, a runnable Python example and a "Try it in the app" button.
import "katex/dist/katex.min.css";
import "../../styles/handbook.css";
import { motion, useReducedMotion } from "motion/react";
import { useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import { useTranslation } from "react-i18next";
import { BY_NAME } from "../../app/catalog";
import { useNav } from "../../app/nav";
import { engine } from "../../engine/client";
import { loadPreset, selectModel, toast } from "../../engine/store";
import { CHAPTERS, CHAPTER_BY_ID, type Chapter } from "../../handbook/chapters";
import { Icon } from "../../ui/Icon";
import { highlightPython, localiseCode, renderMarkdown } from "./md";

const TEXTS = import.meta.glob<string>("../../handbook/{en,ru}/*.md", { query: "?raw", import: "default" });
const CODE = import.meta.glob<string>("../../handbook/code/*.py", { query: "?raw", import: "default" });

function useChapterText(id: string, lang: "en" | "ru") {
  const [state, setState] = useState<{ key: string; md: string; py: string } | null>(null);
  const key = `${lang}/${id}`;
  useEffect(() => {
    let live = true;
    const md = TEXTS[`../../handbook/${lang}/${id}.md`] ?? TEXTS[`../../handbook/en/${id}.md`];
    const py = CODE[`../../handbook/code/${id}.py`];
    void Promise.all([md ? md() : Promise.resolve(""), py ? py() : Promise.resolve("")]).then(([m, p]) => {
      if (live) setState({ key, md: m, py: p });
    });
    return () => { live = false; };
  }, [id, lang, key]);
  return state?.key === key ? state : null;
}

export function Handbook() {
  const { t, i18n } = useTranslation();
  const lang = (i18n.language?.startsWith("ru") ? "ru" : "en") as "en" | "ru";
  const { chapter: id, anchor, go } = useNav();
  const ch = CHAPTER_BY_ID[id] ?? CHAPTERS[0];
  const text = useChapterText(ch.id, lang);
  const doc = useRef<HTMLDivElement>(null);
  const reduce = useReducedMotion();

  const parts = useMemo(() => (text ? renderMarkdown(text.md).split(/<p>\{\{(code|try)\}\}<\/p>/) : []), [text]);
  const code = useMemo(() => (text?.py ? localiseCode(text.py, lang) : ""), [text, lang]);

  // jump to the requested section (or the top) whenever the chapter or anchor changes
  useEffect(() => {
    if (!text) return;
    const el = anchor ? doc.current?.querySelector(`#${CSS.escape(anchor)}`) : null;
    if (el) el.scrollIntoView({ block: "start", behavior: reduce ? "auto" : "smooth" });
    else doc.current?.scrollTo({ top: 0 });
  }, [text, anchor, reduce]);

  // in-text links: "#hb/<chapter>/<anchor>" stay inside the Handbook
  const onClick = (e: MouseEvent) => {
    const a = (e.target as HTMLElement).closest("a");
    const href = a?.getAttribute("href") ?? "";
    if (href.startsWith("#hb/")) {
      e.preventDefault();
      const [, c, s] = href.split("/");
      go("handbook", { chapter: c || ch.id, anchor: s ?? null });
    }
  };

  const title = ch.id === "intro" ? t("hb.introTitle") : ch.id === "CatBoost" ? "CatBoost" : ch.id;
  return (
    <div className="handbook" data-testid="handbook">
      <nav className="glass hb-toc" aria-label={t("hb.contents")}>
        <h2>{t("nav.handbook")}</h2>
        {(["basics", "linear", "trees", "other", "unsupervised"] as const).map((g) => (
          <div key={g} className="hb-grp">
            <span className="lbl">{g === "basics" ? t("hb.basics") : t(`groups.${g}.title`)}</span>
            {CHAPTERS.filter((c) => c.group === g).map((c) => (
              <button key={c.id} className={c.id === ch.id ? "on" : ""} aria-current={c.id === ch.id ? "page" : undefined}
                onClick={() => go("handbook", { chapter: c.id, anchor: null })} data-testid={`hb-${c.id}`}>
                {c.id === ch.id && !reduce && <motion.span layoutId="hb-ind" className="hb-ind" transition={{ type: "spring", stiffness: 300, damping: 30 }} />}
                <span className={c.id === "intro" ? "" : "mono"}>{c.id === "intro" ? t("hb.introTitle") : c.id.replace(/(Classifier|Regression|Regressor)$/, "\u200b$1")}</span>
              </button>
            ))}
          </div>
        ))}
      </nav>
      <article className="solid hb-doc" ref={doc} onClick={onClick} lang={lang} aria-busy={!text}>
        <header className="hb-head">
          <span className="lbl">{ch.id === "intro" ? t("hb.basics") : t(`groups.${ch.group}.title`)}</span>
          <h1 className={ch.id === "intro" ? "" : "mono"}>{title}</h1>
          {ch.model && <Glance ch={ch} />}
          {ch.model && <TryButton ch={ch} />}
        </header>
        {text && (
          <motion.div key={text.key} className="hb-body" initial={reduce ? { opacity: 0 } : { opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }}>
            {parts.map((p, i) =>
              i % 2 === 0 ? <div key={i} dangerouslySetInnerHTML={{ __html: p }} />
                : p === "code" ? <CodeBlock key={i} code={code} file={`${ch.id}.py`} />
                  : <TryButton key={i} ch={ch} big />,
            )}
          </motion.div>
        )}
      </article>
    </div>
  );
}

/** The card summary (what/how, when, strength, weakness), so the chapter starts where the card ends. */
function Glance({ ch }: { ch: Chapter }) {
  const { t } = useTranslation();
  const names = ch.id === "CatBoost" ? ["CatBoostClassifier"] : [ch.model!];
  const m = names[0];
  return (
    <dl className="hb-glance">
      <dt>{t("hb.whatHow")}</dt><dd>{t(`models.${m}.desc`)}</dd>
      <dt>{t("home.useWhen")}</dt><dd>{t(`models.${m}.when`)}</dd>
      <dt className="pro">+</dt><dd>{t(`models.${m}.pro`)}</dd>
      <dt className="con">−</dt><dd>{t(`models.${m}.con`)}</dd>
    </dl>
  );
}

function TryButton({ ch, big }: { ch: Chapter; big?: boolean }) {
  const { t } = useTranslation();
  const go = useNav((s) => s.go);
  if (!ch.model) return null;
  const run = async () => {
    try {
      selectModel(ch.model!);
      if (ch.generator) await engine.request({ type: "load_dataset", spec: { kind: "generator", generator: ch.generator } });
      else if (ch.preset) await loadPreset(ch.preset);
      go("workspace");
    } catch (e) {
      const err = e as { message?: { en: string; ru: string } };
      toast("hb", err.message ?? { en: "Couldn't open the model.", ru: "Не удалось открыть модель." });
    }
  };
  const what = ch.generator ? t("hb.tryGenerated") : t(`presets.${ch.preset}.name`, { defaultValue: ch.preset });
  return (
    <div className={`hb-try${big ? " big" : ""}`}>
      <button className="btn pri" onClick={() => void run()} data-testid="hb-try"><Icon name="play" fill /><span className="bl">{t("hb.try")}</span></button>
      <span className="hint">{t("hb.tryWith", { model: BY_NAME[ch.model]?.name ?? ch.model, data: what })}</span>
    </div>
  );
}

function CodeBlock({ code, file }: { code: string; file: string }) {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);
  const html = useMemo(() => highlightPython(code), [code]);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch { /* clipboard blocked: the text is selectable */ }
  };
  return (
    <figure className="hb-code">
      <figcaption><span className="mono">{file}</span>
        <button className="btn sm gh" onClick={() => void copy()} aria-label={t("hb.copy")}><Icon name={copied ? "check" : "copy"} size={14} />{copied ? t("hb.copied") : t("hb.copy")}</button>
      </figcaption>
      <pre><code dangerouslySetInnerHTML={{ __html: html }} /></pre>
    </figure>
  );
}
