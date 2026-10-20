// Guided first run: four short coach marks anchored to the workspace panels.
import { AnimatePresence, motion } from "motion/react";
import { useLayoutEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useUi } from "../app/ui";
import { SPRING } from "../ui/controls";

const STEPS = ["viz", "loss", "params", "metric"] as const;

export function startGuide() {
  useUi.setState({ guideStep: 1 });
}

export function Guide() {
  const { t } = useTranslation();
  const step = useUi((s) => s.guideStep);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);
  const target = step > 0 ? STEPS[step - 1] : null;
  useLayoutEffect(() => {
    if (!target) return;
    const place = () => {
      const el = document.querySelector(`[data-guide="${target}"]`);
      if (!el) { setPos(null); return; }
      const r = el.getBoundingClientRect();
      const w = 300;
      let left = target === "params" ? r.left - w - 16 : r.left + 24;
      let top = target === "viz" ? r.top + 60 : target === "params" ? r.top + 120 : r.top - 160;
      left = Math.max(12, Math.min(innerWidth - w - 12, left));
      top = Math.max(48, Math.min(innerHeight - 180, top));
      setPos({ left, top });
    };
    place();
    const id = setInterval(place, 400);
    return () => clearInterval(id);
  }, [target]);
  const close = () => useUi.setState({ guideStep: 0 });
  return (
    <AnimatePresence>
      {target && pos && (
        <motion.div key={target} className="guide" role="dialog" aria-label={t("guide.label")} style={{ left: pos.left, top: pos.top }}
          initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={SPRING}>
          <span className="lbl">{t("guide.step", { n: step, total: STEPS.length })}</span>
          <b>{t(`guide.${target}.title`)}</b>
          <span>{t(`guide.${target}.body`)}</span>
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
            <button className="btn gh sm" onClick={close}>{t("guide.skip")}</button>
            <button className="btn pri sm" onClick={() => (step >= STEPS.length ? close() : useUi.setState({ guideStep: step + 1 }))}>
              {step >= STEPS.length ? t("guide.done") : t("guide.next")}
            </button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
