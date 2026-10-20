import i18n from "i18next";
import { initReactI18next, useTranslation } from "react-i18next";
import en from "../locales/en.json";
import ru from "../locales/ru.json";
import { useSettings } from "../app/settings";
import type { Bi } from "../engine/types";

// Dev-only long-string mode (Ctrl+Shift+L or ?pseudo=1): every translated string, and every
// engine-provided label, grows by ~40 % with longer words, to catch overflow before a real
// language does. Never active in production builds.
const PSEUDO_KEY = "sightline.pseudo";
let pseudo = import.meta.env.DEV && (new URLSearchParams(location.search).has("pseudo") || safeGet(PSEUDO_KEY) === "1");

function safeGet(k: string) {
  try { return localStorage.getItem(k); } catch { return null; }
}

/** Stretch letters-only runs by ~40 % (doubling every 2nd–3rd letter); tags and numbers stay intact. */
export function stretch(text: string): string {
  return text.replace(/(<[^>]*>)|([A-Za-zА-Яа-яЁё]{2,})/g, (m, tag: string | undefined) => {
    if (tag) return m;
    let out = "";
    for (let i = 0; i < m.length; i++) out += m[i] + (i % 5 === 1 || i % 5 === 3 ? m[i].toLowerCase() : "");
    return out;
  });
}

void i18n.use(initReactI18next).use({
  type: "postProcessor",
  name: "pseudo",
  process: (value: string) => (pseudo ? stretch(value) : value),
}).init({
  resources: { en: { translation: en }, ru: { translation: ru } },
  lng: useSettings.getState().lang,
  fallbackLng: "en",
  interpolation: { escapeValue: false },
  returnNull: false,
  postProcess: ["pseudo"],
});
document.documentElement.lang = useSettings.getState().lang;

export function togglePseudo() {
  if (!import.meta.env.DEV) return;
  pseudo = !pseudo;
  try { localStorage.setItem(PSEUDO_KEY, pseudo ? "1" : "0"); } catch { /* private mode */ }
  void i18n.changeLanguage(i18n.language); // re-render every translated string
}

export function setLang(lang: "en" | "ru") {
  void i18n.changeLanguage(lang);
  document.documentElement.lang = lang;
  useSettings.getState().set({ lang });
}

/** Pick the right half of an {en, ru} pair coming from the engine. */
export function useBi() {
  const { i18n: inst } = useTranslation();
  const lang = (inst.language?.startsWith("ru") ? "ru" : "en") as "en" | "ru";
  return (b: Bi | undefined | null) => {
    const v = b ? b[lang] || b.en : "";
    return pseudo ? stretch(v) : v;
  };
}

export default i18n;
