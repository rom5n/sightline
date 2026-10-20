import { create } from "zustand";
import { loadSettingsFile, saveSettingsFile } from "./tauri";

export type Theme = "dark" | "light" | "system";
export type Lang = "en" | "ru";

export interface Recent {
  model: string;
  dataset: string;
  rows: number;
  metric: number | null;
  metricName: string;
  at: number;
  preset: string | null;
  settings: Record<string, number>;
}

export interface Settings {
  theme: Theme;
  lang: Lang;
  recent: Recent[];
  recentData: { name: string; preset: string | null; settings: Record<string, number>; rows: number; at: number }[];
  layouts: Record<string, unknown>;
  firstRunDone: boolean;
  speed: number;
}

const KEY = "sightline.settings";
const DEFAULTS: Settings = { theme: "system", lang: "en", recent: [], recentData: [], layouts: {}, firstRunDone: false, speed: 1 };

function readLocal(): Settings {
  try {
    return { ...DEFAULTS, ...(JSON.parse(localStorage.getItem(KEY) || "{}") as Partial<Settings>) };
  } catch {
    return { ...DEFAULTS };
  }
}

interface SettingsStore extends Settings {
  set: (p: Partial<Settings>) => void;
  pushRecent: (r: Recent) => void;
  pushRecentData: (d: Settings["recentData"][number]) => void;
}

let saveTimer: number | undefined;
function persist(s: Settings) {
  const json = JSON.stringify(s);
  try {
    localStorage.setItem(KEY, json);
  } catch {
    /* quota */
  }
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => void saveSettingsFile(json), 300) as unknown as number;
}

function pick(s: SettingsStore): Settings {
  const { theme, lang, recent, recentData, layouts, firstRunDone, speed } = s;
  return { theme, lang, recent, recentData, layouts, firstRunDone, speed };
}

export const useSettings = create<SettingsStore>((set, get) => ({
  ...readLocal(),
  set: (p) => {
    set(p);
    persist(pick(get()));
  },
  pushRecent: (r) => {
    const rest = get().recent.filter((x) => x.model !== r.model);
    get().set({ recent: [r, ...rest].slice(0, 6) });
  },
  pushRecentData: (d) => {
    const rest = get().recentData.filter((x) => !(x.name === d.name && x.preset === d.preset));
    get().set({ recentData: [d, ...rest].slice(0, 6) });
  },
}));

/** The app-data file (Tauri) wins over localStorage once it has loaded. */
export async function hydrateSettings() {
  const txt = await loadSettingsFile();
  if (!txt) return;
  try {
    const s = { ...DEFAULTS, ...(JSON.parse(txt) as Partial<Settings>) };
    useSettings.setState(s);
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    /* corrupt file: keep local copy */
  }
}

export function resolvedTheme(t: Theme): "dark" | "light" {
  if (t !== "system") return t;
  return matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}
