import { create } from "zustand";

export type Screen = "home" | "workspace" | "data" | "insights" | "decide" | "handbook";
export type DataTab = "presets" | "table" | "generator";

interface Nav {
  screen: Screen;
  /** true when Data was opened from Home (no model chosen) */
  standaloneData: boolean;
  dataTab: DataTab;
  advancedOpen: boolean;
  perfOverlay: boolean;
  /** open Handbook chapter (and optional section anchor) */
  chapter: string;
  anchor: string | null;
  go: (screen: Screen, opts?: { standalone?: boolean; dataTab?: DataTab; chapter?: string; anchor?: string | null }) => void;
}

export const useNav = create<Nav>((set) => ({
  screen: "home",
  standaloneData: false,
  dataTab: "presets",
  advancedOpen: false,
  perfOverlay: false,
  chapter: "intro",
  anchor: null,
  go: (screen, opts) =>
    set((s) => ({
      screen,
      standaloneData: opts?.standalone ?? (screen === "data" ? s.standaloneData : false),
      dataTab: opts?.dataTab ?? s.dataTab,
      advancedOpen: false,
      chapter: opts?.chapter ?? s.chapter,
      anchor: opts?.anchor !== undefined ? opts.anchor : screen === "handbook" ? null : s.anchor,
    })),
}));
