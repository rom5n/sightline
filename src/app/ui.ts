import { create } from "zustand";

/** View state shared by the workspace panels (and mirrored to pop-out windows). */
export interface UiState {
  heatLayer: "proba" | "border";
  kmLayers: ("regions" | "paths")[];
  pointsOnly: boolean;
  addClass: number;
  hoverId: number | null;
  decidePoint: number | null;
  treeNode: number | null;
  treeView: "tree" | "regions";
  mlpView: "both" | "boundary" | "network";
  insightHover: number | null;
  showOutliers: boolean;
  guideStep: number;
}

export const useUi = create<UiState>(() => ({
  heatLayer: "proba",
  kmLayers: ["regions", "paths"],
  pointsOnly: false,
  addClass: 0,
  hoverId: null,
  decidePoint: null,
  treeNode: null,
  treeView: "tree",
  mlpView: "both",
  insightHover: null,
  showOutliers: true,
  guideStep: 0,
}));
