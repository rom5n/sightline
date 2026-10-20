// Payload types (the envelopes live in protocol.gen.ts, generated from Pydantic).
import type { DatasetInfo, Done, ExplainSteps, Frame, Preview, Ready, Status } from "./protocol.gen";

export type ND<T> = { shape: number[]; data: T };
export type Bi = { en: string; ru: string };
export type ParamValue = number | string | boolean | null | number[];

export interface ParamSchema {
  name: string;
  kind: "float" | "int" | "bool" | "enum" | "mixed" | "layers" | "priors" | "fixed";
  default: ParamValue;
  initial: ParamValue;
  label: Bi;
  hint: Bi;
  lo?: number;
  hi?: number;
  step?: number;
  log?: boolean;
  options?: ParamValue[];
  noneLabel?: Bi;
  section?: string;
  enabledIf?: Record<string, ParamValue[]>;
  visibleIf?: Record<string, ParamValue[]>;
  onlyTag?: Bi;
  disabled?: Bi;
}

export type LossKind = "loss" | "inertia" | "boost" | "trees" | "onestep" | "onestep_reg" | "knn" | "pca" | "depth";

export interface ModelSchema {
  name: string;
  family: string;
  group: "linear" | "trees" | "other" | "unsupervised";
  task: "classification" | "regression" | "clustering" | "reduction";
  lossKind: LossKind;
  iterative: boolean;
  stepUnit: "iteration" | "epoch" | "level" | "trees" | "stage" | "none";
  lockWhileTraining: boolean;
  expensive: boolean;
  explainFamily: string;
  main: ParamSchema[];
  advanced: ParamSchema[];
}

export interface GridData {
  bounds: [number, number, number, number];
  w: number;
  h: number;
  k: number;
  data: ND<Float32Array>;
}

export interface TreeNode {
  id: number;
  depth: number;
  leaf: boolean;
  feature: number | null;
  name: string | null;
  threshold: number | null;
  left: number | null;
  right: number | null;
  n: number;
  impurity: number;
  counts: number[];
  pred: number;
}

export interface TreeData { nodes: TreeNode[]; depth: number; leaves: number; node_count: number; truncated: boolean }

export interface NetworkData {
  sizes: number[];
  coefs: ND<Float32Array>[];
  intercepts: ND<Float32Array>[];
  activation: string;
  out_activation: string;
}

export interface ProbeData { row_id: number; layers: { act: number[]; pre?: number[] }[] }

export interface Viz {
  kind: "proba" | "value" | "curve" | "clusters" | "dbscan" | "projection";
  grid?: GridData;
  score?: boolean;
  line?: { w: [number, number]; b: number } | null;
  tree?: TreeData;
  network?: NetworkData;
  probe?: ProbeData | null;
  labels?: ND<Int32Array>;
  centers?: ND<Float32Array>;
  path?: ND<Float32Array>;
  support?: ND<Int32Array>;
  core?: ND<Uint8Array>;
  xy?: ND<Float32Array>;
  bounds?: [number, number, number, number];
  arrows?: { name: string; dx: number; dy: number }[];
  x?: ND<Float32Array>;
  y?: ND<Float32Array>;
  range?: [number, number];
  point_pred?: ND<Float32Array>;
}

export type FrameMsg = Omit<Frame, "viz"> & { viz: Viz; extra: Record<string, unknown> };
export type DatasetMsg = Omit<DatasetInfo, "ids" | "xy" | "label" | "test" | "plane"> & {
  ids: ND<Int32Array>;
  xy: ND<Float32Array>;
  label: ND<Float32Array>;
  test: ND<Uint8Array>;
  plane: { mode: "raw" | "pca" | "xy" | "line"; axes: [string, string]; bounds: [number, number, number, number]; editable: boolean; explained: [number, number] | null } | null;
  compat: Compat;
  table: TableData;
};
export interface Compat {
  ok: boolean;
  code: string;
  params: Record<string, unknown>;
  fixes: { kind: "switch_model" | "change_target" | "preset" | "edit_table"; model?: string; column?: string | null; preset?: string }[];
}
export interface TableColumn { name: string; type: "numeric" | "category"; catKind: "text" | "number" }
export interface TableData { columns: TableColumn[]; rows: unknown[][]; target: string | null }
export type ReadyMsg = Omit<Ready, "models"> & { models: ModelSchema[] };
export type StatusMsg = Status;
export type DoneMsg = Done;
export type ExplainMsg = ExplainSteps & { data: Record<string, unknown> };
export type PreviewMsg = Omit<Preview, "xy" | "label" | "extra" | "rows" | "columns" | "classes"> & {
  xy: ND<Float32Array>; label: ND<Float32Array>; extra: Record<string, unknown>; rows: unknown[][]; columns: string[]; classes: string[];
};
export interface PresetInfo { id: string; task: string; settings: Record<string, { min: number; max: number; step: number; default: number }> }
