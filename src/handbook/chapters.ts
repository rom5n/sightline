// Handbook table of contents. Chapter texts live in en/<id>.md and ru/<id>.md, the runnable
// example in code/<id>.py (one file for both languages: "# @en" / "# @ru" comment lines).
import { CATALOG } from "../app/catalog";

export interface Chapter {
  id: string;
  group: "basics" | "linear" | "trees" | "other" | "unsupervised";
  /** model opened by "Try it in the app" (none for the intro) */
  model?: string;
  /** dataset preset loaded with it… */
  preset?: string;
  /** …or a generated table (CatBoost: a text column shows native categories) */
  generator?: Record<string, unknown>;
}

const GEN_WITH_CATEGORY = {
  rows: 500, mode: "rules", seed: 42,
  columns: [
    { name: "x1", type: "numeric", dist: "uniform", min: -3, max: 3 },
    { name: "x2", type: "numeric", dist: "normal", mean: 0, std: 1 },
    { name: "color", type: "category", values: [{ v: "red", w: 40 }, { v: "green", w: 35 }, { v: "blue", w: 25 }] },
  ],
  target: { name: "label", classes: ["A", "B"], source: "rule", rule: "circle", on: ["x1", "x2"], noise: 0.1 },
};

const TRY: Record<string, string> = {
  LinearRegression: "linear", LogisticRegression: "moons", Ridge: "linear", Lasso: "linear", ElasticNet: "linear",
  SGDClassifier: "blobs", DecisionTreeClassifier: "xor", RandomForestClassifier: "moons", ExtraTreesClassifier: "moons",
  GradientBoostingClassifier: "circles", HistGradientBoostingClassifier: "spirals", AdaBoostClassifier: "xor",
  BaggingClassifier: "moons", SVC: "circles", KNeighborsClassifier: "moons", GaussianNB: "blobs",
  MLPClassifier: "circles", KMeans: "blobs", DBSCAN: "moons", PCA: "blobs",
};

export const CHAPTERS: Chapter[] = [
  { id: "intro", group: "basics" },
  ...CATALOG.filter((m) => !m.name.startsWith("CatBoost")).flatMap((m): Chapter[] => {
    const ch: Chapter = { id: m.name, group: m.group, model: m.name, preset: TRY[m.name] };
    // CatBoost (classifier + regressor) shares one chapter, placed after the other ensembles
    return m.name === "BaggingClassifier" ? [ch, { id: "CatBoost", group: "trees", model: "CatBoostClassifier", generator: GEN_WITH_CATEGORY }] : [ch];
  }),
];

export const CHAPTER_BY_ID: Record<string, Chapter> = Object.fromEntries(CHAPTERS.map((c) => [c.id, c]));

/** The chapter that explains a model (both CatBoost models share one). */
export function chapterOf(model: string): string {
  return model.startsWith("CatBoost") ? "CatBoost" : CHAPTER_BY_ID[model] ? model : "intro";
}
