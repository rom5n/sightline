// Static model catalog: lets Home render instantly, before the engine is up.
export type Task = "regression" | "classification" | "clustering" | "reduction";
export type Preview = "line" | "coef" | "linb" | "tree" | "forest" | "boost" | "svm" | "knn" | "nb" | "net" | "cluster" | "dbscan" | "pca";

export interface CatalogEntry { name: string; group: "linear" | "trees" | "other" | "unsupervised"; task: Task; pv: Preview; family: string; metric: string }

const E = (name: string, group: CatalogEntry["group"], task: Task, pv: Preview, family: string, metric: string): CatalogEntry =>
  ({ name, group, task, pv, family, metric });

export const CATALOG: CatalogEntry[] = [
  E("LinearRegression", "linear", "regression", "line", "linear", "r2"),
  E("LogisticRegression", "linear", "classification", "linb", "linear", "accuracy"),
  E("Ridge", "linear", "regression", "line", "linear", "r2"),
  E("Lasso", "linear", "regression", "coef", "linear", "r2"),
  E("ElasticNet", "linear", "regression", "coef", "linear", "r2"),
  E("SGDClassifier", "linear", "classification", "linb", "linear", "accuracy"),
  E("DecisionTreeClassifier", "trees", "classification", "tree", "trees", "accuracy"),
  E("RandomForestClassifier", "trees", "classification", "forest", "trees", "accuracy"),
  E("ExtraTreesClassifier", "trees", "classification", "forest", "trees", "accuracy"),
  E("GradientBoostingClassifier", "trees", "classification", "boost", "trees", "accuracy"),
  E("HistGradientBoostingClassifier", "trees", "classification", "boost", "trees", "accuracy"),
  E("AdaBoostClassifier", "trees", "classification", "boost", "trees", "accuracy"),
  E("BaggingClassifier", "trees", "classification", "forest", "trees", "accuracy"),
  E("CatBoostClassifier", "trees", "classification", "boost", "trees", "accuracy"),
  E("CatBoostRegressor", "trees", "regression", "boost", "trees", "r2"),
  E("SVC", "other", "classification", "svm", "svm", "accuracy"),
  E("KNeighborsClassifier", "other", "classification", "knn", "neighbors", "accuracy"),
  E("GaussianNB", "other", "classification", "nb", "bayes", "accuracy"),
  E("MLPClassifier", "other", "classification", "net", "neural", "accuracy"),
  E("KMeans", "unsupervised", "clustering", "cluster", "clustering", "silhouette"),
  E("DBSCAN", "unsupervised", "clustering", "dbscan", "clustering", "silhouette"),
  E("PCA", "unsupervised", "reduction", "pca", "reduction", "variance"),
];

export const BY_NAME: Record<string, CatalogEntry> = Object.fromEntries(CATALOG.map((c) => [c.name, c]));
export const GROUPS = ["linear", "trees", "other", "unsupervised"] as const;

/** Which presets suit which model kind: good | ok | hard | wrong (wrong = incompatible task). */
export function presetFit(model: string | null, preset: string): { level: "good" | "ok" | "hard" | "wrong"; key: string } {
  if (!model) return { level: "good", key: "any" };
  const m = BY_NAME[model];
  if (!m) return { level: "good", key: "any" };
  const reg = preset === "linear";
  if (m.task === "regression") return reg ? { level: "good", key: "reg_good" } : { level: "wrong", key: "reg_only" };
  if (m.task === "classification" && reg) return { level: "wrong", key: "class_only" };
  if (m.task === "clustering") {
    if (preset === "blobs") return { level: "good", key: "cluster_blobs" };
    if (model === "DBSCAN" && (preset === "moons" || preset === "circles")) return { level: "good", key: "dbscan_shapes" };
    return { level: "ok", key: "cluster_shapes" };
  }
  if (m.task === "reduction") return { level: "ok", key: "pca_any" };
  const linear = ["LogisticRegression", "SGDClassifier", "GaussianNB"].includes(model);
  if (preset === "blobs") return { level: "good", key: "most_models" };
  if (linear) {
    if (preset === "moons") return { level: "ok", key: "linear_moons" };
    if (preset === "circles") return { level: "hard", key: "linear_circles" };
    if (preset === "xor") return { level: "hard", key: "linear_xor" };
    if (preset === "spirals") return { level: "hard", key: "linear_spirals" };
  }
  if (preset === "spirals" && ["DecisionTreeClassifier", "AdaBoostClassifier"].includes(model)) return { level: "ok", key: "spirals_hard" };
  return { level: "good", key: "flexible_ok" };
}

export const PRESET_ORDER = ["moons", "blobs", "circles", "linear", "spirals", "xor"] as const;
