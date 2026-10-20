// Verifies that every translation key used in src/ exists in both en.json and ru.json.
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const flat = (o, p = "", out = {}) => {
  for (const [k, v] of Object.entries(o)) {
    const key = p ? `${p}.${k}` : k;
    if (v && typeof v === "object") flat(v, key, out);
    else out[key] = v;
  }
  return out;
};
const en = flat(JSON.parse(readFileSync(join(root, "src/locales/en.json"), "utf8")));
const ru = flat(JSON.parse(readFileSync(join(root, "src/locales/ru.json"), "utf8")));
const has = (d, k) => k in d || `${k}_one` in d || `${k}_other` in d;

const files = [];
const walk = (d) => { for (const f of readdirSync(d)) { const p = join(d, f); if (statSync(p).isDirectory()) walk(p); else if (/\.(tsx?|mjs)$/.test(f)) files.push(p); } };
walk(join(root, "src"));
const used = new Set();
for (const f of files) {
  const s = readFileSync(f, "utf8");
  for (const m of s.matchAll(/\bt\(\s*"([a-zA-Z0-9_.]+)"/g)) used.add(m[1]);
  for (const m of s.matchAll(/\?\s*"([a-z][a-zA-Z0-9]*\.[a-zA-Z0-9_.]+)"\s*:\s*"([a-z][a-zA-Z0-9]*\.[a-zA-Z0-9_.]+)"/g)) { used.add(m[1]); used.add(m[2]); }
}
// dynamic keys
const models = ["LinearRegression", "LogisticRegression", "Ridge", "Lasso", "ElasticNet", "SGDClassifier", "DecisionTreeClassifier", "RandomForestClassifier", "ExtraTreesClassifier", "GradientBoostingClassifier", "HistGradientBoostingClassifier", "AdaBoostClassifier", "BaggingClassifier", "SVC", "KNeighborsClassifier", "GaussianNB", "MLPClassifier", "KMeans", "DBSCAN", "PCA"];
for (const m of models) { used.add(`models.${m}.desc`); used.add(`models.${m}.keywords`); }
for (const t of ["regression", "classification", "clustering", "reduction"]) { used.add(`task.${t}`); used.add(`task.${t}Long`); }
for (const f of ["linear", "trees", "svm", "neighbors", "bayes", "neural", "clustering", "reduction"]) used.add(`family.${f}`);
for (const g of ["linear", "trees", "other", "unsupervised"]) { used.add(`groups.${g}.title`); used.add(`groups.${g}.desc`); }
for (const p of ["moons", "blobs", "circles", "linear", "spirals", "xor"]) for (const k of ["name", "desc", "task"]) used.add(`presets.${p}.${k}`);
for (const m of ["accuracy", "r2", "silhouette", "variance"]) { used.add(`metric.${m}`); for (const s of ["Title", "About", "Tip", "After"]) used.add(`metric.${m}${s}`); }
for (const u of ["iteration", "epoch", "level", "trees", "stage", "none"]) { used.add(`unit.${u}`); used.add(`unit.${u}Plural`); used.add(`unit.${u}Nom`); used.add(`ctl.unit.${u}`); }
for (const c of ["needs_features", "missing_values", "too_few_rows", "needs_two_features", "no_target", "needs_category_target", "one_class", "too_many_classes", "class_too_small", "needs_numeric_target"]) { used.add(`compat.${c}.title`); used.add(`compat.${c}.body`); }
for (const k of ["switch_model", "change_target", "preset", "edit_table"]) { used.add(`fix.${k}.text`); used.add(`fix.${k}.btn`); }
for (const k of ["any", "reg_good", "reg_only", "class_only", "cluster_blobs", "dbscan_shapes", "cluster_shapes", "pca_any", "most_models", "linear_moons", "linear_circles", "linear_xor", "linear_spirals", "spirals_hard", "flexible_ok"]) used.add(`fitBadge.${k}`);
for (const k of ["solver", "optimizer", "stop", "repro", "other", "shape", "sampling", "cat"]) used.add(`adv.sec.${k}`);
for (const k of ["dark", "light", "system"]) used.add(`theme.${k}`);
for (const k of ["loss", "inertia", "boost", "trees", "depth", "onestep", "onestep_reg", "knn", "pca"]) used.add(`states.lossIdle.${k}`);
for (const k of ["boundary", "tree", "boundaryNet", "kmeans", "dbscan", "pca", "regLine", "regSurface"]) { used.add(`viz.${k}`); used.add(`viz.${k}Tip`); }
for (const k of ["loss", "inertia", "boost", "trees", "depth", "onestep", "onestepReg", "knn", "pca"]) { used.add(`loss.${k}`); used.add(`loss.${k}Tip`); }
for (const k of ["viz", "loss", "params", "metric"]) { used.add(`guide.${k}.title`); used.add(`guide.${k}.body`); }
const steps = { logistic: [1, 2, 3, 5], sigmoid: [4], sign: [4], sgd: [1, 2, 3, 5], multi: [1, 2, 3, 4], multiSgd: [3], reg: [1, 2, 3, 4], tree: [1, 2, 3, 4], km: [1, 2, 3, 4], mlp: [1, 2, 3, 4], forest: [1, 2, 3, 4], boost: [1, 2, 3, 4], boostMulti: [4], ada: [1, 2, 3], knn: [1, 2, 3, 4], svc: [1, 2, 3], svcMulti: [2, 3], nb: [1, 2, 3, 4], db: [1, 2, 3, 4, 5], pca: [1, 2, 3], cat: [1, 2, 3, 4], catMulti: [4], catReg: [4] };
for (const [fam, ns] of Object.entries(steps)) for (const n of ns) for (const k of ["title", "short", "detail"]) used.add(`decide.${fam}.${n}.${k}`);
for (const s of ["core", "border", "noise"]) { used.add(`decide.db.${s}`); used.add(`decide.db.${s}Text`); }

// step-family bases picked by a ternary, not keys themselves
for (const b of ["decide.boost", "decide.boostMulti", "decide.multi", "decide.multiSgd", "decide.cat", "decide.catMulti", "decide.catReg"]) used.delete(b);
let bad = 0;
for (const k of [...used].sort()) {
  const e = has(en, k), r = has(ru, k);
  if (!e || !r) { bad++; console.log(`missing ${!e ? "EN" : ""}${!e && !r ? "+" : ""}${!r ? "RU" : ""}: ${k}`); }
}
const enOnly = Object.keys(en).filter((k) => !(k in ru) && !/_(one|other)$/.test(k));
for (const k of enOnly) { bad++; console.log(`RU lacks: ${k}`); }
console.log(`${used.size} keys used, ${bad} problems`);
process.exit(bad ? 1 : 0);
