// Step texts for "How it decides", filled with the real numbers from explain().
import { fmt } from "../../ui/controls";

export interface Step { title: string; short: string; detail: string }
type T = (k: string, o?: Record<string, unknown>) => string;
type D = Record<string, unknown>;
interface Term { name: string; x: number; xm: number; w: number; contrib: number }

const f2 = (v: unknown) => fmt(v as number, 2);
const sumText = (terms: Term[], b: number) => [...terms.map((x) => f2(x.contrib)), f2(b)].join(" + ").replace(/\+ −/g, "− ");

function S(t: T, base: string, i: number, p: Record<string, unknown> = {}): Step {
  return { title: t(`${base}.${i}.title`, p), short: t(`${base}.${i}.short`, p), detail: t(`${base}.${i}.detail`, p) };
}

export function buildSteps(family: string, d: D, t: T, classes: string[], model: string): Step[] {
  const cls = (i: unknown) => classes[i as number] ?? String(i);
  const pred = d.pred as number;
  const truth = d.true as number | null | undefined;
  const verdict = truth == null ? "" : truth === pred ? t("decide.right", { cls: cls(truth) }) : t("decide.wrongIs", { cls: cls(truth) });
  const feats = (terms: Term[]) => terms.map((x) => `${x.name} = ${f2(x.x)}`).join(", ");
  switch (family) {
    case "logistic":
    case "sgd": {
      if (d.binary) {
        const terms = d.terms as Term[];
        const b = d.intercept as number;
        const z = d.z as number;
        const ws = terms.map((x, i) => `w${i + 1} = ${f2(x.w)}`).join(", ");
        const base = `decide.${family}`;
        const p = { feats: feats(terms), ws, sum: sumText(terms, b), z: f2(z), a: cls(0), b: cls(1), prob: f2(d.prob ?? 0), pct: Math.round(((d.prob as number) ?? 0) * 100),
          pick: cls(pred), verdict, cmp: z >= 0 ? "≥" : "<", pcmp: ((d.prob as number) ?? 0) >= 0.5 ? "≥" : "<", scaled: d.scaled ? t("decide.scaledNote") : "", loss: d.loss };
        const last = family === "sgd" && d.prob == null ? "sign" : "sigmoid";
        return [S(t, base, 1, p), S(t, base, 2, p), S(t, base, 3, p), S(t, `decide.${last}`, 4, p), S(t, base, 5, p)];
      }
      const per = d.per_class as { cls: number; z: number }[];
      const pr = (d.proba as number[] | undefined) ?? [];
      const zs = per.map((c) => `z(${cls(c.cls)}) = ${f2(c.z)}`).join(", ");
      const p = { zs, probs: pr.map((v, i) => `${cls(i)} ${Math.round(v * 100)}%`).join(", "), pick: cls(pred), verdict, k: classes.length, feats: feats(per[0] ? ((d.per_class as { terms: Term[] }[])[0].terms) : []) };
      return [S(t, "decide.multi", 1, p), S(t, "decide.multi", 2, p), S(t, family === "sgd" ? "decide.multiSgd" : "decide.multi", 3, p), S(t, "decide.multi", 4, p)];
    }
    case "linear_reg": {
      const terms = d.terms as Term[];
      const b = d.intercept as number;
      const p = { feats: feats(terms), ws: terms.map((x, i) => `w${i + 1} = ${f2(x.w)}`).join(", "), sum: sumText(terms, b), y: f2(d.prediction), truth: f2(d.true), err: f2((d.true as number) - (d.prediction as number)), target: d.target };
      return [S(t, "decide.reg", 1, p), S(t, "decide.reg", 2, p), S(t, "decide.reg", 3, p), S(t, "decide.reg", 4, p)];
    }
    case "tree": {
      const path = d.path as { name: string; threshold_raw: number; raw: number; left: boolean; n: number }[];
      const leaf = d.leaf as { counts: number[]; n: number; pred: number; proba: number[] };
      const out: Step[] = [S(t, "decide.tree", 1, { n: d.root_n, q: path[0] ? `${path[0].name} ≤ ${f2(path[0].threshold_raw)}` : t("decide.tree.noQ") })];
      path.forEach((s, i) => {
        const p = { q: `${s.name} ≤ ${f2(s.threshold_raw)}`, v: f2(s.raw), f: s.name, ans: t(s.left ? "tree.yes" : "tree.no"), dir: t(s.left ? "decide.left" : "decide.right_"), n: s.n, k: i + 1, leaf: i === path.length - 1 ? t("decide.tree.toLeaf") : "" };
        out.push(S(t, "decide.tree", s.left ? 2 : 3, p));
      });
      const k = Math.round(leaf.counts[leaf.pred] ?? 0);
      out.push(S(t, "decide.tree", 4, { k, n: Math.round(leaf.counts.reduce((a, b) => a + b, 0)), cls: cls(leaf.pred), pct: Math.round(leaf.proba[leaf.pred] * 100), verdict,
        mix: leaf.counts.map((c, i) => `${Math.round(c)} ${cls(i)}`).filter((s) => !s.startsWith("0 ")).join(" · ") }));
      return out;
    }
    case "kmeans": {
      const dist = d.distances as number[];
      const sorted = dist.map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0]);
      const p = { x: (d.x as number[]).map((v, i) => `${(d.features as string[])[i]} = ${f2(v)}`).join(", "), k: dist.length, best: (d.nearest as number) + 1, bd: f2(sorted[0][0]),
        second: sorted[1] ? sorted[1][1] + 1 : "", sd: sorted[1] ? f2(sorted[1][0]) : "", clear: sorted[1] && sorted[1][0] > sorted[0][0] * 1.5 ? t("decide.km.clear") : t("decide.km.close"), rounds: d.rounds };
      return [S(t, "decide.km", 1, p), S(t, "decide.km", 2, p), S(t, "decide.km", 3, p), S(t, "decide.km", 4, p)];
    }
    case "mlp": {
      const layers = d.layers as { act: number[]; pre?: number[] }[];
      const L = layers.length;
      const out: Step[] = [S(t, "decide.mlp", 1, { x: layers[0].act.map((v, i) => `${(d.features as string[])[i] ?? "x" + (i + 1)} = ${f2(v)}`).slice(0, 4).join(", ") })];
      for (let l = 1; l < L - 1; l++) {
        const a = layers[l].act;
        const fire = a.filter((v) => v > 1e-9).length;
        out.push(S(t, "decide.mlp", 2, { l, fire, n: a.length, act: d.activation, next: l === 1 ? t("decide.mlp.first") : t("decide.mlp.deeper") }));
      }
      const outL = layers[L - 1];
      const pr = d.proba as number[];
      const z = outL.pre ?? [];
      out.push(S(t, "decide.mlp", 3, { z: z.map((v) => f2(v)).join(", "), oa: d.out_activation, p: pr.map((v, i) => `${cls(i)} ${f2(v)}`).join(", ") }));
      out.push(S(t, "decide.mlp", 4, { pick: cls(pred), p: f2(pr[pred]), verdict }));
      return out;
    }
    case "forest": {
      const trees = d.trees as { proba: number[] }[];
      const avg = d.avg as number[];
      const votes = trees.filter((x) => x.proba.indexOf(Math.max(...x.proba)) === pred).length;
      const p = { n: d.n, pick: cls(pred), p: f2(avg[pred]), votes, verdict, model };
      return [S(t, "decide.forest", 1, p), S(t, "decide.forest", 2, p), S(t, "decide.forest", 3, p), S(t, "decide.forest", 4, p)];
    }
    case "boost": {
      const base = d.base as number[];
      const score = d.score as number[];
      const pr = d.proba as number[];
      const p = { n: d.n, base: base.map(f2).join(", "), score: score.map(f2).join(", "), p: f2(pr[pred]), pick: cls(pred), verdict, binary: d.binary ? 1 : 0 };
      return [S(t, "decide.boost", 1, p), S(t, "decide.boost", 2, p), S(t, "decide.boost", 3, p), S(t, d.binary ? "decide.boost" : "decide.boostMulti", 4, p)];
    }
    case "catboost": {
      type Lv = { feature: string; kind: string; border?: number; value: unknown; yes: boolean; ctr?: string };
      const trees = d.trees as { levels: Lv[]; leaf: number; leaves: number; value: number[] }[];
      const cols = d.columns as { name: string; value: unknown; cat: boolean }[];
      const cats = cols.filter((c) => c.cat);
      const reg = d.task === "regression";
      const ci = reg || d.binary ? 0 : pred;
      const score = (d.score as number[])[ci];
      const t0 = trees[0];
      const q = t0 ? t0.levels.map((l) => (l.kind === "num" ? `${l.feature} > ${f2(l.border)}` : t("decide.cat.ctrQ", { f: l.feature }))).join(" · ") : "";
      const p = {
        row: cols.map((c) => `${c.name} = ${c.cat ? `"${String(c.value)}"` : f2(c.value)}`).join(", "),
        cats: cats.length ? cats.map((c) => c.name).join(", ") : t("decide.cat.noCats"),
        depth: d.depth, leaves: t0?.leaves ?? 0, leaf: (t0?.leaf ?? 0) + 1, q, lv: f2(t0?.value[ci] ?? 0),
        n: d.n, lr: f2(d.learning_rate), score: f2(score), base: f2((d.base as number[])[ci]),
        pick: reg ? "" : cls(pred), p: reg ? "" : f2((d.proba as number[])[pred]), verdict: reg ? "" : verdict,
        y: reg ? f2(d.value) : "", truth: reg ? f2(d.true) : "", err: reg ? f2((d.true as number) - (d.value as number)) : "", target: d.target,
      };
      const last = reg ? "decide.catReg" : d.binary ? "decide.cat" : "decide.catMulti";
      return [S(t, "decide.cat", 1, p), S(t, "decide.cat", 2, p), S(t, "decide.cat", 3, p), S(t, last, 4, p)];
    }
    case "ada": {
      const share = d.share as number[];
      const p = { n: d.n, pick: cls(pred), share: share.map((v, i) => `${cls(i)} ${Math.round(v * 100)}%`).join(", "), verdict };
      return [S(t, "decide.ada", 1, p), S(t, "decide.ada", 2, p), S(t, "decide.ada", 3, p)];
    }
    case "knn": {
      const votes = d.votes as number[];
      const nb = d.neighbors as { cls: number }[];
      const p = { k: d.k, votes: votes.map((v, i) => (v ? `${model ? Math.round(v * 100) / 100 : v} ${cls(i)}` : "")).filter(Boolean).join(" · "), pick: cls(pred), p: f2((d.proba as number[])[pred]), verdict,
        weighted: d.weights === "distance" ? t("decide.knn.weighted") : "", self: d.is_train ? t("decide.knn.self") : "", same: nb.filter((x) => x.cls === pred).length };
      return [S(t, "decide.knn", 1, p), S(t, "decide.knn", 2, p), S(t, "decide.knn", 3, p), S(t, "decide.knn", 4, p)];
    }
    case "svc": {
      const dec = d.decision as number[];
      const p = { nsv: d.n_sv, n: d.n_train, kernel: d.kernel, sum: f2(d.sum), b: f2(d.intercept), dec: dec.map(f2).join(", "), pick: cls(pred), verdict, side: dec[0] >= 0 ? cls(1) : cls(0) };
      if (!d.binary) return [S(t, "decide.svc", 1, p), S(t, "decide.svcMulti", 2, p), S(t, "decide.svcMulti", 3, p)];
      return [S(t, "decide.svc", 1, p), S(t, "decide.svc", 2, p), S(t, "decide.svc", 3, p)];
    }
    case "nb": {
      const pri = d.priors as number[];
      const post = d.posterior as number[];
      const feats2 = d.features as string[];
      const x = d.x as number[];
      const p = { pri: pri.map((v, i) => `${cls(i)} ${f2(v)}`).join(" / "), x: feats2.map((f, i) => `${f} = ${f2(x[i])}`).slice(0, 4).join(", "), post: post.map((v, i) => `${cls(i)} ${Math.round(v * 100)}%`).join(", "), pick: cls(pred), verdict, d: feats2.length };
      return [S(t, "decide.nb", 1, p), S(t, "decide.nb", 2, p), S(t, "decide.nb", 3, p), S(t, "decide.nb", 4, p)];
    }
    case "dbscan": {
      const p = { eps: f2(d.eps), ms: d.min_samples, count: d.count, status: t(`decide.db.${d.status}`), label: (d.label as number) >= 0 ? t("decide.db.cluster", { i: (d.label as number) + 1 }) : t("decide.db.noiseLbl") };
      return [S(t, "decide.db", 1, p), S(t, "decide.db", 2, p), S(t, "decide.db", 3, p), S(t, "decide.db", d.status === "noise" ? 5 : 4, p)];
    }
    case "pca": {
      const sc = d.scores as number[];
      const p = { pc1: f2(sc[0]), pc2: f2(sc[1] ?? 0), whiten: d.whiten ? t("decide.pca.whiten") : "" };
      return [S(t, "decide.pca", 1, p), S(t, "decide.pca", 2, p), S(t, "decide.pca", 3, p)];
    }
  }
  return [];
}
