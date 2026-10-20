// Lists user-visible string literals in .tsx files that bypass i18n:
// JSX text, and aria-label / title / placeholder / alt attributes with literal text.
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";

const files = [];
const walk = (d) => { for (const f of readdirSync(d)) { const p = join(d, f); if (statSync(p).isDirectory()) walk(p); else if (p.endsWith(".tsx")) files.push(p); } };
walk("src");
// technical tokens that are the same in every language (parameter names, units, symbols)
const ok = /^([\s\d.,:;·×→←↑↓≤≥=+\-−–—/%()[\]{}|#*'"`!?]|x\d|y|p|k|n|PC\d|e|N\/A|σ|Σ|∑|ε|β|α|λ|μ|√|ƒ|log|exp|max|min|argmax|softmax|relu|tanh|sigmoid|gini|entropy|true|false|None|null|auto|R²|IQR|OOB|ID|CSV|KNN|SVC|MLP|PCA)+$/i;
let n = 0;
for (const f of files) {
  const src = readFileSync(f, "utf8");
  const lines = src.split("\n");
  lines.forEach((line, i) => {
    if (/^\s*(\/\/|\*|\/\*)/.test(line)) return;
    const hits = [];
    for (const m of line.matchAll(/>([^<>{}]*[A-Za-zА-Яа-яЁё]{2,}[^<>{}]*)</g)) if (!ok.test(m[1].trim())) hits.push(m[1].trim());
    for (const m of line.matchAll(/\b(aria-label|title|placeholder|alt)="([^"]*[A-Za-zА-Яа-яЁё]{2,}[^"]*)"/g)) hits.push(`${m[1]}="${m[2]}"`);
    for (const h of hits) { n++; console.log(`${f}:${i + 1}: ${h}`); }
  });
}
console.log(`${n} hard-coded strings`);
