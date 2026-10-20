// Handbook rendering: Markdown (marked) with KaTeX math, heading anchors, and a small Python
// highlighter. Everything is bundled, so it works offline.
import katex from "katex";
import { marked } from "marked";

const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/**
 * `$$…$$` display math, `$…$` inline math, `## Title {#id}` headings with stable anchors,
 * `{{code}}` / `{{try}}` placeholders (left in the HTML for the component to fill).
 */
export function renderMarkdown(src: string): string {
  const keep: string[] = [];
  const hold = (html: string) => `%%K${keep.push(html) - 1}%%`;
  // code spans and fences first, so a "$" inside code is not math
  src = src.replace(/(```[\s\S]*?```|`[^`\n]+`)/g, (m) => hold(marked.parse(m, { async: false }).replace(/^<p>|<\/p>\n?$/g, "")));
  src = src.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex: string) => hold(katex.renderToString(tex.trim(), { displayMode: true, throwOnError: false, output: "html" })));
  src = src.replace(/\$([^\s$](?:[^$\n]*?[^\s$])?)\$/g, (_, tex: string) => hold(katex.renderToString(tex, { throwOnError: false, output: "html" })));
  // headings with an explicit anchor become their own block (blank lines around the placeholder)
  src = src.replace(/^(#{2,3}) (.+?) \{#([\w-]+)\}[ \t]*$/gm, (_, h: string, text: string, id: string) =>
    "\n" + hold(`<h${h.length} id="${id}">${marked.parseInline(text, { async: false })}</h${h.length}>`) + "\n");
  let html = marked.parse(src, { async: false, gfm: true });
  // a block-level placeholder alone in a paragraph replaces the paragraph itself
  html = html.replace(/<p>(%%K\d+%%)<\/p>/g, "$1");
  for (let i = 0; i < 3 && html.includes("%%K"); i++) html = html.replace(/%%K(\d+)%%/g, (_, n: string) => keep[Number(n)]);
  // tables scroll sideways instead of widening the page
  return html.replace(/<table>/g, '<div class="tscroll"><table>').replace(/<\/table>/g, "</table></div>");
}

/** Keep comment lines for one language: "# @en text" / "# @ru text" become "# text" or disappear. */
export function localiseCode(code: string, lang: "en" | "ru"): string {
  return code
    .split("\n")
    .filter((l) => { const m = l.match(/^\s*# @(en|ru)\b/); return !m || m[1] === lang; })
    .map((l) => l.replace(/# @(en|ru) ?/, "# "))
    .join("\n")
    .trim();
}

const KW = new Set("False None True and as assert break class continue def del elif else except finally for from if import in is lambda not or pass raise return try while with yield".split(" "));
const BUILTIN = new Set("print len range round zip enumerate sorted sum min max abs list dict float int str".split(" "));

/** Minimal Python highlighting: comments, strings, numbers, keywords, builtins, called names. */
export function highlightPython(code: string): string {
  const re = /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|f?"(?:\\.|[^"\\\n])*"|f?'(?:\\.|[^'\\\n])*')|\b(\d+(?:\.\d+)?(?:e[-+]?\d+)?)\b|\b([A-Za-z_]\w*)\b(\s*\()?/g;
  let out = "";
  let last = 0;
  for (const m of code.matchAll(re)) {
    out += esc(code.slice(last, m.index));
    last = m.index + m[0].length;
    if (m[1]) out += `<span class="c">${esc(m[1])}</span>`;
    else if (m[2]) out += `<span class="s">${esc(m[2])}</span>`;
    else if (m[3]) out += `<span class="n">${m[3]}</span>`;
    else if (m[4]) {
      const w = m[4];
      const cls = KW.has(w) ? "k" : BUILTIN.has(w) ? "b" : m[5] ? "f" : "";
      out += (cls ? `<span class="${cls}">${w}</span>` : w) + (m[5] ?? "");
    }
  }
  return out + esc(code.slice(last));
}
