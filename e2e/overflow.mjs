// In-page text overflow audit, shared by the screenshot tool and the e2e layout test.
// Finds text that is clipped by an ancestor, spills out of its box, or is cut by an ellipsis.
export function auditOverflow() {
  const out = [];
  const seen = new Set();
  const clipOf = (el) => {
    for (let e = el.parentElement; e && e !== document.body; e = e.parentElement) {
      const s = getComputedStyle(e);
      if (s.overflowX !== "visible" || s.overflowY !== "visible") return e;
    }
    return null;
  };
  const name = (el) => {
    const cls = typeof el.className === "string" ? el.className.trim().split(/\s+/).slice(0, 2).join(".") : "";
    return `${el.tagName.toLowerCase()}${cls ? "." + cls : ""}`;
  };
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    const txt = n.textContent.trim();
    if (!txt || txt.length < 2) continue;
    const el = n.parentElement;
    if (!el || seen.has(el)) continue;
    const st = getComputedStyle(el);
    if (st.visibility === "hidden" || st.display === "none" || +st.opacity === 0) continue;
    if (el.closest("[aria-hidden='true'], .dv-live-region, .perf, script, style, svg, [data-audit-skip]")) continue;
    const range = document.createRange();
    range.selectNodeContents(n);
    const r = range.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) continue;
    if (r.bottom < 0 || r.top > innerHeight || r.right < 0 || r.left > innerWidth) continue;
    seen.add(el);
    const issue = (kind) => out.push({ kind, el: name(el), text: txt.slice(0, 60) });
    // ellipsis that actually truncates: allowed in fixed slots when the full text is in a title
    if (st.textOverflow === "ellipsis" && el.scrollWidth > el.clientWidth + 1) { issue(el.closest("[title]") ? "ellipsis-titled" : "ellipsis"); continue; }
    // text wider than its own box with clipping
    if ((st.overflowX === "hidden" || st.overflowX === "clip") && el.scrollWidth > el.clientWidth + 1) { issue("clipped-self"); continue; }
    // text cut by a clipping ancestor (scrolling containers are fine)
    const c = clipOf(el);
    if (c) {
      const cs = getComputedStyle(c);
      const scrolls = /(auto|scroll)/.test(cs.overflowX + cs.overflowY);
      const cr = c.getBoundingClientRect();
      if (!scrolls && (r.right > cr.right + 1.5 || r.left < cr.left - 1.5 || r.bottom > cr.bottom + 1.5)) { issue("clipped-by:" + name(c)); continue; }
    }
    // text that spills out of its element box (overflow visible)
    const er = el.getBoundingClientRect();
    // (glyph boxes of large numerals overhang a tight line-height by design)
    const tol = Math.max(3, parseFloat(st.fontSize) * 0.3);
    if (st.display !== "inline" && (r.right > er.right + 2 || r.bottom > er.bottom + tol)) issue("spills");
    // off-screen horizontally
    if (r.right > innerWidth + 1) issue("offscreen");
  }
  return out;
}
