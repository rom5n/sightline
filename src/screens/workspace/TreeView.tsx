// Tree structure: nodes grow in level by level (scale + fade from the parent), edges draw
// themselves; pan with drag, zoom with the wheel, double-click to fit.
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import type { TreeData, TreeNode } from "../../engine/types";
import { fmt } from "../../ui/controls";

const NW = 150;
const NH = 66;
const GX = 18;
const GY = 46;

interface Laid { n: TreeNode; x: number; y: number; parent: Laid | null; side: "yes" | "no" | null }

export function layoutTree(tree: TreeData, collapsed: (n: TreeNode) => boolean = () => false): { nodes: Laid[]; w: number; h: number; depth: number } {
  const by = new Map(tree.nodes.map((n) => [n.id, n]));
  let maxD = 0;
  const out: Laid[] = [];
  let leafX = 0;
  const walk = (id: number, parent: Laid | null, side: Laid["side"]): Laid | null => {
    const n = by.get(id);
    if (!n) return null;
    const me: Laid = { n, x: 0, y: n.depth * (NH + GY), parent, side };
    out.push(me);
    maxD = Math.max(maxD, n.depth);
    if (n.leaf || n.left == null || n.right == null || !by.has(n.left) || collapsed(n)) {
      me.x = leafX;
      leafX += NW + GX;
      return me;
    }
    const l = walk(n.left, me, "yes");
    const r = walk(n.right, me, "no");
    me.x = l && r ? (l.x + r.x) / 2 : leafX;
    return me;
  };
  walk(0, null, null);
  const w = Math.max(NW, leafX - GX);
  const h = (maxD + 1) * (NH + GY) - GY;
  return { nodes: out, w, h, depth: maxD };
}

export function TreeView({ tree, classes, selected, onSelect, path, current, testId, criterion = "gini" }: {
  tree: TreeData; classes: string[]; selected?: number | null; onSelect?: (id: number | null) => void;
  path?: number[]; current?: number | null; testId?: string; criterion?: string;
}) {
  const { t } = useTranslation();
  const reduce = useReducedMotion();
  const box = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(1000);
  const [expanded, setExpanded] = useState<Set<number>>(new Set());
  // Show as many levels as fit at a readable size; deeper subtrees collapse into a "+N" card.
  const cutDepth = useMemo(() => {
    const byDepth = new Map<number, number>();
    for (const n of tree.nodes) if (n.leaf) byDepth.set(n.depth, (byDepth.get(n.depth) ?? 0) + 1);
    for (let d = tree.depth; d >= 2; d--) {
      const slots = tree.nodes.filter((n) => n.depth === d || (n.leaf && n.depth < d)).length;
      if (slots * (NW + GX) * 0.7 <= width) return d;
    }
    return 2;
  }, [tree, width]);
  const onPathSet = useMemo(() => new Set(path ?? []), [path]);
  const sizeOf = useMemo(() => {
    const by = new Map(tree.nodes.map((n) => [n.id, n]));
    const memo = new Map<number, number>();
    const count = (id: number): number => {
      if (memo.has(id)) return memo.get(id)!;
      const n = by.get(id);
      const c = !n || n.leaf || n.left == null || n.right == null ? 0 : 2 + count(n.left) + count(n.right);
      memo.set(id, c);
      return c;
    };
    return count;
  }, [tree]);
  const isCollapsed = (n: TreeNode) => !n.leaf && n.depth >= cutDepth && !expanded.has(n.id) && !(onPathSet.has(n.id) && onPathSet.size > 0);
  const L = useMemo(() => layoutTree(tree, isCollapsed), [tree, cutDepth, expanded, onPathSet]); // eslint-disable-line react-hooks/exhaustive-deps
  const [xf, setXf] = useState({ s: 1, x: 0, y: 0 });
  const fitted = useRef(false);
  const fit = () => {
    const r = box.current?.getBoundingClientRect();
    if (!r) return;
    if (Math.abs(r.width - width) > 40) setWidth(r.width);
    // fit, but never so small that the cards become unreadable; wide trees can be panned
    const s = Math.max(0.5, Math.min(1.2, (r.width - 8) / L.w, (r.height - 8) / L.h));
    const root = L.nodes[0];
    const x = L.w * s <= r.width ? (r.width - L.w * s) / 2 : r.width / 2 - (root ? (root.x + NW / 2) * s : 0);
    setXf({ s, x, y: Math.max(0, (r.height - L.h * s) / 2) * 0.3 });
  };
  useLayoutEffect(() => {
    if (!fitted.current || L.w > 0) fit();
    fitted.current = true;
    const ro = new ResizeObserver(() => fit());
    if (box.current) ro.observe(box.current);
    return () => ro.disconnect();
  }, [L.w, L.h]);
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    let drag: { x: number; y: number } | null = null;
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      const px = e.clientX - r.left, py = e.clientY - r.top;
      setXf((p) => {
        const s = Math.max(0.15, Math.min(3, p.s * Math.exp(-e.deltaY * 0.0016)));
        const k = s / p.s;
        return { s, x: px - (px - p.x) * k, y: py - (py - p.y) * k };
      });
    };
    const down = (e: PointerEvent) => {
      if ((e.target as HTMLElement).closest(".node")) return;
      drag = { x: e.clientX, y: e.clientY };
      el.setPointerCapture(e.pointerId);
    };
    const move = (e: PointerEvent) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag = { x: e.clientX, y: e.clientY };
      setXf((p) => ({ ...p, x: p.x + dx, y: p.y + dy }));
    };
    const up = () => (drag = null);
    el.addEventListener("wheel", wheel, { passive: false });
    el.addEventListener("pointerdown", down);
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
    el.addEventListener("dblclick", fit);
    return () => {
      el.removeEventListener("wheel", wheel);
      el.removeEventListener("pointerdown", down);
      el.removeEventListener("pointermove", move);
      el.removeEventListener("pointerup", up);
      el.removeEventListener("dblclick", fit);
    };
  }, [L]);

  // a light pulse travels along the edge into the current node (How it decides)
  const pulse = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = pulse.current;
    if (!el || current == null || reduce) return;
    const b = L.nodes.find((x) => x.n.id === current);
    const a = b?.parent;
    if (!a || !b) { el.style.opacity = "0"; return; }
    const x0 = a.x + NW / 2, y0 = a.y + NH, x1 = b.x + NW / 2, y1 = b.y, ym = (y0 + y1) / 2;
    const t0 = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const k = Math.min(1, (now - t0) / 650);
      const u = 1 - k;
      const x = u * u * u * x0 + 3 * u * u * k * x0 + 3 * u * k * k * x1 + k * k * k * x1;
      const y = u * u * u * y0 + 3 * u * u * k * ym + 3 * u * k * k * ym + k * k * k * y1;
      el.style.transform = `translate(${x - 6}px, ${y - 6}px)`;
      el.style.opacity = String(k < 0.85 ? 1 : (1 - k) / 0.15);
      if (k < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [current, L, reduce]);
  const onPath = new Set(path ?? []);
  const total = (n: TreeNode) => n.counts.reduce((a, b) => a + b, 0) || 1;
  return (
    <div ref={box} className="treev" data-testid={testId} role="tree" aria-label={t("tree.aria", { depth: tree.depth, leaves: tree.leaves })}>
      <div className="tv-inner" style={{ transform: `translate(${xf.x}px, ${xf.y}px) scale(${xf.s})`, width: L.w, height: L.h }}>
        <div ref={pulse} className="pulse-dot" style={{ opacity: 0, zIndex: 3 }} aria-hidden="true" />
        <svg className="tv-edges" width={L.w} height={L.h} aria-hidden="true">
          <AnimatePresence>
            {L.nodes.filter((d) => d.parent).map((d) => {
              const p = d.parent!;
              const x0 = p.x + NW / 2, y0 = p.y + NH, x1 = d.x + NW / 2, y1 = d.y;
              const ym = (y0 + y1) / 2;
              const on = onPath.has(d.n.id) && onPath.has(p.n.id);
              return (
                <motion.path key={`${p.n.id}-${d.n.id}`} className={on ? "on" : ""} d={`M${x0} ${y0}C${x0} ${ym} ${x1} ${ym} ${x1} ${y1}`}
                  initial={reduce ? { opacity: 0 } : { pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: 1 }} exit={{ opacity: 0 }}
                  transition={{ duration: 0.35, delay: reduce ? 0 : 0.05, ease: "easeOut" }} />
              );
            })}
          </AnimatePresence>
        </svg>
        <AnimatePresence>
          {L.nodes.filter((d) => d.parent).map((d) => {
            const p = d.parent!;
            const x = (p.x + NW / 2 + d.x + NW / 2) / 2, y = (p.y + NH + d.y) / 2;
            return <motion.span key={`yn-${d.n.id}`} className="yn" style={{ left: x, top: y }} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>{t(d.side === "yes" ? "tree.yes" : "tree.no")}</motion.span>;
          })}
        </AnimatePresence>
        <AnimatePresence>
          {L.nodes.map((d) => {
            const n = d.n;
            const tot = total(n);
            const cls = classes[n.pred] ?? String(n.pred);
            const isOn = onPath.has(n.id);
            const folded = isCollapsed(n);
            const state = path ? (current === n.id ? "cur" : isOn ? "on" : "dim") : selected === n.id ? "sel" : "";
            const fromX = d.parent ? d.parent.x - d.x : 0, fromY = d.parent ? d.parent.y - d.y : 0;
            return (
              <motion.div key={n.id} role="treeitem" aria-selected={selected === n.id} tabIndex={0}
                className={`node ${state}`} style={{ left: d.x, top: d.y, borderColor: n.leaf && !state ? `var(--c${n.pred % 5})` : undefined }}
                initial={reduce ? { opacity: 0 } : { opacity: 0, scale: 0.6, x: fromX * 0.6, y: fromY * 0.6 }}
                animate={{ opacity: state === "dim" ? 0.28 : 1, scale: 1, x: 0, y: 0 }} exit={{ opacity: 0, scale: 0.8 }}
                transition={{ type: "spring", stiffness: 300, damping: 30 }}
                onClick={() => (folded ? setExpanded((s) => new Set([...s, n.id])) : onSelect?.(selected === n.id ? null : n.id))}
                onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (folded ? setExpanded((s) => new Set([...s, n.id])) : onSelect?.(selected === n.id ? null : n.id))}
                title={folded ? t("tree.expand") : undefined}>
                {folded ? (
                  <span className="q" style={{ color: "var(--ink2)" }}>{t("tree.folded", { n: sizeOf(n.id) })}</span>
                ) : n.leaf ? (
                  <span className="q"><span className={`dot k${n.pred % 5} s${n.pred % 4}`} />{t("tree.leaf", { cls })}</span>
                ) : (
                  <span className="q">{n.name} ≤ {fmt(n.threshold, 2)}</span>
                )}
                {(() => {
                  const stat = n.leaf ? t("tree.leafStat", { n: n.n, k: Math.round(n.counts[n.pred] ?? 0), cls }) : t("tree.nodeStat", { n: n.n, crit: criterion, imp: fmt(n.impurity, 2) });
                  return <span className="m" title={stat}>{stat}</span>;
                })()}
                <div className="cbar">
                  {n.counts.map((c, i) => (c > 0 ? <div key={i} className={`k${i % 5}`} style={{ flex: c / tot }} /> : null))}
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
      {tree.truncated && <span className="hint" style={{ position: "absolute", left: 8, bottom: 4 }}>{t("tree.truncated", { n: tree.node_count })}</span>}
    </div>
  );
}

export const TREE_GEOM = { NW, NH };
