// Spreadsheet-style table editor: typed columns, one target, inline validation,
// missing-value fixes, undo/redo, CSV import/export, keyboard navigation.
import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { create } from "zustand";
import { saveTextFile } from "../../app/tauri";
import { engine } from "../../engine/client";
import { toast, useSession } from "../../engine/store";
import type { TableColumn, TableData } from "../../engine/types";
import { Icon } from "../../ui/Icon";
import { Seg, Toggle } from "../../ui/controls";
import { afterDataChosen } from "./DataScreen";

type Cell = string | number | null;
interface Row { id: number; v: Cell[] }
interface Snap { cols: TableColumn[]; rows: Row[]; target: string | null }

interface TableStore extends Snap {
  name: string;
  sel: number; // selected column
  cur: [number, number]; // row index, col index
  undo: Snap[];
  redo: Snap[];
  loadedFrom: number; // dataset version it was copied from
  load: (t: TableData, name: string, version?: number) => void;
  apply: (f: (s: Snap) => Snap) => void;
  doUndo: () => void;
  doRedo: () => void;
  use: () => Promise<void>;
}

const SAMPLE: TableData = {
  columns: [
    { name: "weight_g", type: "numeric", catKind: "text" },
    { name: "diameter_cm", type: "numeric", catKind: "text" },
    { name: "color", type: "category", catKind: "text" },
    { name: "type", type: "category", catKind: "text" },
  ],
  rows: [[150, 7.1, "red", "apple"], [170, 7.4, "green", "apple"], [120, 6.2, "yellow", "lemon"], [180, 6.9, "green", "pear"], [160, 7.6, "red", "apple"],
    [110, 5.9, "yellow", "lemon"], [195, 7.0, "green", "pear"], [140, 7.3, "red", "apple"], [null, 6.1, "yellow", "lemon"], [175, 6.8, "yellow", "pear"],
    [165, 7.5, "red", "apple"], [125, 6.0, "yellow", "lemon"]],
  target: "type",
};

function toRows(t: TableData): Row[] {
  const n = t.columns.length;
  return t.rows.map((r, i) => (r.length === n + 1 ? { id: Number(r[0]), v: r.slice(1) as Cell[] } : { id: i, v: r as Cell[] }));
}

export const useTable = create<TableStore>((set, get) => ({
  cols: SAMPLE.columns,
  rows: toRows(SAMPLE),
  target: SAMPLE.target,
  name: "fruit",
  sel: 2,
  cur: [0, 0],
  undo: [],
  redo: [],
  loadedFrom: -1,
  load: (t, name, version = -1) => set({ cols: t.columns, rows: toRows(t), target: t.target, name, sel: Math.max(0, t.columns.length - 1), cur: [0, 0], undo: [], redo: [], loadedFrom: version }),
  apply: (f) => {
    const s = get();
    const snap: Snap = { cols: s.cols, rows: s.rows, target: s.target };
    const next = f(snap);
    set({ ...next, undo: [...s.undo.slice(-99), snap], redo: [] });
  },
  doUndo: () => {
    const s = get();
    const prev = s.undo[s.undo.length - 1];
    if (!prev) return;
    set({ ...prev, undo: s.undo.slice(0, -1), redo: [...s.redo, { cols: s.cols, rows: s.rows, target: s.target }] });
  },
  doRedo: () => {
    const s = get();
    const nx = s.redo[s.redo.length - 1];
    if (!nx) return;
    set({ ...nx, redo: s.redo.slice(0, -1), undo: [...s.undo, { cols: s.cols, rows: s.rows, target: s.target }] });
  },
  use: async () => {
    const s = get();
    const table = { columns: s.cols, rows: s.rows.map((r) => [r.id, ...r.v.map((c, j) => coerce(c, s.cols[j]))]), target: s.target };
    try {
      await engine.request({ type: "load_dataset", spec: { kind: "table", table, name: s.name } });
      afterDataChosen();
    } catch (e) {
      const err = e as { message?: { en: string; ru: string } };
      toast("table", err.message ?? { en: "Couldn't use this table.", ru: "Не удалось использовать таблицу." });
    }
  },
}));

function isMissing(c: Cell) {
  return c === null || c === "" || (typeof c === "number" && Number.isNaN(c));
}
function coerce(c: Cell, col: TableColumn): Cell {
  if (isMissing(c)) return null;
  if (col.type === "numeric") {
    const n = typeof c === "number" ? c : Number(String(c).replace(",", ".").replace("−", "-"));
    return Number.isFinite(n) ? n : null;
  }
  if (col.catKind === "number") {
    const n = Number(c);
    return Number.isFinite(n) ? n : String(c);
  }
  return String(c);
}
function invalid(c: Cell, col: TableColumn) {
  if (isMissing(c)) return false;
  if (col.type === "numeric") return !Number.isFinite(typeof c === "number" ? c : Number(String(c).replace(",", ".").replace("−", "-")));
  return false;
}

function toCsv(s: Snap): string {
  const esc = (v: Cell) => {
    if (v === null) return "";
    const t = String(v);
    return /[",\n]/.test(t) ? `"${t.replace(/"/g, '""')}"` : t;
  };
  return [s.cols.map((c) => esc(c.name)).join(","), ...s.rows.map((r) => r.v.map(esc).join(","))].join("\n");
}

export function TableEditor() {
  const { t } = useTranslation();
  const st = useTable();
  const ds = useSession((s) => s.dataset);
  const [editing, setEditing] = useState<[number, number] | null>(null);
  // render only the rows in view (tables can have tens of thousands of rows)
  const scroller = useRef<HTMLDivElement>(null);
  const [win, setWin] = useState({ top: 0, h: 800 });
  const [draft, setDraft] = useState("");
  const gridRef = useRef<HTMLDivElement>(null);
  // start from the dataset in use (once per dataset version) unless the user already has a table
  useEffect(() => {
    if (ds && st.loadedFrom !== ds.version && ds.kind !== "preset" && ds.table.columns.length) st.load(ds.table, ds.name, ds.version);
  }, [ds?.version]); // eslint-disable-line react-hooks/exhaustive-deps

  const { cols, rows, target, sel, cur } = st;
  const missing = useMemo(() => {
    const out: { col: number; rows: number[] }[] = [];
    cols.forEach((_, j) => {
      const r = rows.map((x, i) => (isMissing(x.v[j]) ? i : -1)).filter((i) => i >= 0);
      if (r.length) out.push({ col: j, rows: r });
    });
    return out;
  }, [cols, rows]);
  const nextId = () => rows.reduce((m, r) => Math.max(m, r.id), -1) + 1;

  const commitEdit = (ri: number, ci: number, text: string) => {
    const col = cols[ci];
    const val: Cell = text.trim() === "" ? null : col.type === "numeric" && Number.isFinite(Number(text.replace(",", ".").replace("−", "-"))) ? Number(text.replace(",", ".").replace("−", "-")) : text;
    st.apply((s) => ({ ...s, rows: s.rows.map((r, i) => (i === ri ? { ...r, v: r.v.map((c, j) => (j === ci ? val : c)) } : r)) }));
  };
  const startEdit = (ri: number, ci: number, initial?: string) => {
    setEditing([ri, ci]);
    const c = rows[ri]?.v[ci];
    setDraft(initial ?? (c === null ? "" : String(c)));
  };
  const move = (dr: number, dc: number) => {
    const r = Math.max(0, Math.min(rows.length - 1, cur[0] + dr));
    const c = Math.max(0, Math.min(cols.length - 1, cur[1] + dc));
    useTable.setState({ cur: [r, c], sel: c });
    const sc = scroller.current;
    if (sc) {
      const y = r * 38;
      if (y < sc.scrollTop) sc.scrollTop = y;
      else if (y + 38 + 60 > sc.scrollTop + sc.clientHeight) sc.scrollTop = y + 98 - sc.clientHeight;
    }
    requestAnimationFrame(() => gridRef.current?.querySelector<HTMLElement>(`[data-cell="${r}-${c}"]`)?.focus());
  };
  const onKey = (e: React.KeyboardEvent) => {
    if (editing) return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); e.shiftKey ? st.doRedo() : st.doUndo(); return; }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "y") { e.preventDefault(); st.doRedo(); return; }
    const k = e.key;
    if (k === "ArrowDown") move(1, 0);
    else if (k === "ArrowUp") move(-1, 0);
    else if (k === "ArrowRight" || (k === "Tab" && !e.shiftKey)) move(0, 1);
    else if (k === "ArrowLeft" || (k === "Tab" && e.shiftKey)) move(0, -1);
    else if (k === "Enter" || k === "F2") startEdit(cur[0], cur[1]);
    else if (k === "Delete" || k === "Backspace") commitEdit(cur[0], cur[1], "");
    else if (k.length === 1 && !e.ctrlKey && !e.metaKey) { startEdit(cur[0], cur[1], k); e.preventDefault(); return; }
    else return;
    e.preventDefault();
  };
  const fillMean = (ci: number) => st.apply((s) => {
    const nums = s.rows.map((r) => r.v[ci]).filter((c) => !isMissing(c)).map(Number).filter(Number.isFinite);
    const mean = nums.length ? Math.round((nums.reduce((a, b) => a + b, 0) / nums.length) * 1000) / 1000 : 0;
    const mode = (() => {
      const cnt = new Map<string, number>();
      for (const r of s.rows) if (!isMissing(r.v[ci])) cnt.set(String(r.v[ci]), (cnt.get(String(r.v[ci])) ?? 0) + 1);
      return [...cnt.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? null;
    })();
    const fill = s.cols[ci].type === "numeric" ? mean : mode;
    return { ...s, rows: s.rows.map((r) => (isMissing(r.v[ci]) ? { ...r, v: r.v.map((c, j) => (j === ci ? fill : c)) } : r)) };
  });
  const dropMissing = (ci: number) => st.apply((s) => ({ ...s, rows: s.rows.filter((r) => !isMissing(r.v[ci])) }));

  const col = cols[sel];
  const values = useMemo(() => {
    if (!col || col.type !== "category") return [];
    const cnt = new Map<string, number>();
    for (const r of rows) if (!isMissing(r.v[sel])) cnt.set(String(r.v[sel]), (cnt.get(String(r.v[sel])) ?? 0) + 1);
    return [...cnt.entries()];
  }, [col, rows, sel]);
  const tcol = cols.findIndex((c) => c.name === target);
  const tclasses = useMemo(() => {
    if (tcol < 0 || cols[tcol].type !== "category") return [];
    const cnt = new Map<string, number>();
    for (const r of rows) if (!isMissing(r.v[tcol])) cnt.set(String(r.v[tcol]), (cnt.get(String(r.v[tcol])) ?? 0) + 1);
    const nat = (a: string) => (Number.isFinite(Number(a)) ? [0, Number(a), ""] : [1, 0, a]) as [number, number, string];
    return [...cnt.entries()].sort((a, b) => { const x = nat(a[0]), y = nat(b[0]); return x[0] - y[0] || x[1] - y[1] || x[2].localeCompare(y[2]); });
  }, [tcol, rows, cols]);
  const clsIdx = useMemo(() => new Map(tclasses.map(([v], i) => [v, i])), [tclasses]);
  const tmpl = `44px repeat(${cols.length}, minmax(120px, 1fr))`;
  const [newVal, setNewVal] = useState("");

  return (
    <div className="tbl-wrap" data-testid="table-editor">
      <section className="solid sheet">
        <div className="sheet-bar">
          <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
            <button className="btn sm" onClick={() => st.apply((s) => {
              let n = s.cols.length + 1, name = `col${n}`;
              while (s.cols.some((c) => c.name === name)) name = `col${++n}`;
              return { ...s, cols: [...s.cols, { name, type: "numeric", catKind: "text" }], rows: s.rows.map((r) => ({ ...r, v: [...r.v, null] })) };
            })}><Icon name="plus" />{t("table.column")}</button>
            <button className="btn sm" onClick={() => st.apply((s) => ({ ...s, rows: [...s.rows, { id: nextId(), v: s.cols.map(() => null) }] }))}><Icon name="plus" />{t("table.row")}</button>
            <button className="btn sm gh" onClick={st.doUndo} disabled={!st.undo.length} aria-label={t("table.undo")}><Icon name="undo" />{t("table.undo")}</button>
            <button className="btn sm gh" onClick={st.doRedo} disabled={!st.redo.length} aria-label={t("table.redo")}><Icon name="redo" />{t("table.redo")}</button>
            <button className="btn sm gh" onClick={() => void saveTextFile(`${st.name || "table"}.csv`, toCsv(st))}><Icon name="download" />{t("table.export")}</button>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span className="hint">{t("table.size", { rows: rows.length, cols: cols.length })}</span>
            {target && <span className="tag" style={{ color: "var(--acc-t)", borderColor: "var(--acc-t)" }}><Icon name="target" size={13} />{t("table.targetIs", { name: target })}</span>}
          </div>
        </div>
        <div className="sheet-scroll" onKeyDown={onKey} ref={scroller}
          onScroll={(e) => setWin({ top: e.currentTarget.scrollTop, h: e.currentTarget.clientHeight })}>
          <div className="grid" role="grid" aria-label={st.name} ref={gridRef} style={{ gridTemplateColumns: tmpl }}>
            <div className="gh" role="columnheader"><span className="lbl">#</span></div>
            {cols.map((c, j) => (
              <div key={j} role="columnheader" tabIndex={0} className={`gh${sel === j ? " sel" : ""}${c.name === target ? " tgt" : ""}`} onClick={() => useTable.setState({ sel: j })}
                onKeyDown={(e) => e.key === "Enter" && useTable.setState({ sel: j })}>
                <div className="cn">{c.name}{c.name === target && <span className="tag" style={{ height: 20, background: "var(--acc)", color: "var(--acc-ink)", borderColor: "transparent", fontFamily: "var(--font)" }}>{t("table.target")}</span>}</div>
                <div className="ty"><b>{c.type === "numeric" ? "123" : "Aa"}</b>{c.type === "numeric" ? t("table.numeric") : `${t("table.category")} · ${c.catKind === "number" ? t("table.numbers") : t("table.text")}`}</div>
              </div>
            ))}
            {(() => {
              const ROW = 38;
              const first = Math.max(0, Math.floor(win.top / ROW) - 10);
              return <div style={{ gridColumn: "1 / -1", height: first * ROW }} aria-hidden="true" />;
            })()}
            {rows.slice(Math.max(0, Math.floor(win.top / 38) - 10), Math.min(rows.length, Math.ceil((win.top + win.h) / 38) + 10)).map((r, k) => {
              const i = Math.max(0, Math.floor(win.top / 38) - 10) + k;
              return (
              <RowCells key={r.id} r={r} i={i} cols={cols} target={target} clsIdx={clsIdx} sel={sel} cur={cur} editing={editing} draft={draft} setDraft={setDraft}
                onFocusCell={(c) => useTable.setState({ cur: [i, c], sel: c })} onStartEdit={(c) => startEdit(i, c)}
                onCommit={(c, txt, dir) => { commitEdit(i, c, txt); setEditing(null); if (dir) move(dir[0], dir[1]); }}
                onCancel={() => setEditing(null)}
                onDelete={() => st.apply((s) => ({ ...s, rows: s.rows.filter((_, k2) => k2 !== i) }))} />
              );
            })}
            <div style={{ gridColumn: "1 / -1", height: Math.max(0, rows.length - Math.min(rows.length, Math.ceil((win.top + win.h) / 38) + 10)) * 38 }} aria-hidden="true" />
          </div>
          <div style={{ padding: "4px 14px 4px 58px" }}>
            <button className="btn sm gh" style={{ paddingLeft: 0 }} onClick={() => st.apply((s) => ({ ...s, rows: [...s.rows, { id: nextId(), v: s.cols.map(() => null) }] }))}><Icon name="plus" />{t("table.addRow")}</button>
          </div>
        </div>
        {missing.length > 0 && (
          <div className="miss-bar" role="alert">
            <Icon name="warn" style={{ color: "var(--warn)" }} />
            <span style={{ fontSize: 13, color: "var(--ink)" }} dangerouslySetInnerHTML={{ __html: t("table.missing", { count: missing[0].rows.length, col: cols[missing[0].col].name, rows: missing[0].rows.slice(0, 3).map((x) => x + 1).join(", ") }) }} />
            <div style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
              <button className="btn sm" onClick={() => fillMean(missing[0].col)}>{cols[missing[0].col].type === "numeric" ? t("table.fillMean") : t("table.fillMode")}</button>
              <button className="btn sm gh" onClick={() => dropMissing(missing[0].col)}>{t("table.dropRow")}</button>
            </div>
          </div>
        )}
      </section>

      <aside className="glass aside">
        {col && (
          <>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", paddingBottom: 12 }}>
              <h2 style={{ fontSize: 15, fontWeight: 600 }}>{t("table.settings")}</h2><span className="tag mono">{col.name}</span>
            </div>
            <div className="sect" style={{ borderTop: 0, paddingTop: 0 }}>
              <label className="lbl" htmlFor="cname">{t("table.name")}</label>
              <ColName key={sel + col.name} value={col.name} taken={cols.map((c) => c.name)} onRename={(name) => st.apply((s) => ({
                ...s, target: s.target === col.name ? name : s.target, cols: s.cols.map((c, j) => (j === sel ? { ...c, name } : c)),
              }))} />
            </div>
            <div className="sect">
              <span className="lbl">{t("table.type")}</span>
              <Seg size="sm" value={col.type} ariaLabel={t("table.type")} style={{ alignSelf: "flex-start" }}
                onChange={(v) => st.apply((s) => ({ ...s, cols: s.cols.map((c, j) => (j === sel ? { ...c, type: v } : c)) }))}
                options={[{ value: "numeric", label: `123 ${t("table.numeric")}` }, { value: "category", label: `Aa ${t("table.category")}` }]} />
              <span className="hint">{t("table.typeHint")}</span>
            </div>
            {col.type === "category" && (
              <>
                <div className="sect">
                  <span className="lbl">{t("table.valuesAre")}</span>
                  <Seg size="sm" value={col.catKind} ariaLabel={t("table.valuesAre")} style={{ alignSelf: "flex-start" }}
                    onChange={(v) => st.apply((s) => ({ ...s, cols: s.cols.map((c, j) => (j === sel ? { ...c, catKind: v } : c)) }))}
                    options={[{ value: "text", label: t("table.text") }, { value: "number", label: t("table.numbers") }]} />
                  <span className="hint">{t("table.numbersHint")}</span>
                </div>
                <div className="sect">
                  <div style={{ display: "flex", justifyContent: "space-between" }}><span className="lbl">{t("table.allowed")}</span><span className="hint">{t("table.count")}</span></div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                    {values.map(([v, n]) => (
                      <span className="chipv" key={v}>{v} <span className="hint mono">{n}</span>
                        <button aria-label={t("table.removeValue", { v })} onClick={() => st.apply((s) => ({ ...s, rows: s.rows.map((r) => (String(r.v[sel]) === v ? { ...r, v: r.v.map((c, j) => (j === sel ? null : c)) } : r)) }))}>
                          <Icon name="close" size={12} />
                        </button>
                      </span>
                    ))}
                    <input className="fld" style={{ width: 90, height: 28 }} placeholder={t("table.value")} value={newVal} aria-label={t("table.addValue")}
                      onChange={(e) => setNewVal(e.target.value)}
                      onKeyDown={(e) => { if (e.key === "Enter" && newVal.trim()) { const v = newVal.trim(); st.apply((s) => ({ ...s, rows: [...s.rows, { id: nextId(), v: s.cols.map((_, j) => (j === sel ? v : null)) }] })); setNewVal(""); } }} />
                  </div>
                  <span className="hint">{t("table.oneHot")}</span>
                </div>
              </>
            )}
            <div className="sect">
              <label style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10 }}>
                <span style={{ fontSize: 13, fontWeight: 500 }}>{t("table.useTarget")}</span>
                <Toggle checked={target === col.name} label={t("table.useTarget")} onChange={(on) => st.apply((s) => ({ ...s, target: on ? col.name : null }))} />
              </label>
              <span className="hint" dangerouslySetInnerHTML={{ __html: target ? t("table.targetHint", { name: target }) : t("table.noTarget") }} />
            </div>
            <div className="sect">
              <span className="lbl">{t("table.readyFor")}</span>
              {!target ? <span style={{ fontSize: 13 }}>{t("table.readyUnsup")}</span>
                : cols[tcol]?.type === "category" ? (
                  <>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13 }}><Icon name="check" style={{ color: "var(--acc-t)" }} />{t("table.readyClass", { count: tclasses.length })}</div>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>{tclasses.map(([v, n], i) => <span className="tag" key={v}><span className={`dot k${i % 5} s${i % 4}`} />{v} {n}</span>)}</div>
                    <span className="hint">{t("table.readyClassHint")}</span>
                  </>
                ) : (
                  <>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13 }}><Icon name="check" style={{ color: "var(--acc-t)" }} />{t("table.readyReg")}</div>
                    <span className="hint">{t("table.readyRegHint")}</span>
                  </>
                )}
            </div>
            <button className="btn sm gh warn" style={{ alignSelf: "flex-start", marginTop: "auto" }} disabled={cols.length <= 1}
              onClick={() => st.apply((s) => ({ ...s, target: s.target === col.name ? null : s.target, cols: s.cols.filter((_, j) => j !== sel), rows: s.rows.map((r) => ({ ...r, v: r.v.filter((_, j) => j !== sel) })) }))}>
              {t("table.deleteColumn")}
            </button>
          </>
        )}
      </aside>
    </div>
  );
}

function ColName({ value, taken, onRename }: { value: string; taken: string[]; onRename: (n: string) => void }) {
  const [v, setV] = useState(value);
  const bad = v.trim() === "" || (v !== value && taken.includes(v.trim()));
  return <input id="cname" className={`fld wide${bad ? " bad" : ""}`} value={v} onChange={(e) => setV(e.target.value)}
    onBlur={() => { if (!bad && v.trim() !== value) onRename(v.trim()); else setV(value); }} onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()} />;
}

function RowCells({ r, i, cols, target, clsIdx, sel, cur, editing, draft, setDraft, onFocusCell, onStartEdit, onCommit, onCancel, onDelete }: {
  r: Row; i: number; cols: TableColumn[]; target: string | null; clsIdx: Map<string, number>; sel: number; cur: [number, number]; editing: [number, number] | null; draft: string;
  setDraft: (s: string) => void; onFocusCell: (c: number) => void; onStartEdit: (c: number) => void; onCommit: (c: number, txt: string, dir?: [number, number]) => void;
  onCancel: () => void; onDelete: () => void;
}) {
  const { t } = useTranslation();
  const tcls = target ? cols.findIndex((c) => c.name === target) : -1;
  return (
    <>
      <div className="gc n" role="rowheader" title={t("table.rowMenu")} onDoubleClick={onDelete}>
        <span>{i + 1}</span>
        <button className="btn gh ico sm" style={{ marginLeft: "auto", width: 22, height: 22, minHeight: 22, opacity: 0.5 }} aria-label={t("table.deleteRow", { n: i + 1 })} onClick={onDelete}><Icon name="close" size={12} /></button>
      </div>
      {r.v.map((c, j) => {
        const isEdit = editing && editing[0] === i && editing[1] === j;
        const miss = isMissing(c);
        const bad = invalid(c, cols[j]);
        const cls = `gc${sel === j ? " colsel" : ""}${cols[j].name === target ? " tgt" : ""}${miss ? " miss" : ""}${bad ? " bad" : ""}${cur[0] === i && cur[1] === j ? " cur" : ""}${isEdit ? " edit" : ""}`;
        if (isEdit) {
          return (
            <div key={j} className={cls} role="gridcell">
              <input autoFocus value={draft} aria-label={`${cols[j].name}, ${t("table.rowN", { n: i + 1 })}`} onChange={(e) => setDraft(e.target.value)}
                onBlur={() => onCommit(j, draft)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") { e.preventDefault(); onCommit(j, draft, [1, 0]); }
                  else if (e.key === "Tab") { e.preventDefault(); onCommit(j, draft, [0, e.shiftKey ? -1 : 1]); }
                  else if (e.key === "Escape") { e.preventDefault(); onCancel(); }
                  e.stopPropagation();
                }} />
            </div>
          );
        }
        return (
          <div key={j} className={cls} role="gridcell" tabIndex={cur[0] === i && cur[1] === j ? 0 : -1} data-cell={`${i}-${j}`}
            title={bad ? t("table.notNumber") : undefined} onClick={() => onFocusCell(j)} onDoubleClick={() => onStartEdit(j)} onFocus={() => onFocusCell(j)}>
            {j === tcls && !miss && cols[j].type === "category" && <span className={`dot k${(clsIdx.get(String(c)) ?? 0) % 5} s${(clsIdx.get(String(c)) ?? 0) % 4}`} style={{ marginRight: 8 }} />}
            {miss ? t("table.empty") : String(c)}
          </div>
        );
      })}
    </>
  );
}
