import { useTranslation } from "react-i18next";
import { useNav, type DataTab } from "../../app/nav";
import { openTextFile } from "../../app/tauri";
import { engine } from "../../engine/client";
import { toast, useSession } from "../../engine/store";
import type { TableData } from "../../engine/types";
import { Icon } from "../../ui/Icon";
import { Seg } from "../../ui/controls";
import { Generator } from "./Generator";
import { Presets } from "./Presets";
import { TableEditor, useTable } from "./TableEditor";

export async function importCsv() {
  const f = await openTextFile();
  if (!f) return;
  try {
    const r = await engine.request<{ table: TableData }>({ type: "parse_csv", csv: f.text });
    // mark it as current so the editor doesn't swap it for the dataset in use
    useTable.getState().load(r.table, f.name.replace(/\.[^.]+$/, ""), useSession.getState().dataset?.version ?? -1);
    useNav.getState().go("data", { dataTab: "table" });
  } catch (e) {
    const err = e as { message?: { en: string; ru: string } };
    toast("bad_csv", err.message ?? { en: "Couldn't read this file.", ru: "Не удалось прочитать файл." });
  }
}

export function DataScreen() {
  const { t } = useTranslation();
  const tab = useNav((s) => s.dataTab);
  const go = useNav((s) => s.go);
  return (
    <div className="data" data-testid="data-screen">
      <div className="data-head">
        <div style={{ display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
          <h1>{t("data.title")}</h1>
          <Seg ariaLabel={t("data.source")} value={tab} onChange={(v: DataTab) => go("data", { dataTab: v })}
            options={[{ value: "presets", label: t("data.presets") }, { value: "table", label: t("data.table") }, { value: "generator", label: t("data.generator") }]} />
          {tab === "presets" && <span className="hint" style={{ fontSize: 13 }}>{t("data.presetsHint")}</span>}
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn sm" onClick={() => void importCsv()}><Icon name="upload" />{t("data.importCsv")}</button>
          {tab === "table" && <TableUseButton />}
        </div>
      </div>
      {tab === "presets" && <Presets />}
      {tab === "table" && <TableEditor />}
      {tab === "generator" && <Generator />}
    </div>
  );
}

function TableUseButton() {
  const { t } = useTranslation();
  const use = useTable((s) => s.use);
  return <button className="btn sm pri" onClick={() => void use()} data-testid="use-table">{t("data.useTable")}</button>;
}

/** After choosing data: back to the workspace, or Home when no model is picked yet. */
export function afterDataChosen() {
  const nav = useNav.getState();
  if (nav.standaloneData) nav.go("home");
  else nav.go("workspace");
}
