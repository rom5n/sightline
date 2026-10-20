import { useMemo } from "react";
import { useSession } from "../../engine/store";
import { Scatter } from "../../charts/Scatter";

/** 44×36 thumbnail of the current dataset for the params aside. */
export function DatasetThumb() {
  const ds = useSession((s) => s.dataset);
  const xy = useMemo(() => (ds ? (ds.xy.data.length > 1200 ? ds.xy.data.subarray(0, 1200) : ds.xy.data) : null), [ds]);
  return (
    <div style={{ width: 44, height: 36, borderRadius: 8, background: "var(--solid)", border: "1px solid var(--line)", padding: 3, flex: "none" }} aria-hidden="true">
      <Scatter xy={xy} labels={ds?.label.data ?? null} single={ds?.task !== "classification"} size={2.4} />
    </div>
  );
}
