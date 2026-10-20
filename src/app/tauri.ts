// Thin wrappers so the same UI runs inside Tauri and in a plain browser (dev / Playwright).
export const isTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

async function core() {
  return import("@tauri-apps/api/core");
}

export async function invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  const c = await core();
  return c.invoke<T>(cmd, args);
}

export async function engineInfo(): Promise<{ port: number; token: string }> {
  if (isTauri) {
    for (let i = 0; i < 200; i++) {
      const info = await invoke<{ port: number; token: string } | null>("engine_info");
      if (info) return info;
      await new Promise((r) => setTimeout(r, 25));
    }
  }
  const q = new URLSearchParams(location.search);
  return { port: Number(q.get("enginePort") ?? import.meta.env.VITE_ENGINE_PORT ?? 8765), token: q.get("engineToken") ?? import.meta.env.VITE_ENGINE_TOKEN ?? "dev" };
}

export async function win() {
  const { getCurrentWindow } = await import("@tauri-apps/api/window");
  return getCurrentWindow();
}

export async function loadSettingsFile(): Promise<string | null> {
  if (!isTauri) return null;
  try {
    return await invoke<string | null>("load_settings");
  } catch {
    return null;
  }
}

export async function saveSettingsFile(json: string): Promise<void> {
  if (!isTauri) return;
  try {
    await invoke("save_settings", { json });
  } catch {
    /* settings are also mirrored to localStorage */
  }
}

/** Opens a CSV via the native dialog (Tauri) or an <input type=file> (browser). */
export async function openTextFile(): Promise<{ name: string; text: string } | null> {
  if (isTauri) {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const path = await open({ multiple: false, filters: [{ name: "CSV", extensions: ["csv", "txt", "tsv"] }] });
    if (!path || Array.isArray(path)) return null;
    const text = await invoke<string>("read_text_file", { path });
    return { name: String(path).split(/[\\/]/).pop() ?? "data.csv", text };
  }
  return new Promise((resolve) => {
    const inp = document.createElement("input");
    inp.type = "file";
    inp.accept = ".csv,.txt,.tsv";
    inp.onchange = async () => {
      const f = inp.files?.[0];
      resolve(f ? { name: f.name, text: await f.text() } : null);
    };
    inp.click();
  });
}

export async function saveTextFile(name: string, text: string): Promise<boolean> {
  if (isTauri) {
    const { save } = await import("@tauri-apps/plugin-dialog");
    const path = await save({ defaultPath: name, filters: [{ name: "CSV", extensions: ["csv"] }] });
    if (!path) return false;
    await invoke("write_text_file", { path, contents: text });
    return true;
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  return true;
}

export async function openPopout(label: string, panel: string, title: string, w = 900, h = 640): Promise<void> {
  const url = `${location.pathname}#popout=${encodeURIComponent(panel)}`;
  if (isTauri) {
    const { WebviewWindow } = await import("@tauri-apps/api/webviewWindow");
    const existing = await WebviewWindow.getByLabel(label);
    if (existing) {
      await existing.setFocus();
      return;
    }
    new WebviewWindow(label, { url, title, width: w, height: h, minWidth: 420, minHeight: 300, decorations: false, transparent: false });
    return;
  }
  window.open(url, label, `width=${w},height=${h}`);
}
