import "@fontsource/onest/400.css";
import "@fontsource/onest/500.css";
import "@fontsource/onest/600.css";
import "@fontsource/onest/700.css";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/500.css";
import "@fontsource/jetbrains-mono/600.css";
import "@fontsource/jetbrains-mono/700.css";
import "dockview-react/dist/styles/dockview.css";
import "./styles/tokens.css";
import "./styles/shell.css";
import "./styles/home.css";
import "./styles/workspace.css";
import "./styles/data.css";
import "./styles/decide.css";
import "./i18n";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App, PopoutApp } from "./app/App";
import { initEngine } from "./engine/store";
import { hydrateSettings, useSettings } from "./app/settings";
import { startMirrorClient, startMirrorHost } from "./app/mirror";
import { applyTheme } from "./app/Shell";
import { setLang } from "./i18n";
import { isTauri, invoke } from "./app/tauri";

const pop = /popout=([^&]+)/.exec(location.hash)?.[1];

if (isTauri) {
  // Mica / Acrylic on Windows 11 (the Rust side reports which material it applied)
  void invoke<string>("window_material").then((m) => { if (m === "mica" || m === "acrylic") document.documentElement.dataset.material = "mica"; }).catch(() => undefined);
}

if (pop) {
  startMirrorClient(decodeURIComponent(pop), (theme, lang) => {
    useSettings.setState({ theme: theme as never, lang: lang as never });
    applyTheme(theme as never);
    setLang(lang as never);
  });
} else {
  initEngine();
  startMirrorHost();
  void hydrateSettings().then(() => {
    const s = useSettings.getState();
    applyTheme(s.theme, false);
    setLang(s.lang);
  });
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>{pop ? <PopoutApp panel={decodeURIComponent(pop)} /> : <App />}</StrictMode>,
);

// first paint is interactive; record it for the perf report
requestAnimationFrame(() => requestAnimationFrame(() => {
  (window as unknown as { __sightlineInteractive?: number }).__sightlineInteractive = performance.now();
}));
