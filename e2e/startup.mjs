// Cold-start timing of an installed build, with Python removed from PATH.
//   node e2e/startup.mjs "%LOCALAPPDATA%\Sightline\sightline.exe"
// Prints launch -> interactive Home and launch -> engine ready (wall clock).
import { chromium } from "@playwright/test";
import { spawn } from "node:child_process";

const exe = process.argv[2];
const PATH = (process.env.PATH ?? "").split(";").filter((p) => !/python|\\uv\\|\.venv|conda/i.test(p)).join(";");
const env = { ...process.env, PATH, WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS: "--remote-debugging-port=9333" };
const t0 = Date.now();
const child = spawn(exe, [], { env, detached: true, stdio: "ignore" });
child.unref();

let browser;
for (let i = 0; i < 300 && !browser; i++) {
  try { browser = await chromium.connectOverCDP("http://127.0.0.1:9333"); } catch { await new Promise((r) => setTimeout(r, 50)); }
}
const page = browser.contexts()[0].pages()[0];
await page.waitForFunction(() => window.__sightlineInteractive != null, null, { timeout: 30000 });
const interactiveAt = await page.evaluate(() => performance.timeOrigin + window.__sightlineInteractive);
await page.waitForFunction(() => !document.querySelector(".engine-pill"), null, { timeout: 60000 });
const engineAt = Date.now();
console.log(JSON.stringify({ launchToInteractiveMs: Math.round(interactiveAt - t0), launchToEngineReadyMs: engineAt - t0, pythonOnPath: false }));
await browser.close().catch(() => undefined);
