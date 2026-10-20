// Records the README's live-training GIF: MLPClassifier learning "Circles", then converts it with ffmpeg.
// usage (dev server + engine running, ffmpeg on PATH): node e2e/gif.mjs [out.gif]
import { chromium } from "@playwright/test";
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const out = process.argv[2] ?? "docs/screenshots/training.gif";
const dir = mkdtempSync(join(tmpdir(), "sightline-gif-"));
const browser = await chromium.launch({ channel: process.platform === "win32" ? "msedge" : undefined });
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
await page.addInitScript(() => localStorage.setItem("sightline.settings", JSON.stringify({ theme: "dark", lang: "en", firstRunDone: true })));
await page.goto(process.env.APP_URL ?? "http://localhost:1420/");
await page.locator('[data-testid="card-MLPClassifier"]').click();
await page.locator(".topnav .center button").nth(1).click();
await page.locator('[data-testid="preset-circles"] button').last().click();
await page.locator(".topnav .center button").nth(0).click();
await page.waitForTimeout(1500);
// screenshots in a loop (no video codec needed); the real frame rate goes to ffmpeg
// ~15 fps for 9 s; "Train" is pressed after the first half second
const t0 = Date.now();
let n = 0;
for (let pressed = false; Date.now() - t0 < 9000; n++) {
  if (!pressed && Date.now() - t0 > 500) { pressed = true; await page.locator('[data-testid="train-btn"]').click(); }
  const next = t0 + (n + 1) * 66;
  await page.screenshot({ path: join(dir, `f${String(n).padStart(4, "0")}.jpg`), type: "jpeg", quality: 92 });
  await page.waitForTimeout(Math.max(0, next - Date.now()));
}
const fps = (n / ((Date.now() - t0) / 1000)).toFixed(2);
await browser.close();
execFileSync("ffmpeg", ["-y", "-v", "error", "-framerate", fps, "-i", join(dir, "f%04d.jpg"), "-vf",
  "scale=880:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
  "-loop", "0", out]);
rmSync(dir, { recursive: true, force: true });
console.log("wrote", out, "at", fps, "fps");
