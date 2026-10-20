// Measures the numbers in the brief in the bundled Chromium (dev server + engine running):
//   npm run engine & npm run dev; node e2e/perf.mjs [out-dir]
import { launchChromium } from "./browser.mjs";
import { writeFileSync } from "node:fs";

const outDir = process.argv[2] ?? ".";
const browser = await launchChromium();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
await page.goto(process.env.APP_URL ?? "http://localhost:1420/");
const wait = (ms) => page.waitForTimeout(ms);
const results = {};
page.on("pageerror", (e) => console.log("page error:", e.message));

await page.keyboard.press("Control+Shift+P"); // timing overlay
// --- startup: first interactive paint, engine handshake
results.interactiveMs = await page.evaluate(() => window.__sightlineInteractive ?? null);
results.navStartToLoad = await page.evaluate(() => {
  const n = performance.getEntriesByType("navigation")[0];
  return n ? Math.round(n.loadEventEnd) : null;
});
await page.waitForFunction(() => !document.querySelector(".engine-pill"), null, { timeout: 30000 }).catch(() => undefined);

async function measure(seconds) {
  return page.evaluate(async (sec) => {
    const frames = [];
    let long = 0, longest = 0;
    const obs = new PerformanceObserver((l) => { for (const e of l.getEntries()) { long++; longest = Math.max(longest, e.duration); } });
    try { obs.observe({ type: "longtask" }); } catch { /* unsupported */ }
    const t0 = performance.now();
    await new Promise((res) => {
      const f = (t) => { frames.push(t); if (t - t0 < sec * 1000) requestAnimationFrame(f); else res(); };
      requestAnimationFrame(f);
    });
    obs.disconnect();
    const d = frames.slice(1).map((t, i) => t - frames[i]).sort((a, b) => a - b);
    const avg = d.reduce((a, b) => a + b, 0) / d.length;
    return { fps: Math.round(1000 / avg), p95FrameMs: +d[Math.floor(d.length * 0.95)].toFixed(1), worstFrameMs: +d[d.length - 1].toFixed(1), longTasks: long, longestTaskMs: Math.round(longest) };
  }, seconds);
}

const goHome = async () => { await page.evaluate(() => window.location.hash = ""); };
const openModel = async (m) => { await page.locator(`[data-testid="card-${m}"]`).click(); await wait(400); };

// --- 5,000 points + 160×100 heatmap while an MLP trains
await page.locator(".topnav button", { hasText: /Models|Модели/ }).first().click().catch(() => undefined);
await wait(300);
await openModel("MLPClassifier");
// load 5000 rows through the generator (2 numeric columns + circle rule)
await page.locator(".topnav .center button", { hasText: /Data|Данные/ }).click();
await wait(400);
await page.locator(".seg button", { hasText: /Generator|Генератор/ }).click();
await wait(400);
// drop the sample category column so the plane is the raw 2-D feature space (not a PCA projection)
await page.locator('button[aria-label*="color"]').first().click().catch(() => undefined);
await wait(300);
const setRows = async (n) => {
  await page.evaluate((v) => {
    const el = document.querySelector("#g-rows");
    const pos = Math.round(((Math.log(v) - Math.log(20)) / (Math.log(50000) - Math.log(20))) * 1000); // log slider 20…50,000
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(el, String(pos));
    el.dispatchEvent(new Event("input", { bubbles: true }));
  }, n);
  await wait(300);
};
await setRows(5000);
await page.locator(".seg button", { hasText: /Circle|Круг/ }).click();
await wait(300);
await page.locator('[data-testid="generate"]').click();
await wait(1500);
await page.locator('[data-testid="train-btn"]').click();
await wait(1500);
results.training5000 = await measure(5);
results.perfOverlay5000 = await page.evaluate(() => document.querySelector('[data-testid="perf"]')?.textContent ?? null);
if (outDir) await page.screenshot({ path: `${outDir}/perf-mlp-5000.png` });

// --- 50,000 instanced points: render while training, then while panning/zooming
await page.locator('[data-testid="train-btn"]').click().catch(() => undefined); // pause
await page.locator(".topnav .center button", { hasText: /Data|Данные/ }).click();
await wait(400);
await page.locator(".seg button", { hasText: /Generator|Генератор/ }).click();
await wait(400);
await setRows(50000);
await page.locator('[data-testid="generate"]').click();
await wait(3000);
results.points50000Rendered = await page.evaluate(() => document.querySelector('[data-testid="perf"]')?.textContent ?? null);
await page.locator('[data-testid="train-btn"]').click();
await wait(2000);
results.training50000 = await measure(5);
const box = await page.locator(".plot-canvas").first().boundingBox();
if (box) {
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
  await page.mouse.move(cx, cy);
  const m = measure(3);
  for (let i = 0; i < 12; i++) { await page.mouse.wheel(0, i < 6 ? -120 : 120); await wait(120); }
  await page.mouse.down(); await page.mouse.move(cx + 120, cy + 60, { steps: 20 }); await page.mouse.up();
  results.panZoom50000 = await m;
}
if (outDir) await page.screenshot({ path: `${outDir}/perf-50000.png` });

// --- parameter change -> first new frame (LogisticRegression)
await page.locator(".topnav .left button").first().click();
await wait(400);
await openModel("LogisticRegression");
await page.locator(".topnav .center button", { hasText: /Data|Данные/ }).click();
await wait(400);
await page.locator(".seg button", { hasText: /Presets|Наборы/ }).click();
await wait(500);
await page.locator('[data-testid="use-moons"], [data-testid="preset-moons"] button').first().click();
await wait(600);
await page.locator('[data-testid="train-btn"]').click();
await page.waitForSelector('[data-testid="train-btn"]:has-text("again"), [data-testid="train-btn"]:has-text("заново")', { timeout: 30000 });
const lat = [];
for (const v of ["0.1", "10", "0.5", "50", "1"]) {
  const chip = page.locator('[data-param="C"] input.val');
  await chip.click();
  await chip.fill(v);
  const t = await page.evaluate(() => performance.now());
  await chip.press("Enter");
  const ms = await page.waitForFunction((t0) => {
    const s = window.__sightlineLastFrameAt;
    return s && s > t0 ? s - t0 : false;
  }, t, { timeout: 5000 }).then((h) => h.jsonValue()).catch(() => null);
  lat.push(ms == null ? null : Math.round(ms));
  await wait(1500);
}
results.paramToFrameMs = lat;
results.idle = await measure(2);
console.log(JSON.stringify(results, null, 2));
writeFileSync(`${outDir}/perf.json`, JSON.stringify(results, null, 2));
await browser.close();
