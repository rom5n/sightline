// Visual check of chart lines at 125 % scaling on a production build, plus canvas opacity.
//   APP_URL=http://127.0.0.1:4173/ [MODEL=…] [PRESET=…] [THEME=dark|light] node e2e/lines-check.mjs <out-dir>
import { chromium } from "@playwright/test";

const out = process.argv[2] ?? ".";
const browser = await chromium.launch({ channel: "msedge" });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1.25 });
const model = process.env.MODEL ?? "LogisticRegression", preset = process.env.PRESET ?? "moons";
await page.addInitScript((t) => localStorage.setItem("sightline.settings", JSON.stringify({ theme: t, lang: "en", firstRunDone: true })), process.env.THEME ?? "dark");
page.on("pageerror", (e) => console.log("page error:", e.message));
const wait = (ms) => page.waitForTimeout(ms);
await page.goto(process.env.APP_URL ?? "http://127.0.0.1:4173/");
await wait(1500);
await page.locator(`[data-testid="card-${model}"]`).click();
await wait(500);
await page.locator(".topnav .center button").nth(1).click();
await wait(500);
await page.locator(`[data-testid="preset-${preset}"] button`).last().click();
await wait(800);
await page.locator('[data-testid="train-btn"]').click();
await wait(5000);
await page.screenshot({ path: `${out}/${model}-full.png` });
await page.locator(".plot-canvas").first().screenshot({ path: `${out}/${model}-plot.png` });
await page.locator(".chart-box").first().screenshot({ path: `${out}/${model}-loss.png` });
// every 2-D chart canvas must be opaque (copy / save image)
console.log(await page.evaluate(() => [...document.querySelectorAll("canvas")].map((c) => {
  const ctx = c.getContext("2d");
  if (!ctx) return `${c.className || "canvas"}: webgl`;
  const d = ctx.getImageData(1, 1, 1, 1).data;
  return `${c.className || c.parentElement?.className}: alpha ${d[3]}`;
})));
// the snapshot "Copy image" / "Save image as" takes of every canvas, WebGL included
console.log(await page.evaluate(async () => Promise.all([...document.querySelectorAll("canvas")].map(async (c) => {
  const img = new Image();
  img.src = c.toDataURL("image/png");
  await img.decode();
  const t = document.createElement("canvas");
  t.width = img.width; t.height = img.height;
  const x = t.getContext("2d");
  x.drawImage(img, 0, 0);
  const d = x.getImageData(0, 0, t.width, t.height).data;
  let min = 255;
  for (let i = 3; i < d.length; i += 4) min = Math.min(min, d[i]);
  return `${c.className || c.parentElement?.className}: min alpha ${min}`;
}))));
await browser.close();
