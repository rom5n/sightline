// Drives the running app (CDP on 9333) to a typical state, so e2e/mem.ps1 can be read at each step.
//   node e2e/mem-flow.mjs idle | train | long
import { chromium } from "@playwright/test";

const what = process.argv[2] ?? "idle";
const browser = await chromium.connectOverCDP("http://127.0.0.1:9333");
const page = browser.contexts()[0].pages()[0];
await page.waitForFunction(() => !document.querySelector(".engine-pill"), null, { timeout: 60000 });
if (what === "train" || what === "long") {
  await page.locator(`[data-testid="card-${what === "long" ? "MLPClassifier" : "LogisticRegression"}"]`).click();
  await page.waitForTimeout(600);
  await page.locator(".topnav .center button", { hasText: /Data|Данные/ }).click();
  await page.waitForTimeout(400);
  await page.locator(".seg button", { hasText: /Presets|Наборы/ }).click();
  await page.waitForTimeout(400);
  await page.locator('[data-testid="use-moons"], [data-testid="preset-moons"] button').first().click();
  await page.waitForTimeout(800);
  await page.locator('[data-testid="train-btn"]').click();
  await page.waitForTimeout(4000);
  // long: wait for the run to finish (MLP, max_iter 200: hundreds of frames fill the scrubber ring)
  if (what === "long") await page.waitForSelector('[data-testid="train-btn"]:has-text("again"), [data-testid="train-btn"]:has-text("заново")', { timeout: 120000 });
}
await page.waitForTimeout(1500);
console.log(what, "ready; JS heap MB", await page.evaluate(() => Math.round((performance.memory?.usedJSHeapSize ?? 0) / 1048576)));
await browser.close().catch(() => undefined);
