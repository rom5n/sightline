// Screenshot + store snapshot of the running app (CDP on 9333).  node e2e/cdp-shot.mjs out.png
import { chromium } from "@playwright/test";

const browser = await chromium.connectOverCDP("http://127.0.0.1:9333");
const page = browser.contexts()[0].pages()[0];
await page.screenshot({ path: process.argv[2] ?? "shot.png" });
console.log(await page.evaluate(() => ({
  status: document.querySelector('[data-testid="train-btn"]')?.textContent,
  perf: document.querySelector('[data-testid="perf"]')?.textContent,
})));
await browser.close().catch(() => undefined);
