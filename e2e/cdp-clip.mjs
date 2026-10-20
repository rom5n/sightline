// Screenshot regions of the running app at device resolution (CDP on 9333).
//   node e2e/cdp-clip.mjs <out-prefix> [selector ...]
import { chromium } from "@playwright/test";

const [prefix = "clip", ...sels] = process.argv.slice(2);
const browser = await chromium.connectOverCDP("http://127.0.0.1:9333");
const page = browser.contexts()[0].pages()[0];
console.log("dpr", await page.evaluate(() => devicePixelRatio));
for (const [i, s] of (sels.length ? sels : [".plot-canvas", ".lc canvas, .line-chart canvas"]).entries()) {
  const el = page.locator(s).first();
  if (await el.count()) await el.screenshot({ path: `${prefix}-${i}.png` });
  else console.log("missing", s);
}
await browser.close().catch(() => undefined);
