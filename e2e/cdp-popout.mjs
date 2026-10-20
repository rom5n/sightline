// Pops the Loss panel out into its own Tauri window and screenshots both windows.
import { chromium } from "@playwright/test";
const out = process.argv[2] ?? ".";
const b = await chromium.connectOverCDP("http://127.0.0.1:9333");
const ctx = b.contexts()[0];
const main = ctx.pages().find((p) => !p.url().includes("popout"));
await main.reload();
await main.waitForTimeout(2500);
await main.locator('[data-testid="card-LogisticRegression"]').click();
await main.waitForTimeout(500);
await main.locator(".topnav .center button").nth(1).click();
await main.waitForTimeout(800);
await main.locator('[data-testid="preset-moons"] button').last().click();
await main.waitForTimeout(800);
await main.locator('[data-testid="train-btn"]').click();
await main.waitForTimeout(1500);
const waitPop = new Promise((res) => ctx.on("page", res));
await main.locator('.dv-groupview[data-kind="loss"] .pbtns button:nth-child(2)').click();
const pop = await Promise.race([waitPop, new Promise((r) => setTimeout(() => r(null), 8000))]);
console.log("popout page:", pop ? pop.url() : "none");
if (pop) {
  await pop.waitForTimeout(3000);
  await pop.screenshot({ path: `${out}/popout-loss.png` });
  console.log("main has loss panel:", await main.locator('.dv-groupview[data-kind="loss"]').count());
  await pop.close();
  await main.waitForTimeout(1500);
  console.log("after close, main has loss panel:", await main.locator('.dv-groupview[data-kind="loss"]').count());
}
await main.screenshot({ path: `${out}/popout-main.png` });
await b.close().catch(() => undefined);
