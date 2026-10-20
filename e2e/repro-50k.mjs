// Repro: MLP on 50,000 generated rows; prints train button / epoch text once a second.
import { chromium } from "@playwright/test";

const browser = await chromium.connectOverCDP("http://127.0.0.1:9333");
const page = browser.contexts()[0].pages()[0];
const wait = (ms) => page.waitForTimeout(ms);
await page.locator(".topnav .left button").first().click();
await wait(400);
await page.locator('[data-testid="card-MLPClassifier"]').click();
await wait(400);
if (process.env.PAUSE_FIRST) {
  // the perf script's sequence: train on the current data, pause, then replace the dataset
  await page.locator('[data-testid="train-btn"]').click();
  await wait(1500);
  await page.locator('[data-testid="train-btn"]').click();
  await wait(500);
  console.log("paused:", await page.locator('[data-testid="train-btn"]').textContent());
}
await page.locator(".topnav .center button", { hasText: /Data|Данные/ }).click();
await wait(400);
await page.locator(".seg button", { hasText: /Generator|Генератор/ }).click();
await wait(400);
await page.locator('button[aria-label*="color"]').first().click().catch(() => undefined);
await page.evaluate((v) => {
  const el = document.querySelector("#g-rows");
  const pos = Math.round(((Math.log(v) - Math.log(20)) / (Math.log(50000) - Math.log(20))) * 1000);
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(el, String(pos));
  el.dispatchEvent(new Event("input", { bubbles: true }));
}, 50000);
await wait(300);
await page.locator(".seg button", { hasText: /Circle|Круг/ }).click();
await page.locator('[data-testid="generate"]').click();
await wait(2500);
console.log("after generate:", await page.locator('[data-testid="train-btn"]').textContent());
await page.locator('[data-testid="train-btn"]').click();
const t0 = Date.now();
for (let i = 0; i < 12; i++) {
  await wait(1000);
  console.log(((Date.now() - t0) / 1000).toFixed(0) + "s", await page.evaluate(() => ({
    btn: document.querySelector('[data-testid="train-btn"]')?.textContent,
    ctl: document.querySelector(".ctl")?.textContent?.slice(0, 80),
  })));
}
await page.screenshot({ path: process.argv[2] ?? "repro.png" });
await browser.close().catch(() => undefined);
