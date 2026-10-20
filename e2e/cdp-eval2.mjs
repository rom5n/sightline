import { chromium } from "@playwright/test";
const browser = await chromium.connectOverCDP("http://127.0.0.1:9333");
const page = browser.contexts()[0].pages()[0];
console.log(await page.evaluate(process.argv[2]));
await browser.close().catch(() => undefined);
