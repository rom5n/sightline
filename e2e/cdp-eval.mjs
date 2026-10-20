// Evaluates an expression in the Tauri main page over CDP: node e2e/cdp-eval.mjs "<js>"
import { chromium } from "@playwright/test";
const b = await chromium.connectOverCDP("http://127.0.0.1:9333");
const page = b.contexts()[0].pages().find((p) => !p.url().includes("popout"));
console.log(JSON.stringify(await page.evaluate(process.argv[2]), null, 1));
await b.close().catch(() => undefined);
