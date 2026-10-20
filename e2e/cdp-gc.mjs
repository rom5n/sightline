// Forces a full GC in the running app (CDP on 9333) and prints JS heap before/after.
import { chromium } from "@playwright/test";

const browser = await chromium.connectOverCDP("http://127.0.0.1:9333");
const page = browser.contexts()[0].pages()[0];
const cdp = await page.context().newCDPSession(page);
const heap = async () => (await cdp.send("Runtime.getHeapUsage")).usedSize / 1048576;
const before = await heap();
await cdp.send("HeapProfiler.collectGarbage");
const after = await heap();
const counters = await cdp.send("Memory.getDOMCounters").catch(() => null);
console.log(JSON.stringify({ heapBeforeMB: +before.toFixed(1), heapAfterMB: +after.toFixed(1), counters }));
await browser.close().catch(() => undefined);
