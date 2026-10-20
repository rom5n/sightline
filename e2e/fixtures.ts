// Each Playwright worker talks to its own engine (one engine = one shared session by design, so two
// workers on one engine would see each other's model and data). Port = 8765 + worker slot; the
// UI reads it from ?enginePort= (src/app/tauri.ts). playwright.config.ts starts one engine per slot.
import { test as base, expect } from "@playwright/test";

export const ENGINE_BASE_PORT = 8765;

export const test = base.extend({
  page: async ({ page }, use, testInfo) => {
    const port = String(ENGINE_BASE_PORT + testInfo.parallelIndex);
    const goto = page.goto.bind(page);
    page.goto = (url, options) => {
      const u = new URL(url, "http://127.0.0.1");
      u.searchParams.set("enginePort", port);
      return goto(u.pathname + u.search + u.hash, options);
    };
    await use(page);
  },
});

export { expect };
