import { defineConfig } from "@playwright/test";
import { CHROMIUM_ARGS } from "./e2e/browser.mjs";
import { ENGINE_BASE_PORT } from "./e2e/fixtures";

const WORKERS = 2;

// Runs against the Vite dev server + a real Python engine (no Tauri needed).
// Browser: Playwright's bundled Chromium only (`npx playwright install chromium`), a fresh
// temporary context per test, no password manager or Windows sign-in — see e2e/browser.mjs.
export default defineConfig({
  testDir: "e2e",
  testMatch: /.*\.spec\.ts/,
  timeout: 90_000,
  expect: { timeout: 15_000 },
  workers: WORKERS,
  reporter: [["list"]],
  use: {
    browserName: "chromium",
    launchOptions: { args: CHROMIUM_ARGS },
    baseURL: "http://127.0.0.1:1420",
    viewport: { width: 1440, height: 900 },
    trace: "retain-on-failure",
  },
  webServer: [
    // one engine per worker (see e2e/fixtures.ts)
    ...Array.from({ length: WORKERS }, (_, i) => {
      const port = ENGINE_BASE_PORT + i;
      return { command: "node scripts/engine.mjs", port, reuseExistingServer: true, timeout: 60_000, env: { SIGHTLINE_PORT: String(port), SIGHTLINE_TOKEN: "dev" } };
    }),
    { command: "npx vite --port 1420 --strictPort", port: 1420, reuseExistingServer: true, timeout: 60_000 },
  ],
});
