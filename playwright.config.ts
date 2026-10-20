import { defineConfig } from "@playwright/test";

// Runs against the Vite dev server + a real Python engine (no Tauri needed).
// Windows: the installed Microsoft Edge (no download). Elsewhere: Playwright's Chromium
// (`npx playwright install chromium`).
export default defineConfig({
  testDir: "e2e",
  testMatch: /.*\.spec\.ts/,
  timeout: 90_000,
  expect: { timeout: 15_000 },
  workers: 1,
  reporter: [["list"]],
  use: {
    channel: process.platform === "win32" ? "msedge" : undefined,
    baseURL: "http://127.0.0.1:1420",
    viewport: { width: 1440, height: 900 },
    trace: "retain-on-failure",
  },
  webServer: [
    { command: "node scripts/engine.mjs", port: 8765, reuseExistingServer: true, timeout: 60_000, env: { SIGHTLINE_PORT: "8765", SIGHTLINE_TOKEN: "dev" } },
    { command: "npx vite --port 1420 --strictPort", port: 1420, reuseExistingServer: true, timeout: 60_000 },
  ],
});
