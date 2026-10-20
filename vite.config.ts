import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Tauri expects a fixed dev port.
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: { port: 1420, strictPort: true, host: "127.0.0.1" },
  // msgpack is imported only by the Web Worker, which the dependency scanner does not crawl; without
  // this a cold dev server finds it late and reloads the page (seen as flaky first tests in CI)
  optimizeDeps: { include: ["@msgpack/msgpack"] },
  build: { target: "es2022", sourcemap: false, chunkSizeWarningLimit: 1500 },
  worker: { format: "es" },
});
