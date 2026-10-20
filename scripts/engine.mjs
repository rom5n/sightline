// Dev helper: run the engine on a fixed port for the browser / Playwright (no Tauri).
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const venv = join(root, "engine", ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
const port = process.env.SIGHTLINE_PORT ?? "8765";
const token = process.env.SIGHTLINE_TOKEN ?? "dev";
const child = spawn(existsSync(venv) ? venv : process.env.PYTHON ?? (process.platform === "win32" ? "python" : "python3"), ["-m", "sightline_engine", "--port", port, "--token", token],
  { stdio: "inherit", cwd: join(root, "engine") });
const stop = () => child.kill();
process.on("SIGINT", stop);
process.on("SIGTERM", stop);
child.on("exit", (c) => process.exit(c ?? 0));
