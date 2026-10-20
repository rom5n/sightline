// Runs the engine's Python (engine/.venv) with the given arguments.
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const venv = join(root, "engine", ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
const py = existsSync(venv) ? venv : process.env.PYTHON ?? (process.platform === "win32" ? "python" : "python3");
const r = spawnSync(py, process.argv.slice(2), { stdio: "inherit", cwd: root, env: { ...process.env, PYTHONPATH: join(root, "engine") } });
process.exit(r.status ?? 1);
