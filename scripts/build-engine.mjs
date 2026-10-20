// Builds the PyInstaller one-folder engine into engine/dist/sightline-engine/ for the current OS.
// Uses engine/.venv when it exists, else $PYTHON or the python on PATH (CI).
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const engine = join(root, "engine");
const venv = join(engine, ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
const py = existsSync(venv) ? venv : process.env.PYTHON ?? (process.platform === "win32" ? "python" : "python3");
const r = spawnSync(py, ["-m", "PyInstaller", "--noconfirm", "--clean", "--distpath", "dist", "--workpath", "build", "sightline-engine.spec"],
  { stdio: "inherit", cwd: engine });
process.exit(r.status ?? 1);
