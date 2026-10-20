// The app version lives in package.json. Tauri reads it from there (tauri.conf.json → "../package.json");
// the other manifests carry a copy that this script keeps in step.
//
//   node scripts/version.mjs          check every copy matches package.json (CI runs this)
//   node scripts/version.mjs 1.2.0    set the version everywhere
//   node scripts/version.mjs --tag v1.2.0   check, and that the tag matches (release workflow)
import { readFileSync, writeFileSync } from "node:fs";

const COPIES = [
  ["src-tauri/Cargo.toml", /^(version = ")([^"]+)(")/m],
  ["engine/pyproject.toml", /^(version = ")([^"]+)(")/m],
  ["engine/sightline_engine/__init__.py", /^(__version__ = ")([^"]+)(")/m],
];
const pkg = JSON.parse(readFileSync("package.json", "utf8"));
const arg = process.argv[2];

if (arg && arg !== "--tag") {
  if (!/^\d+\.\d+\.\d+(-[\w.]+)?$/.test(arg)) throw new Error(`not a version: ${arg}`);
  pkg.version = arg;
  writeFileSync("package.json", JSON.stringify(pkg, null, 2) + "\n");
  for (const [file, re] of COPIES) writeFileSync(file, readFileSync(file, "utf8").replace(re, `$1${arg}$3`));
  // the lock file records the root package version too
  const lock = JSON.parse(readFileSync("package-lock.json", "utf8"));
  lock.version = arg;
  lock.packages[""].version = arg;
  writeFileSync("package-lock.json", JSON.stringify(lock, null, 2) + "\n");
  console.log(`version set to ${arg} — also update Cargo.lock with: cargo update -p sightline --manifest-path src-tauri/Cargo.toml`);
} else {
  const bad = COPIES.map(([file, re]) => [file, readFileSync(file, "utf8").match(re)?.[2]]).filter(([, v]) => v !== pkg.version);
  if (arg === "--tag" && process.argv[3] !== `v${pkg.version}`) bad.push([`tag ${process.argv[3]}`, process.argv[3]]);
  for (const [file, v] of bad) console.error(`${file}: ${v} ≠ package.json ${pkg.version}`);
  if (bad.length) process.exit(1);
  console.log(`version ${pkg.version} consistent`);
}
