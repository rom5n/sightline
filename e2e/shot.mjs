// Dev helper: screenshots of app states at a given size for visual review.
// usage: node e2e/shot.mjs <out-dir> [width] [height] [script-name]
import { launchChromium } from "./browser.mjs";
import { writeFileSync } from "node:fs";
import { auditOverflow } from "./overflow.mjs";
const [out = "shots", W = "1440", H = "900", only = ""] = process.argv.slice(2);
// AUDIT=1: after every screenshot, list text that overflows or is clipped (written to audit.json)
const audits = {};
const browser = await launchChromium();
const page = await browser.newPage({ viewport: { width: Number(W), height: Number(H) }, deviceScaleFactor: 1 });
const theme = process.env.THEME ?? "dark", lang = process.env.LANG_UI ?? "en";
await page.addInitScript(([t, l]) => localStorage.setItem("sightline.settings", JSON.stringify({ theme: t, lang: l, firstRunDone: true })), [theme, lang]);
page.on("console", (m) => { if (m.type() === "error") console.log("console error:", m.text()); });
page.on("pageerror", (e) => console.log("page error:", e.message));
const shot0 = page.screenshot.bind(page);
page.screenshot = async (o) => {
  const r = await shot0(o);
  if (process.env.AUDIT) audits[o.path.split("/").pop()] = await page.evaluate(auditOverflow);
  return r;
};
const url = process.env.APP_URL ?? "http://localhost:1420/";
const wait = (ms) => page.waitForTimeout(ms);
const clickTab = (name) => page.locator(".topnav .center button").nth({ Workspace: 0, Data: 1, Insights: 2, "How it decides": 3 }[name] ?? 0).click();
const open = async (model) => { await page.goto(url); await wait(600); await page.locator(`[data-testid="card-${model}"]`).click(); await wait(400); };
const preset = async (id, settings) => {
  await clickTab("Data"); await wait(500);
  await page.locator(`[data-testid="preset-${id}"] button`).last().click(); await wait(700);
};
const train = async (ms = 3000) => { await page.locator('[data-testid="train-btn"]').click(); await wait(ms); };
const shots = {
  async home() { await page.goto(url); await wait(800); await page.screenshot({ path: `${out}/home.png` }); },
  async logistic() { await open("LogisticRegression"); await preset("moons"); await page.evaluate(() => localStorage.setItem("x", "1")); await train(1200); await page.keyboard.press(" "); await wait(400); await page.screenshot({ path: `${out}/ws-logistic-paused.png` }); await page.keyboard.press(" "); await wait(3500); await page.screenshot({ path: `${out}/ws-logistic.png` }); },
  async tree() { await open("DecisionTreeClassifier"); await preset("xor"); await train(7000); await page.screenshot({ path: `${out}/ws-tree.png` }); },
  async kmeans() { await open("KMeans"); await preset("blobs"); await train(5000); await page.screenshot({ path: `${out}/ws-kmeans.png` }); },
  async mlp() { await open("MLPClassifier"); await preset("circles"); await train(4000); await page.screenshot({ path: `${out}/ws-mlp.png` }); },
  async decide() { await open("LogisticRegression"); await preset("moons"); await train(4000); await clickTab("How it decides"); await wait(1500); for (let i = 0; i < 3; i++) { await page.locator('[data-testid="next"]').click(); await wait(400); } await page.screenshot({ path: `${out}/decide-logistic.png` }); },
  async data() { await page.goto(url); await wait(600); await page.locator(".topnav .left .seg button").nth(1).click(); await wait(1500); await page.screenshot({ path: `${out}/data-presets.png` }); await page.locator(".data-head .seg button").nth(1).click(); await wait(800); await page.screenshot({ path: `${out}/data-table.png` }); await page.locator(".data-head .seg button").nth(2).click(); await wait(1200); await page.screenshot({ path: `${out}/data-generator.png` }); },
  async insights() { await open("LogisticRegression"); await preset("moons"); await clickTab("Insights"); await wait(1500); await page.screenshot({ path: `${out}/insights.png` }); },
  async advanced() { await open("MLPClassifier"); await page.locator('[data-testid="advanced-btn"]').click(); await wait(800); await page.screenshot({ path: `${out}/advanced.png` }); },
  async states() { await open("LinearRegression"); await preset("moons").catch(() => undefined); await page.goto(url); await wait(500); await page.locator(`[data-testid="card-LinearRegression"]`).click(); await wait(800); await page.screenshot({ path: `${out}/state-incompatible.png` }); },
  async decidetree() { await open("DecisionTreeClassifier"); await preset("xor"); await train(6000); await clickTab("How it decides"); await wait(1500); await page.locator('[data-testid="next"]').click(); await wait(300); await page.locator('[data-testid="next"]').click(); await wait(900); await page.screenshot({ path: `${out}/decide-tree.png` }); },
  async decidekm() { await open("KMeans"); await preset("blobs"); await train(5000); await clickTab("How it decides"); await wait(1500); for (let i = 0; i < 2; i++) { await page.locator('[data-testid="next"]').click(); await wait(400); } await page.screenshot({ path: `${out}/decide-kmeans.png` }); },
  async decidemlp() { await open("MLPClassifier"); await preset("circles"); await train(9000); await clickTab("How it decides"); await wait(1500); for (let i = 0; i < 2; i++) { await page.locator('[data-testid="next"]').click(); await wait(500); } await page.screenshot({ path: `${out}/decide-mlp.png` }); },
  async handbook() {
    const ch = process.env.CHAPTER ?? "intro";
    await page.goto(url); await page.waitForSelector('[data-testid="home"]'); await wait(400);
    await page.locator(".topnav .left .seg button").nth(2).click();
    await page.waitForSelector('[data-testid="handbook"]', { timeout: 15000 });
    await page.locator(`[data-testid="hb-${ch}"]`).click(); await wait(1500);
    await page.screenshot({ path: `${out}/handbook-${ch}.png` });
    const doc = page.locator(".hb-doc");
    for (const [i, a] of ["steps", "formulas", "params", "code"].entries()) {
      await doc.evaluate((el, id) => el.querySelector(`#${id}`)?.scrollIntoView(), a); await wait(400);
      await page.screenshot({ path: `${out}/handbook-${ch}-${i + 1}-${a}.png` });
    }
  },
  async catboost() {
    await open("CatBoostClassifier"); await clickTab("Data"); await wait(500);
    await page.locator(".data-head .seg button").nth(2).click(); await wait(800);
    await page.locator('[data-testid="generate"]').click(); await wait(1200);
    await train(1000); await page.screenshot({ path: `${out}/ws-catboost-training.png` });
    await page.waitForFunction(() => /again|заново/.test(document.querySelector('[data-testid="train-btn"]')?.textContent ?? ""), null, { timeout: 90000 });
    await wait(800); await page.screenshot({ path: `${out}/ws-catboost.png` });
    await clickTab("How it decides"); await wait(2500);
    await page.locator('[data-testid="next"]').click(); await wait(600);
    await page.screenshot({ path: `${out}/decide-catboost.png` });
  },
};
if (process.env.RECTS) shots.rects = async () => { await shots.decidekm(); console.log(await page.evaluate((sels) => sels.map((q) => { const e = document.querySelector(q); if (!e) return q + ": none"; const r = e.getBoundingClientRect(); return `${q}: y=${Math.round(r.y)} h=${Math.round(r.height)} sh=${e.scrollHeight}`; }), [".screen", ".dec", ".dec-main", ".dec-main > div:nth-child(2)", ".dec-body", ".dec > aside"])); };
for (const [k, f] of Object.entries(shots)) {
  if (only && !only.split(",").includes(k)) continue;
  try { await f(); console.log("ok", k); } catch (e) { console.log("FAIL", k, e.message.split("\n")[0]); }
}
if (process.env.AUDIT) writeFileSync(`${out}/audit.json`, JSON.stringify(audits, null, 1));
await browser.close();
