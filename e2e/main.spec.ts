import { expect, test } from "./fixtures";
import { home, openModel, tab, trainToEnd, usePreset } from "./helpers";

test("home lists 22 models, filters and searches", async ({ page }) => {
  await home(page);
  await expect(page.locator('[data-testid^="card-"]')).toHaveCount(22);
  await page.locator(".home-head .seg button", { hasText: "Clustering" }).click();
  await expect(page.locator('[data-testid^="card-"]')).toHaveCount(2);
  await page.locator(".home-head .seg button", { hasText: "All" }).click();
  await page.getByRole("searchbox", { name: "Search models" }).fill("knn");
  await expect(page.getByTestId("card-KNeighborsClassifier")).toBeVisible();
  await expect(page.locator('[data-testid^="card-"]')).toHaveCount(1);
  // the description opens from the round "i" on the card
  await page.getByRole("button", { name: "About KNeighborsClassifier" }).hover();
  await expect(page.getByTestId("about-KNeighborsClassifier")).toContainText("Use for");
});

test("language switches at runtime and is remembered", async ({ page }) => {
  await home(page);
  await page.getByRole("radio", { name: "RU" }).click();
  await expect(page.getByRole("heading", { name: "Выберите модель" })).toBeVisible();
  await page.reload();
  await expect(page.getByRole("heading", { name: "Выберите модель" })).toBeVisible();
  await page.getByRole("radio", { name: "EN" }).click();
  await expect(page.getByRole("heading", { name: "Choose a model" })).toBeVisible();
});

test("train LogisticRegression: frames, pause, step, reset, timeline", async ({ page }) => {
  await home(page);
  await openModel(page, "LogisticRegression");
  await usePreset(page, "moons");
  await expect(page.locator(".cta-chip")).toBeVisible();
  await page.getByTestId("train-btn").click();
  await expect(page.getByTestId("train-btn")).toHaveText(/Training/);
  await page.keyboard.press(" ");
  await expect(page.locator(".tag", { hasText: /Paused at iteration/ })).toBeVisible();
  const before = await page.getByTestId("iter").textContent();
  await page.getByTestId("step-btn").click();
  await expect(page.getByTestId("iter")).not.toHaveText(before ?? "");
  await page.getByTestId("train-btn").click(); // "Continue training"
  await expect(page.getByTestId("train-btn")).toHaveText(/again/, { timeout: 60_000 });
  await expect(page.locator(".tag", { hasText: /Converged|Stopped/ })).toBeVisible();
  const acc = Number((await page.getByTestId("metric").locator(".big").textContent())?.replace("−", "-"));
  expect(acc).toBeGreaterThan(0.6);
  // the timeline scrubber replays recorded frames
  const scrub = page.locator("input.scrub");
  await expect(scrub).toBeVisible();
  await scrub.focus();
  await page.keyboard.press("Home");
  await expect(page.getByTestId("iter")).toContainText("1 /");
  await page.getByTestId("reset-btn").click();
  await expect(page.getByTestId("train-btn")).toHaveText(/^Train$/);
});

test("how it decides walks through the real steps", async ({ page }) => {
  await home(page);
  await openModel(page, "LogisticRegression");
  await usePreset(page, "moons");
  await trainToEnd(page);
  await tab(page, /How it decides/);
  await expect(page.getByTestId("step-of")).toHaveText("step 1 of 5");
  for (let i = 0; i < 4; i++) await page.getByTestId("next").click();
  await expect(page.getByTestId("step-of")).toHaveText("step 5 of 5");
  await expect(page.locator(".st.cur .sh")).toContainText(/0\.5 → class/);
  await page.getByTestId("back").click();
  await expect(page.getByTestId("step-of")).toHaveText("step 4 of 5");
});

test("incompatible data offers real fixes", async ({ page }) => {
  await home(page);
  await openModel(page, "LogisticRegression");
  await usePreset(page, "moons");
  await page.locator(".topnav .left button").first().click();
  await openModel(page, "LinearRegression");
  await expect(page.getByTestId("incompatible")).toBeVisible();
  await expect(page.locator(".fix")).toHaveCount(3);
  await expect(page.getByTestId("train-btn")).toBeDisabled();
  await page.locator(".fix button", { hasText: "Switch model" }).click();
  await expect(page.locator(".topnav .mname .mono")).toHaveText("LogisticRegression");
  await expect(page.getByTestId("incompatible")).toHaveCount(0);
});

test("generator builds reproducible data and insights describe it", async ({ page }) => {
  await home(page);
  await openModel(page, "KNeighborsClassifier");
  await tab(page, /^Data$/);
  await page.locator(".data-head .seg button", { hasText: "Generator" }).click();
  await expect(page.getByTestId("generator")).toBeVisible();
  await page.getByTestId("generate").click();
  await expect(page.getByTestId("workspace")).toBeVisible();
  await expect(page.locator(".aside")).toContainText("500 rows");
  await tab(page, /Insights/);
  await expect(page.getByTestId("insights")).toContainText("Basic stats");
  await expect(page.getByTestId("insights")).toContainText("Class balance");
});

test("table editor: edit, undo, use as dataset", async ({ page }) => {
  await home(page);
  await openModel(page, "DecisionTreeClassifier");
  await tab(page, /^Data$/);
  const csv = ["weight_g,diameter_cm,color,type", "150,7.1,red,apple", "170,7.4,green,apple", "120,6.2,yellow,lemon", "180,6.9,green,pear",
    "160,7.6,red,apple", "110,5.9,yellow,lemon", "195,7.0,green,pear", "140,7.3,red,apple", ",6.1,yellow,lemon", "175,6.8,yellow,pear",
    "165,7.5,red,apple", "125,6.0,yellow,lemon"].join(String.fromCharCode(10));
  const chooser = page.waitForEvent("filechooser");
  await page.locator(".data-head button", { hasText: "Import CSV" }).click();
  await (await chooser).setFiles({ name: "fruit.csv", mimeType: "text/csv", buffer: Buffer.from(csv) });
  const editor = page.getByTestId("table-editor");
  await expect(editor).toBeVisible();
  await expect(editor.locator(".miss-bar")).toBeVisible();
  await editor.locator(".miss-bar button", { hasText: "Fill with column mean" }).click();
  await expect(editor.locator(".miss-bar")).toHaveCount(0);
  await editor.getByRole("button", { name: "Undo" }).click();
  await expect(editor.locator(".miss-bar")).toBeVisible();
  await editor.locator(".miss-bar button", { hasText: "Drop row" }).click();
  await page.getByTestId("use-table").click();
  await expect(page.getByTestId("workspace")).toBeVisible();
  await trainToEnd(page);
  await expect(page.getByTestId("tree-view")).toBeVisible();
});

test("advanced drawer explains disabled rows and applies", async ({ page }) => {
  await home(page);
  await openModel(page, "MLPClassifier");
  await page.getByTestId("advanced-btn").click();
  const adv = page.getByTestId("advanced");
  await expect(adv).toContainText("17 parameters");
  await expect(adv.locator('[data-param="momentum"]')).toContainText("solver='sgd' only");
  await adv.getByRole("searchbox").fill("beta");
  await expect(adv.locator(".row")).toHaveCount(2);
  await page.getByTestId("adv-apply").click();
  await expect(adv).toHaveCount(0);
});

test("plot: zoom, add a point, delete a point", async ({ page }) => {
  await home(page);
  await openModel(page, "KNeighborsClassifier");
  await usePreset(page, "blobs");
  await trainToEnd(page);
  const plot = page.getByTestId("plot-main").locator(".plot-area");
  const box = (await plot.boundingBox())!;
  const ticks = page.getByTestId("plot-main").locator(".ticks-x .tk").first();
  const t0 = await ticks.textContent();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  for (let i = 0; i < 5; i++) await page.mouse.wheel(0, -300);
  await expect.poll(async () => ticks.textContent()).not.toBe(t0);
  await plot.dblclick({ position: { x: 5, y: 5 } });
  await expect.poll(async () => ticks.textContent()).toBe(t0); // view eased back home
  await page.waitForTimeout(400);
  const rows = async () => Number((await page.locator(".aside .hint").first().textContent())?.match(/(\d+) rows/)?.[1]);
  const n0 = await rows();
  await page.mouse.click(box.x + 8, box.y + 8);
  await expect.poll(rows).toBe(n0 + 1);
  await page.mouse.move(box.x + 8, box.y + 8);
  await page.keyboard.down("Alt");
  await page.mouse.click(box.x + 8, box.y + 8);
  await page.keyboard.up("Alt");
  await expect.poll(rows).toBe(n0);
});

test("handbook: chapter with formulas and code, try it, open from a model", async ({ page }) => {
  await home(page);
  await page.locator(".topnav .left .seg button", { hasText: "Handbook" }).click();
  await expect(page.getByTestId("handbook")).toBeVisible();
  await page.getByTestId("hb-LogisticRegression").click();
  await expect(page.locator(".hb-doc h1")).toHaveText("LogisticRegression");
  await expect(page.locator(".hb-body .katex").first()).toBeVisible();
  await expect(page.locator(".hb-code pre")).toContainText("LogisticRegression");
  await expect(page.locator(".hb-body table").first()).toBeVisible();
  // in-text link to the intro chapter
  await page.locator('.hb-body a[href^="#hb/intro/"]').first().click();
  await expect(page.locator(".hb-doc h1")).toHaveText(/Start here/);
  await page.getByTestId("hb-LogisticRegression").click();
  await page.getByTestId("hb-try").first().click();
  await expect(page.getByTestId("workspace")).toBeVisible();
  await expect(page.locator(".topnav .mname .mono")).toHaveText("LogisticRegression");
  // and back: the book button opens this model's chapter
  await page.getByTestId("open-chapter").click();
  await expect(page.locator(".hb-doc h1")).toHaveText("LogisticRegression");
});
