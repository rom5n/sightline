import { expect, test } from "@playwright/test";
import { home, openModel, tab, trainToEnd, usePreset } from "./helpers";

// Every model trains on real data, shows its visual, its Loss-panel state and a "How it decides" walk.
const CASES: [string, string, RegExp][] = [
  ["LinearRegression", "linear", /Fitted in one step · errors/],
  ["LogisticRegression", "moons", /^Loss/],
  ["Ridge", "linear", /Fitted in one step · errors/],
  ["Lasso", "linear", /^Loss/],
  ["ElasticNet", "linear", /^Loss/],
  ["SGDClassifier", "blobs", /^Loss/],
  ["DecisionTreeClassifier", "xor", /Fitted in one step/],
  ["RandomForestClassifier", "moons", /Accuracy as trees are added/],
  ["ExtraTreesClassifier", "moons", /Accuracy as trees are added/],
  ["GradientBoostingClassifier", "circles", /Error per boosting stage/],
  ["HistGradientBoostingClassifier", "spirals", /Error per boosting stage/],
  ["AdaBoostClassifier", "xor", /Error per boosting stage/],
  ["BaggingClassifier", "moons", /Accuracy as trees are added/],
  ["CatBoostClassifier", "moons", /Error per boosting stage/],
  ["CatBoostRegressor", "linear", /Error per boosting stage/],
  ["SVC", "circles", /Fitted in one step/],
  ["KNeighborsClassifier", "moons", /Nothing to train · try k/],
  ["GaussianNB", "blobs", /Fitted in one step/],
  ["MLPClassifier", "circles", /^Loss/],
  ["KMeans", "blobs", /^Inertia/],
  ["DBSCAN", "blobs", /Fitted in one step/],
  ["PCA", "blobs", /Variance kept per component/],
];

for (const [model, preset, lossTitle] of CASES) {
  test(`${model} trains on ${preset} and explains a point`, async ({ page }) => {
    test.setTimeout(120_000);
    await home(page);
    await openModel(page, model);
    await usePreset(page, preset);
    await page.locator(".ctl .seg button", { hasText: "4×" }).click({ timeout: 1500 }).catch(() => undefined);
    await trainToEnd(page);
    await expect(page.locator(".dv-groupview[data-kind=loss]")).toContainText(lossTitle);
    await expect(page.getByTestId("metric").locator(".big")).not.toHaveText("—");
    await tab(page, /How it decides/);
    await expect(page.getByTestId("step-of")).toHaveText(/step 1 of [1-9]\d*/, { timeout: 20_000 }); // explanation arrived
    const n = Number((await page.getByTestId("step-of").textContent())!.match(/of (\d+)/)![1]);
    for (let i = 1; i < n; i++) await page.getByTestId("next").click();
    await expect(page.getByTestId("step-of")).toHaveText(`step ${n} of ${n}`);
  });
}

test("CatBoost reads a text column natively and shows it", async ({ page }) => {
  test.setTimeout(180_000);
  await home(page);
  await openModel(page, "CatBoostClassifier");
  await expect(page.getByTestId("cat-note")).toBeVisible();
  // generator default has a text column "color"
  await page.locator(".topnav .center button", { hasText: /^Data$/ }).click();
  await page.locator(".data-head .seg button", { hasText: "Generator" }).click();
  await page.getByTestId("generate").click();
  await expect(page.getByTestId("workspace")).toBeVisible();
  await expect(page.getByTestId("cat-note")).toContainText("cat_features: color");
  await trainToEnd(page);
  await tab(page, /How it decides/);
  await expect(page.locator(".dcard").first()).toContainText("category");
});
