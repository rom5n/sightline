// Every Handbook chapter renders in both languages: formulas without KaTeX errors, the code block,
// the "Try it" button, no leftover placeholders — and no text overflow (the layout audit).
import { expect, test } from "@playwright/test";
import { auditOverflow } from "./overflow.mjs";

const CHAPTERS = ["intro", "LinearRegression", "LogisticRegression", "Ridge", "Lasso", "ElasticNet", "SGDClassifier",
  "DecisionTreeClassifier", "RandomForestClassifier", "ExtraTreesClassifier", "GradientBoostingClassifier",
  "HistGradientBoostingClassifier", "AdaBoostClassifier", "BaggingClassifier", "CatBoost", "SVC",
  "KNeighborsClassifier", "GaussianNB", "MLPClassifier", "KMeans", "DBSCAN", "PCA"];

for (const lang of ["en", "ru"]) {
  test(`handbook renders every chapter (${lang})`, async ({ page }) => {
    test.setTimeout(180_000);
    await page.setViewportSize({ width: 1280, height: 720 });
    await page.addInitScript((l) => localStorage.setItem("sightline.settings", JSON.stringify({ theme: "dark", lang: l, firstRunDone: true })), lang);
    await page.goto("/");
    await expect(page.getByTestId("home")).toBeVisible();
    await page.locator(".topnav .left .seg button").nth(2).click();
    await expect(page.getByTestId("handbook")).toBeVisible();
    for (const ch of CHAPTERS) {
      await page.getByTestId(`hb-${ch}`).click();
      const body = page.locator(".hb-body");
      await expect(body.locator(".hb-code pre"), `${lang}/${ch}: code`).toHaveCount(1);
      if (ch !== "intro") await expect(page.getByTestId("hb-try"), `${lang}/${ch}: try`).toHaveCount(2);
      await expect(body, `${lang}/${ch}: placeholders`).not.toContainText("{{");
      await expect(body, `${lang}/${ch}: raw math`).not.toContainText("$$");
      expect(await body.locator(".katex-error").count(), `${lang}/${ch}: KaTeX errors`).toBe(0);
      expect(await body.locator(".katex").count(), `${lang}/${ch}: formulas`).toBeGreaterThan(3);
      // audit the top and the parameter table of the chapter
      for (const anchor of ["", "params"]) {
        if (anchor) await page.locator(`.hb-doc #${anchor}`).scrollIntoViewIfNeeded();
        const issues = ((await page.evaluate(auditOverflow)) as { kind: string }[]).filter((i) => i.kind !== "ellipsis-titled");
        expect(issues, `${lang}/${ch}#${anchor}: ${JSON.stringify(issues)}`).toEqual([]);
      }
    }
  });
}
