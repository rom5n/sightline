// Layout regression test: no text overflows or is clipped on the main screens, in Russian and in
// the dev-only long-string mode (every string ~40 % longer), at the smallest and a large window.
// Truncation with an ellipsis is allowed only where the full text is available in a title.
import { expect, test, type Page } from "@playwright/test";
import { auditOverflow } from "./overflow.mjs";
import { openModel } from "./helpers";

type Issue = { kind: string; el: string; text: string };

async function audit(page: Page, where: string) {
  await page.waitForTimeout(1600); // let chips, ripples and springs settle
  const issues = ((await page.evaluate(auditOverflow)) as Issue[]).filter((i) => i.kind !== "ellipsis-titled");
  expect(issues, `${where}: ${JSON.stringify(issues, null, 1)}`).toEqual([]);
}

const cases = [
  { lang: "ru", w: 1280, h: 720, pseudo: false },
  { lang: "ru", w: 1920, h: 1080, pseudo: false },
  { lang: "en", w: 1280, h: 720, pseudo: true },
] as const;

for (const c of cases) {
  test(`no overflow: ${c.lang}${c.pseudo ? " +40 %" : ""} ${c.w}×${c.h}`, async ({ page }) => {
    test.setTimeout(120_000);
    await page.setViewportSize({ width: c.w, height: c.h });
    await page.addInitScript((l) => localStorage.setItem("sightline.settings", JSON.stringify({ theme: "dark", lang: l, firstRunDone: true })), c.lang);
    await page.goto(c.pseudo ? "/?pseudo=1" : "/");
    await expect(page.getByTestId("home")).toBeVisible();
    await expect(page.locator(".engine-pill")).toHaveCount(0, { timeout: 30_000 });
    await audit(page, "home");
    await openModel(page, "MLPClassifier");
    await usePreset(page, "circles");
    await page.getByTestId("train-btn").click();
    await page.waitForTimeout(2500);
    await audit(page, "workspace while training");
    await openHome(page);
    await openModel(page, "DecisionTreeClassifier");
    await usePreset(page, "xor");
    await trainToEnd(page);
    await audit(page, "tree workspace");
    await navTab(page, 3);
    await audit(page, "how it decides");
    await navTab(page, 2);
    await audit(page, "insights");
  });
}

// stretched strings defeat text locators: navigate by position / test id
const navTab = (page: Page, i: number) => page.locator(".topnav .center button").nth(i).click();
async function usePreset(page: Page, preset: string) {
  await navTab(page, 1);
  await expect(page.getByTestId("presets")).toBeVisible();
  const use = page.getByTestId(`use-${preset}`);
  if (await use.count()) await use.click();
  else await page.getByTestId(`preset-${preset}`).locator("button").last().click();
  await expect(page.getByTestId("workspace")).toBeVisible();
}
async function trainToEnd(page: Page) {
  await page.getByTestId("train-btn").click();
  await expect(page.getByTestId("step-btn")).toBeEnabled({ timeout: 60_000 });
  await expect(page.getByTestId("train-btn")).not.toHaveAttribute("aria-pressed", "true", { timeout: 60_000 });
}

async function openHome(page: Page) {
  await page.locator(".topnav .left button").first().click();
  await expect(page.getByTestId("home")).toBeVisible();
}
