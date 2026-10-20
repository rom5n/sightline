import { expect, type Page } from "@playwright/test";

export async function home(page: Page) {
  await page.goto("/");
  await expect(page.getByTestId("home")).toBeVisible();
  await expect(page.locator(".engine-pill")).toHaveCount(0, { timeout: 30_000 });
}

export async function openModel(page: Page, model: string) {
  await page.getByTestId(`card-${model}`).click();
  await expect(page.getByTestId("workspace")).toBeVisible();
}

export async function usePreset(page: Page, preset: string) {
  await page.locator(".topnav .center button", { hasText: /^(Data|Данные)$/ }).click();
  await expect(page.getByTestId("presets")).toBeVisible();
  const card = page.getByTestId(`preset-${preset}`);
  const use = page.getByTestId(`use-${preset}`);
  if (await use.count()) await use.click();
  else await card.locator("button").last().click();
  await expect(page.getByTestId("workspace")).toBeVisible();
}

export async function trainToEnd(page: Page) {
  await page.getByTestId("train-btn").click();
  await expect(page.getByTestId("train-btn")).toHaveText(/again|заново/, { timeout: 60_000 });
}

export async function tab(page: Page, name: RegExp) {
  await page.locator(".topnav .center button", { hasText: name }).click();
}
