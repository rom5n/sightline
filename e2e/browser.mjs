// The only browser the tests and dev scripts may launch: Playwright's bundled Chromium
// (from the ms-playwright folder) with a throwaway profile. Never a system Edge/Chrome, never a
// persistent profile, and no password manager, autofill or automatic Windows sign-in.
import { chromium } from "@playwright/test";

export const CHROMIUM_ARGS = [
  // no saved passwords, no save/fill prompts, no OS keychain / Windows credential store
  "--password-store=basic",
  "--disable-save-password-bubble",
  "--disable-features=PasswordManagerOnboarding,PasswordLeakDetection,AutofillServerCommunication,AutofillEnableAccountWalletStorage",
  // never answer NTLM/Negotiate challenges with the Windows account (that is what shows up as failed logons)
  "--auth-server-allowlist=_",
  "--auth-negotiate-delegate-allowlist=_",
  "--disable-sync",
];

/** Bundled Chromium, fresh temporary profile; close it when done. */
export function launchChromium(options = {}) {
  return chromium.launch({ ...options, args: [...CHROMIUM_ARGS, ...(options.args ?? [])] });
}
