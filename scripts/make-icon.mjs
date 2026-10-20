// Sightline app icon: the lime "sight" mark (ring, line, inner circle) on a black rounded tile.
//
//   node scripts/make-icon.mjs
//
// Writes docs/icon/sightline.svg (+ -small.svg), src-tauri/icons/png/<n>x<n>.png (16 … 1024),
// src-tauri/icons/{32x32,64x64,128x128,128x128@2x,icon}.png, icon.ico (16–256) and icon.icns.
import { launchChromium } from "../e2e/browser.mjs";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";

const MARK = (stroke) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024">
<rect x="64" y="64" width="896" height="896" rx="200" fill="#121417"/>
<rect x="64" y="64" width="896" height="896" rx="200" fill="none" stroke="#2A2E33" stroke-width="12"/>
<g fill="none" stroke="#C4EC6A" stroke-width="${stroke}" stroke-linecap="round">
<circle cx="512" cy="512" r="300"/><path d="M212 512H812"/><circle cx="512" cy="512" r="90"/></g></svg>`;
// the Sightline mark: lime ring, sight line and inner circle on a black tile;
// at ≤ 32 px the same mark is drawn with thicker strokes so it stays crisp
const FULL = MARK(56);
const SMALL = MARK(64);

const SIZES = [16, 24, 32, 48, 64, 128, 256, 512, 1024];
mkdirSync("docs/icon", { recursive: true });
mkdirSync("src-tauri/icons/png", { recursive: true });
writeFileSync("docs/icon/sightline.svg", FULL);
writeFileSync("docs/icon/sightline-small.svg", SMALL);

const b = await launchChromium();
const p = await b.newPage({ viewport: { width: 1024, height: 1024 }, deviceScaleFactor: 1 });
const png = {};
for (const n of SIZES) {
  const svg = (n <= 32 ? SMALL : FULL).replace("<svg ", `<svg width="${n}" height="${n}" `);
  await p.setContent(`<html><body style="margin:0;background:transparent">${svg}</body></html>`);
  png[n] = await p.locator("svg").screenshot({ omitBackground: true });
  writeFileSync(`src-tauri/icons/png/${n}x${n}.png`, png[n]);
}
// Windows Store / MSIX tiles (Tauri keeps these next to the other icons)
for (const n of [30, 44, 71, 89, 107, 142, 150, 284, 310]) {
  const svg = (n <= 44 ? SMALL : FULL).replace("<svg ", `<svg width="${n}" height="${n}" `);
  await p.setContent(`<html><body style="margin:0;background:transparent">${svg}</body></html>`);
  writeFileSync(`src-tauri/icons/Square${n}x${n}Logo.png`, await p.locator("svg").screenshot({ omitBackground: true }));
}
await p.setContent(`<html><body style="margin:0;background:transparent">${FULL.replace("<svg ", '<svg width="50" height="50" ')}</body></html>`);
writeFileSync("src-tauri/icons/StoreLogo.png", await p.locator("svg").screenshot({ omitBackground: true }));
await b.close();

// files Tauri references
writeFileSync("src-tauri/icons/32x32.png", png[32]);
writeFileSync("src-tauri/icons/64x64.png", png[64]);
writeFileSync("src-tauri/icons/128x128.png", png[128]);
writeFileSync("src-tauri/icons/128x128@2x.png", png[256]);
writeFileSync("src-tauri/icons/icon.png", png[512]);
writeFileSync("docs/icon/sightline-256.png", png[256]);

// .ico: PNG-compressed entries (Windows Vista+), each size drawn for its size
const icoSizes = [16, 24, 32, 48, 64, 128, 256];
const head = Buffer.alloc(6 + 16 * icoSizes.length);
head.writeUInt16LE(0, 0); head.writeUInt16LE(1, 2); head.writeUInt16LE(icoSizes.length, 4);
let offset = head.length;
icoSizes.forEach((n, i) => {
  const e = 6 + 16 * i;
  head.writeUInt8(n === 256 ? 0 : n, e); head.writeUInt8(n === 256 ? 0 : n, e + 1);
  head.writeUInt16LE(1, e + 4); head.writeUInt16LE(32, e + 6);
  head.writeUInt32LE(png[n].length, e + 8); head.writeUInt32LE(offset, e + 12);
  offset += png[n].length;
});
writeFileSync("src-tauri/icons/icon.ico", Buffer.concat([head, ...icoSizes.map((n) => png[n])]));

// .icns: PNG entries for every macOS slot (1x and @2x)
const ICNS = [["icp4", 16], ["icp5", 32], ["icp6", 64], ["ic07", 128], ["ic08", 256], ["ic09", 512], ["ic10", 1024],
  ["ic11", 32], ["ic12", 64], ["ic13", 256], ["ic14", 512]];
const chunks = ICNS.map(([type, n]) => {
  const h = Buffer.alloc(8);
  h.write(type, 0, "ascii");
  h.writeUInt32BE(png[n].length + 8, 4);
  return Buffer.concat([h, png[n]]);
});
const body = Buffer.concat(chunks);
const top = Buffer.alloc(8);
top.write("icns", 0, "ascii");
top.writeUInt32BE(body.length + 8, 4);
writeFileSync("src-tauri/icons/icon.icns", Buffer.concat([top, body]));
console.log("icons written:", SIZES.join(", "), "| ico", readFileSync("src-tauri/icons/icon.ico").length, "B | icns", top.readUInt32BE(4), "B");
