// Crops a region of a PNG at 2x for close inspection: node e2e/crop.mjs in.png out.png x y w h
import { launchChromium } from "./browser.mjs";
import { readFileSync } from "node:fs";
const [inp, out, x, y, w, h] = process.argv.slice(2);
const b = await launchChromium();
const p = await b.newPage({ viewport: { width: +w * 2, height: +h * 2 } });
const src = "data:image/png;base64," + readFileSync(inp).toString("base64");
await p.setContent(`<body style="margin:0;overflow:hidden"><div style="width:${w * 2}px;height:${h * 2}px;background:url(${src}) -${x * 2}px -${y * 2}px / auto;image-rendering:pixelated;transform-origin:0 0"></div></body>`);
await p.evaluate(([s, x, y]) => { const d = document.querySelector("div"); const i = new Image(); i.src = s; return i.decode().then(() => { d.style.backgroundSize = `${i.width * 2}px ${i.height * 2}px`; }); }, [src, x, y]);
await p.screenshot({ path: out });
await b.close();
