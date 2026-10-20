// Lists WebView2 DevTools targets (Tauri app started with remote debugging).
const r = await fetch(`http://127.0.0.1:${process.argv[2] ?? 9333}/json/list`).then((x) => x.json()).catch((e) => ({ error: e.message }));
console.log(Array.isArray(r) ? r.map((x) => `${x.type} ${x.url}`).join("\n") : r.error);
