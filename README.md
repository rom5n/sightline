<p align="center"><img src="docs/icon/sightline-256.png" width="96" alt="Sightline icon"></p>

<h1 align="center">Sightline</h1>

<p align="center"><b>English</b> · <a href="README.ru.md">Русский</a></p>

<p align="center">See how machine-learning models learn. Pick one of 22 real models, give it data, press <b>Train</b> and watch the decision boundary move, the loss fall and every step of the decision explained — offline, in English or Russian.</p>

<p align="center"><img src="docs/screenshots/training.gif" width="880" alt="MLPClassifier learning the Circles dataset: the boundary closes around the inner ring while the loss falls"></p>

## Download

Get the installer for your system from the [latest release](../../releases/latest):

| System | File |
|---|---|
| Windows 10 / 11 (x64) | `Sightline_<version>_x64-setup.exe` (or `Sightline_<version>_x64_en-US.msi`) |
| macOS 11+ — Apple Silicon | `Sightline_<version>_aarch64.dmg` |
| macOS 11+ — Intel | `Sightline_<version>_x64.dmg` |
| Linux (x64) | `Sightline_<version>_amd64.AppImage`, `Sightline_<version>_amd64.deb` or `Sightline-<version>-1.x86_64.rpm` |

Python is bundled; nothing else needs to be installed. `SHA256SUMS.txt` in the release lists the checksum of every file.

## Features

- **22 models, all real:** 20 scikit-learn models (linear, trees and ensembles, SVC, k-NN, naive Bayes, MLP, k-means, DBSCAN, PCA) plus CatBoostClassifier and CatBoostRegressor, which read text columns natively (`cat_features`, no one-hot).
- **Live training:** the decision boundary morphs on the GPU at 60 fps, the loss panel is never empty (each model shows its own state), Pause / Step / Reset / speed react within 100 ms, and a timeline scrubs back through up to 500 recorded frames.
- **How it decides:** a step-by-step walk through the real computation for any point — tree paths, votes, stage contributions, `coef_ × x`, neighbours, forward passes, distances to centres.
- **Your data:** presets (moons, blobs, circles, spirals, XOR, regression), a spreadsheet-style table editor with CSV import/export, a rule-based generator, and Insights (histograms, pair plot, box plots, outliers, correlation, scaling). Click to add points, drag to move them, Alt+click to delete — the model retrains as you edit.
- **Handbook:** a chapter for every model — idea, a worked example with real numbers, formulas, where it works and fails, a parameter table, Python code that runs, and a "Try it in the app" button.
- **Comfortable:** movable, dockable panels (saved per model), dark and light themes, English and Russian switchable at runtime, keyboard navigation, pan and zoom on every plot.

## Screenshots

| | English | Русский |
|---|---|---|
| Model selection | <img src="docs/screenshots/en/home.png" width="400" alt="Home screen with model cards"> | <img src="docs/screenshots/ru/home.png" width="400" alt="Главный экран с карточками моделей"> |
| Workspace after training | <img src="docs/screenshots/en/workspace.png" width="400" alt="LogisticRegression workspace"> | <img src="docs/screenshots/ru/workspace.png" width="400" alt="Рабочая область LogisticRegression"> |
| How it decides | <img src="docs/screenshots/en/how-it-decides.png" width="400" alt="How it decides, step 4 of 5"> | <img src="docs/screenshots/ru/how-it-decides.png" width="400" alt="Как решает, шаг 4 из 5"> |
| Dataset editor | <img src="docs/screenshots/en/dataset-editor.png" width="400" alt="Table editor with a categorical column"> | <img src="docs/screenshots/ru/dataset-editor.png" width="400" alt="Редактор таблицы"> |
| Handbook | <img src="docs/screenshots/en/handbook.png" width="400" alt="Handbook chapter on decision trees"> | <img src="docs/screenshots/ru/handbook.png" width="400" alt="Глава справочника о деревьях решений"> |

## Installing

The release builds are **unsigned** unless the project's signing secrets are set, so the first launch needs one extra click.

**Windows.** Run `Sightline_<version>_x64-setup.exe`. It installs for the current user (no admin rights), adds a Start-menu shortcut, offers a desktop shortcut on the last page, and registers an uninstaller in *Settings → Apps*. If SmartScreen says *"Windows protected your PC"*, click **More info → Run anyway**. On Windows 10 without WebView2 the installer downloads it.

**macOS.** Open the `.dmg` and drag **Sightline** to *Applications*. An unsigned app is blocked by Gatekeeper the first time: right-click (or Control-click) Sightline in *Applications* → **Open** → **Open**. On macOS 15 and later, open it once, then go to *System Settings → Privacy & Security* and click **Open Anyway**. From a terminal, the same thing is:

```bash
xattr -dr com.apple.quarantine /Applications/Sightline.app
```

**Linux.**

```bash
# AppImage: no installation
chmod +x Sightline_*_amd64.AppImage && ./Sightline_*_amd64.AppImage
```

```bash
# Debian / Ubuntu (installs the WebKitGTK dependencies too)
sudo apt install ./Sightline_*_amd64.deb
```

```bash
# Fedora / RHEL / openSUSE
sudo dnf install ./Sightline-*.x86_64.rpm
```

## Quick start

1. Click **New to machine learning? Start here** on the Home screen for a two-minute guided run, or pick any card. The round **i** on a card explains the model: what it does, when to use it, one strength and one weakness.
2. The workspace opens with a dataset that suits the model. Press **Train** (or Space).
3. Watch the boundary and the Loss panel. **Pause**, **Step** and drag the timeline to replay.
4. Open **How it decides** and click any point to see the real computation for it.
5. Change parameters on the right — fast models retrain immediately. **Advanced** holds the rest.
6. **Data** lets you switch presets, edit a table, generate data by rules or import a CSV.

Shortcuts: Space — train / pause, F11 — maximise a panel, ← / → — previous / next step in How it decides, Ctrl+Shift+P — FPS and timing overlay.

## Build from source

Requirements: Node 22, Rust stable, Python 3.12 with [uv](https://docs.astral.sh/uv/), and on Linux the WebKitGTK packages:

```bash
sudo apt install libwebkit2gtk-4.1-dev libxdo-dev libssl-dev libayatana-appindicator3-dev librsvg2-dev
```

Set up once (on Windows use `.venv/Scripts/python.exe`):

```bash
npm ci
cd engine
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock
cd ..
```

Run the app in development — Vite, the Rust shell and the Python engine (from `engine/.venv`, on a random localhost port with a per-session token):

```bash
npm run tauri dev
```

Browser-only development, without Tauri:

```bash
npm run engine
```

```bash
npm run dev
```

Then open http://127.0.0.1:1420. Build the installers for the current system:

```bash
npm run build:app
```

This builds the engine with PyInstaller into `engine/dist/sightline-engine/` and then runs `tauri build`, which bundles that folder as a resource. Installers land in `src-tauri/target/release/bundle/`.

## Tests

```bash
npm run test:engine      # pytest: schemas vs installed scikit-learn/CatBoost, every adapter × presets, explain == predict, WebSocket flows, every Handbook code example
npm run test:e2e         # Playwright: main flows, all 22 models, every Handbook chapter, layout overflow audit (starts the engine and Vite)
npm run build            # strict TypeScript + production bundle
npm run check:i18n       # every translation key exists in EN and RU
npm run check:version    # the version is the same everywhere
cd engine && .venv/bin/python -m mypy    # strict mypy
cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets -- -D warnings
```

Playwright runs only its bundled Chromium (`npx playwright install chromium`) with a temporary profile, password manager off — never a system Edge or Chrome (see `e2e/browser.mjs`). The [CI workflow](.github/workflows/ci.yml) runs all of these on Linux, Windows and macOS for every push and pull request (the UI tests on Linux).

## Releasing

The version lives in `package.json`; the other manifests are kept in step by a script.

```bash
node scripts/version.mjs 1.1.0
```

```bash
cargo update -p sightline --manifest-path src-tauri/Cargo.toml
```

```bash
git commit -am "Release 1.1.0" && git tag v1.1.0 && git push origin main v1.1.0
```

Pushing a `v*` tag starts the [release workflow](.github/workflows/release.yml): it builds the Windows, macOS (Apple Silicon and Intel) and Linux installers, writes `SHA256SUMS.txt`, collects the commit messages since the previous tag as release notes and publishes the GitHub Release. Re-running it (or *Run workflow* with the same tag) updates the same release and replaces its files.

**Signing (optional).** Without secrets the builds are unsigned (macOS builds are ad-hoc signed). Add these repository secrets to sign:

| Secret | Use |
|---|---|
| `WINDOWS_CERTIFICATE` | Base64 of a code-signing `.pfx` |
| `WINDOWS_CERTIFICATE_PASSWORD` | Its password |
| `APPLE_CERTIFICATE` | Base64 of a *Developer ID Application* `.p12` |
| `APPLE_CERTIFICATE_PASSWORD` | Its password |
| `APPLE_SIGNING_IDENTITY` | e.g. `Developer ID Application: Name (TEAMID)` |
| `APPLE_ID`, `APPLE_PASSWORD`, `APPLE_TEAM_ID` | Notarization: Apple ID, an app-specific password, team ID |

## Project structure

```
engine/                  Python engine (package sightline_engine), tests, PyInstaller spec, pinned deps
  sightline_engine/
    server.py            WebSocket server and session
    runner.py            training worker process (pause / step / speed / cancel)
    datasets.py          presets, generator, tables, CSV, insights, compatibility checks
    prep.py              encoding, scaling, stratified 75/25 split, 2-D plane (raw or PCA)
    protocol.py          Pydantic message models (source of the TypeScript types)
    adapters/            22 model adapters: schema, fit_stepwise, visualize, explain
  tests/                 pytest
src/                     React + TypeScript UI
  engine/                WebSocket + msgpack worker, client, store
  gl/                    WebGL2 plots, network view, marching squares
  screens/               Home, workspace, data, insights, How it decides, Handbook
  handbook/              Handbook chapters (en/, ru/) and their Python examples (code/)
  locales/               en.json, ru.json
src-tauri/               Rust shell: window, sidecar lifecycle, settings, file dialogs; icons
e2e/                     Playwright tests and screenshot / performance tools
scripts/                 engine build, type generation, i18n and version checks, icon generator
docs/                    screenshots, icon sources
.github/workflows/       CI and release
```

Settings and saved layouts are kept in the app-data folder (`%APPDATA%\app.sightline.desktop` on Windows, `~/Library/Application Support/app.sightline.desktop` on macOS, `~/.local/share/app.sightline.desktop` on Linux). Engine errors are logged to `logs/engine.log` in the app's log folder; the UI only shows a friendly message.

## License

[Apache License 2.0](LICENSE).
