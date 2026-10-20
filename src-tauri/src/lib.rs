//! Sightline desktop shell: frameless window with Mica (Windows 11) or a solid
//! fallback, the Python engine as a sidecar on a random localhost port with a
//! per-session token, settings in the app-data folder, and text-file IO for CSV.

use std::fs;
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use rand::{distributions::Alphanumeric, Rng};
use serde::Serialize;
use tauri::{Manager, RunEvent, State};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

struct Engine {
    port: u16,
    token: String,
    child: Mutex<Option<Child>>,
    material: Mutex<String>,
}

#[derive(Serialize)]
struct EngineInfo {
    port: u16,
    token: String,
}

#[tauri::command]
fn engine_info(state: State<Engine>) -> Option<EngineInfo> {
    Some(EngineInfo { port: state.port, token: state.token.clone() })
}

#[tauri::command]
fn window_material(state: State<Engine>) -> String {
    state.material.lock().map(|m| m.clone()).unwrap_or_default()
}

fn settings_path(app: &tauri::AppHandle) -> Option<PathBuf> {
    app.path().app_data_dir().ok().map(|d| d.join("settings.json"))
}

#[tauri::command]
fn load_settings(app: tauri::AppHandle) -> Option<String> {
    settings_path(&app).and_then(|p| fs::read_to_string(p).ok())
}

#[tauri::command]
fn save_settings(app: tauri::AppHandle, json: String) -> Result<(), String> {
    let p = settings_path(&app).ok_or("no app data dir")?;
    if let Some(dir) = p.parent() {
        fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    // write-then-rename so a crash never leaves a half-written file
    let tmp = p.with_extension("json.tmp");
    fs::write(&tmp, json).map_err(|e| e.to_string())?;
    fs::rename(&tmp, &p).map_err(|e| e.to_string())
}

const MAX_TEXT: u64 = 64 * 1024 * 1024;

#[tauri::command]
fn read_text_file(path: String) -> Result<String, String> {
    let meta = fs::metadata(&path).map_err(|e| e.to_string())?;
    if meta.len() > MAX_TEXT {
        return Err("file is larger than 64 MB".into());
    }
    let bytes = fs::read(&path).map_err(|e| e.to_string())?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

#[tauri::command]
fn write_text_file(path: String, contents: String) -> Result<(), String> {
    fs::write(path, contents).map_err(|e| e.to_string())
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .unwrap_or(8765)
}

fn new_token() -> String {
    rand::thread_rng().sample_iter(&Alphanumeric).take(40).map(char::from).collect()
}

/// Dev: run the engine from engine/.venv. Release: the PyInstaller folder shipped as a resource.
fn engine_command(app: &tauri::AppHandle) -> Command {
    if cfg!(debug_assertions) {
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
        let engine = repo.join("engine");
        let py = if cfg!(windows) { engine.join(".venv").join("Scripts").join("python.exe") } else { engine.join(".venv").join("bin").join("python") };
        let mut c = Command::new(if py.exists() { py } else { PathBuf::from("python") });
        c.arg("-m").arg("sightline_engine").current_dir(&engine);
        c
    } else {
        let res = app.path().resource_dir().unwrap_or_default();
        let exe = res.join("engine").join(if cfg!(windows) { "sightline-engine.exe" } else { "sightline-engine" });
        let mut c = Command::new(&exe);
        if let Some(dir) = exe.parent() {
            c.current_dir(dir);
        }
        c
    }
}

fn spawn_engine(app: &tauri::AppHandle, port: u16, token: &str) -> Option<Child> {
    let log = app.path().app_log_dir().ok().map(|d| d.join("engine.log"));
    let mut cmd = engine_command(app);
    cmd.arg("--port").arg(port.to_string())
        .arg("--token").arg(token)
        .arg("--parent-pid").arg(std::process::id().to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if let Some(l) = log {
        cmd.arg("--log").arg(l);
    }
    #[cfg(windows)]
    cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    cmd.spawn().ok()
}

fn apply_material(window: &tauri::WebviewWindow) -> String {
    #[cfg(target_os = "windows")]
    {
        // Mica exists from Windows 11 (build 22000); on Windows 10 keep the solid background.
        if window_vibrancy::apply_mica(window, None).is_ok() {
            return "mica".into();
        }
    }
    let _ = window;
    "none".into()
}

/// The design size is 1440×900; on small or high-DPI screens shrink to fit the work area.
fn fit_to_monitor(window: &tauri::WebviewWindow) {
    if let Ok(Some(m)) = window.current_monitor() {
        let scale = m.scale_factor();
        let area = m.work_area();
        let (aw, ah) = (area.size.width as f64 / scale, area.size.height as f64 / scale);
        let w = 1440f64.min(aw * 0.94).max(1100f64.min(aw));
        let h = 900f64.min(ah * 0.94).max(700f64.min(ah));
        let _ = window.set_size(tauri::LogicalSize::new(w, h));
        let _ = window.center();
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let port = free_port();
            let token = new_token();
            let child = spawn_engine(app.handle(), port, &token);
            let material = app
                .get_webview_window("main")
                .map(|w| {
                    fit_to_monitor(&w);
                    apply_material(&w)
                })
                .unwrap_or_else(|| "none".into());
            app.manage(Engine { port, token, child: Mutex::new(child), material: Mutex::new(material) });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            engine_info,
            window_material,
            load_settings,
            save_settings,
            read_text_file,
            write_text_file
        ])
        .build(tauri::generate_context!())
        .expect("error while building Sightline");

    app.run(|handle, event| {
        if let RunEvent::Exit = event {
            if let Some(state) = handle.try_state::<Engine>() {
                if let Ok(mut c) = state.child.lock() {
                    if let Some(mut child) = c.take() {
                        let _ = child.kill();
                    }
                }
            }
        }
    });
}
