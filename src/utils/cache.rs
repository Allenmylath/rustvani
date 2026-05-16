//! Runtime cache directory resolution and model download.
//!
//! Replaces the compile-time `env!("RUSTVANI_CACHE_DIR")` pattern with
//! runtime resolution so binaries are portable across machines and
//! `cargo publish` / docs.rs builds work correctly.
//!
//! Model files are downloaded on first use if not already present in the
//! cache directory. This ensures deployments where the binary was built on
//! a different machine still work correctly.

use std::path::{Path, PathBuf};

pub const SILERO_NATIVE_URL: &str =
    "https://smartturn-rustvani.s3.ap-south-1.amazonaws.com/silero_vad_16k.bin";

pub const SILERO_ONNX_URL: &str =
    "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx";

pub const SMART_TURN_URL: &str =
    "https://smartturn-rustvani.s3.ap-south-1.amazonaws.com/smart_turn_weights+(1).bin.gz";

/// Return the rustvani cache directory, resolved at **runtime**.
///
/// Priority:
/// 1. `RUSTVANI_CACHE_DIR` environment variable.
/// 2. `$HOME/.rustvani/cache` on Unix.
/// 3. `%USERPROFILE%\.rustvani\cache` on Windows.
/// 4. `./.rustvani/cache` as a last-resort fallback.
pub fn cache_dir() -> PathBuf {
    std::env::var("RUSTVANI_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .unwrap_or_else(|_| ".".into());
            PathBuf::from(home).join(".rustvani").join("cache")
        })
}

/// Default path for the Silero native VAD weights (`silero_vad_16k.bin`).
pub fn silero_native_weights_path() -> PathBuf {
    cache_dir().join("silero_vad_16k.bin")
}

/// Default path for the Silero ONNX model (`silero.onnx`).
pub fn silero_ort_model_path() -> PathBuf {
    cache_dir().join("silero.onnx")
}

/// Default path for the smart-turn weights (`smart_turn_weights.bin.gz`).
pub fn smart_turn_weights_path() -> PathBuf {
    cache_dir().join("smart_turn_weights.bin.gz")
}

/// Ensure `path` exists — download from `url` if not.
///
/// Called at runtime by model constructors so that binaries deployed to a
/// different machine than where they were built still find their weights.
pub fn ensure_model(path: &Path, url: &str, desc: &str) -> Result<(), String> {
    if path.exists() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create cache dir {}: {}", parent.display(), e))?;
    }
    log::info!("rustvani: {} not found — downloading from {}", desc, url);
    download(path, url).map_err(|e| {
        format!(
            "Failed to download {}: {}. \
             Ensure network connectivity or place the file manually at: {}",
            desc,
            e,
            path.display()
        )
    })
}

fn download(dest: &Path, url: &str) -> Result<(), String> {
    let dest_str = dest.to_str().unwrap_or_default();

    // 1. curl
    if let Ok(status) = std::process::Command::new("curl")
        .args(["-fsSL", "--retry", "2", "-o", dest_str, url])
        .status()
    {
        if status.success() && non_empty(dest) {
            log::info!("rustvani: downloaded {} via curl", dest_str);
            return Ok(());
        }
    }

    // 2. wget
    if let Ok(status) = std::process::Command::new("wget")
        .args(["-q", "-O", dest_str, url])
        .status()
    {
        if status.success() && non_empty(dest) {
            log::info!("rustvani: downloaded {} via wget", dest_str);
            return Ok(());
        }
    }

    // 3. PowerShell / pwsh (Windows)
    let ps_cmd = format!(
        "Invoke-WebRequest -Uri '{}' -OutFile '{}' -UseBasicParsing",
        url, dest_str
    );
    for ps in &["powershell", "pwsh"] {
        if let Ok(status) = std::process::Command::new(ps)
            .args(["-ExecutionPolicy", "Bypass", "-Command", &ps_cmd])
            .status()
        {
            if status.success() && non_empty(dest) {
                log::info!("rustvani: downloaded {} via {}", dest_str, ps);
                return Ok(());
            }
        }
    }

    Err("no working download tool found (tried curl, wget, PowerShell, pwsh)".into())
}

fn non_empty(path: &Path) -> bool {
    path.metadata().map(|m| m.len() > 0).unwrap_or(false)
}
