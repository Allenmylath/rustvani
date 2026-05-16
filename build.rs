use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// URL for the smart-turn weights file.
const SMART_TURN_URL: &str =
    "https://smartturn-rustvani.s3.ap-south-1.amazonaws.com/smart_turn_weights+(1).bin.gz";

/// URL for the official Silero VAD ONNX model.
const SILERO_ONNX_URL: &str =
    "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx";

/// URL for the custom-extracted Silero native weights (`silero_vad_16k.bin`).
const SILERO_NATIVE_URL: &str =
    "https://smartturn-rustvani.s3.ap-south-1.amazonaws.com/silero_vad_16k.bin";

fn main() {
    let cache = cache_dir();
    fs::create_dir_all(&cache).unwrap();

    // Emit the cache path for any code that still wants it at compile time.
    println!("cargo:rustc-env=RUSTVANI_CACHE_DIR={}", cache.display());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTVANI_CACHE_DIR");

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // ── Smart-turn weights ───────────────────────────────────────────────
    let weights_cache = cache.join("smart_turn_weights.bin.gz");
    if !weights_cache.exists() {
        let weights_src = manifest.join("src/turn/smart_turn_weights.bin.gz");
        if weights_src.exists() {
            println!("cargo:warning=Copying local smart-turn weights to cache...");
            fs::copy(&weights_src, &weights_cache).unwrap();
        } else {
            download_or_fail(&weights_cache, SMART_TURN_URL, "smart-turn weights");
        }
    }

    // ── Silero ONNX model ────────────────────────────────────────────────
    let silero_onnx_cache = cache.join("silero.onnx");
    if !silero_onnx_cache.exists() {
        let silero_onnx_src = manifest.join("src/vad/data/silero.onnx");
        if silero_onnx_src.exists() {
            println!("cargo:warning=Copying local silero.onnx to cache...");
            fs::copy(&silero_onnx_src, &silero_onnx_cache).unwrap();
        } else {
            download_or_fail(&silero_onnx_cache, SILERO_ONNX_URL, "silero.onnx");
        }
    }

    // ── Silero native weights (custom flat binary) ───────────────────────
    let silero_bin_cache = cache.join("silero_vad_16k.bin");
    if !silero_bin_cache.exists() {
        let silero_bin_src = manifest.join("src/vad/data/silero_vad_16k.bin");
        if silero_bin_src.exists() {
            println!("cargo:warning=Copying local silero_vad_16k.bin to cache...");
            fs::copy(&silero_bin_src, &silero_bin_cache).unwrap();
        } else {
            download_or_fail(&silero_bin_cache, SILERO_NATIVE_URL, "silero_vad_16k.bin");
        }
    }
}

/// Return the cache directory for rustvani model files.
/// Respects `RUSTVANI_CACHE_DIR` env var, otherwise `~/.rustvani/cache`.
fn cache_dir() -> PathBuf {
    env::var("RUSTVANI_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = env::var("HOME")
                .or_else(|_| env::var("USERPROFILE"))
                .expect(
                    "Cannot determine home directory. \
                     Set RUSTVANI_CACHE_DIR env var.",
                );
            PathBuf::from(home).join(".rustvani").join("cache")
        })
}

/// Try to download a file at build time — warn on failure instead of panicking.
/// Missing files are downloaded automatically at runtime on first use.
fn download_or_fail(path: &PathBuf, url: &str, desc: &str) {
    println!("cargo:warning=Downloading {}...", desc);
    if let Err(e) = download(url, path) {
        println!(
            "cargo:warning=Could not pre-download {} ({}). \
             It will be downloaded automatically on first use.",
            desc, e
        );
    }
}

fn download(url: &str, dest: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(dest.parent().unwrap())?;

    // 1. curl
    if let Ok(status) = Command::new("curl")
        .args(&["-fsSL", "--retry", "2", "-o", dest.to_str().unwrap(), url])
        .status()
    {
        if status.success() && dest.exists() && fs::metadata(dest)?.len() > 0 {
            return Ok(());
        }
    }

    // 2. wget
    if let Ok(status) = Command::new("wget")
        .args(&["-q", "-O", dest.to_str().unwrap(), url])
        .status()
    {
        if status.success() && dest.exists() && fs::metadata(dest)?.len() > 0 {
            return Ok(());
        }
    }

    // 3. PowerShell (Windows fallback)
    let ps_cmd = format!(
        "Invoke-WebRequest -Uri '{}' -OutFile '{}' -UseBasicParsing",
        url,
        dest.to_str().unwrap()
    );
    if let Ok(status) = Command::new("powershell")
        .args(&["-ExecutionPolicy", "Bypass", "-Command", &ps_cmd])
        .status()
    {
        if status.success() && dest.exists() && fs::metadata(dest)?.len() > 0 {
            return Ok(());
        }
    }

    // 4. pwsh (PowerShell Core)
    if let Ok(status) = Command::new("pwsh")
        .args(&["-Command", &ps_cmd])
        .status()
    {
        if status.success() && dest.exists() && fs::metadata(dest)?.len() > 0 {
            return Ok(());
        }
    }

    Err("No working download tool found (tried curl, wget, PowerShell, pwsh)".into())
}
