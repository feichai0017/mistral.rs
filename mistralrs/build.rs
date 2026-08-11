use std::{env, ffi::OsStr, path::PathBuf, process::Command};

const COMPUTE_CAP_ENV: &str = "CUDA_COMPUTE_CAP";
const NVCC_ENV: &str = "NVCC";
const TEST_BUILD_ENV: &str = "OXIDE_GEMM_CENSUS_TEST_BUILD";

fn command_output(command: &OsStr, args: &[&str], label: &str) -> String {
    let output = Command::new(command)
        .args(args)
        .output()
        .unwrap_or_else(|error| panic!("failed to execute {label}: {error}"));
    if !output.status.success() {
        panic!(
            "{label} failed with status {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let output = String::from_utf8(output.stdout)
        .unwrap_or_else(|error| panic!("{label} output was not UTF-8: {error}"));
    let output = output.trim();
    if output.is_empty() {
        panic!("{label} must return nonempty output");
    }
    output.to_string()
}

fn required_single_line_env(name: &str) -> String {
    let value = env::var(name).unwrap_or_else(|_| panic!("{name} must be set for gemm-census"));
    if value.is_empty() || value.contains(['\r', '\n']) {
        panic!("{name} must contain one nonempty line");
    }
    value
}

fn hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn canonical_nvcc_path() -> PathBuf {
    let raw_path = required_single_line_env(NVCC_ENV);
    let path = PathBuf::from(&raw_path);
    if !path.is_absolute() {
        panic!("{NVCC_ENV} must be an absolute path, got {raw_path:?}");
    }
    let path = path
        .canonicalize()
        .unwrap_or_else(|error| panic!("failed to canonicalize {NVCC_ENV}={raw_path:?}: {error}"));
    if !path.is_file() {
        panic!("{NVCC_ENV} must resolve to a file, got {}", path.display());
    }
    path
}

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_GEMM_CENSUS");
    if env::var_os("CARGO_FEATURE_GEMM_CENSUS").is_none() {
        return;
    }

    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA");
    println!("cargo:rerun-if-env-changed={COMPUTE_CAP_ENV}");
    println!("cargo:rerun-if-env-changed={NVCC_ENV}");
    println!("cargo:rerun-if-env-changed={TEST_BUILD_ENV}");

    let rustc = env::var_os("RUSTC").expect("Cargo did not provide RUSTC");
    let rustc_version = command_output(&rustc, &["--version"], "build rustc");
    if rustc_version.contains(['\r', '\n']) {
        panic!("build rustc must return one nonempty line");
    }

    let (nvcc_path, nvcc_version_hex, compute_cap, cuda_arch) =
        if env::var_os("CARGO_FEATURE_CUDA").is_some() {
            if env::var_os(TEST_BUILD_ENV).is_some() {
                panic!("{TEST_BUILD_ENV} is forbidden when cuda is enabled");
            }
            let compute_cap = required_single_line_env(COMPUTE_CAP_ENV);
            if compute_cap != "90" {
                panic!("{COMPUTE_CAP_ENV} must be 90, got {compute_cap:?}");
            }
            let nvcc_path = canonical_nvcc_path();
            let nvcc_version = command_output(nvcc_path.as_os_str(), &["--version"], "build nvcc");
            let nvcc_path = nvcc_path
                .to_str()
                .unwrap_or_else(|| panic!("canonical {NVCC_ENV} path must be UTF-8"))
                .to_string();
            (
                nvcc_path,
                hex_bytes(nvcc_version.as_bytes()),
                compute_cap,
                "sm_90a".to_string(),
            )
        } else {
            let test_build = required_single_line_env(TEST_BUILD_ENV);
            if test_build != "1" {
                panic!("{TEST_BUILD_ENV} must be 1 for a non-CUDA test build");
            }
            (
                "test-only".to_string(),
                "test-only".to_string(),
                "test-only".to_string(),
                "test-only".to_string(),
            )
        };

    println!("cargo:rustc-env=OXIDE_GEMM_CENSUS_BUILD_RUSTC_VERSION={rustc_version}");
    println!("cargo:rustc-env=OXIDE_GEMM_CENSUS_BUILD_NVCC_PATH={nvcc_path}");
    println!("cargo:rustc-env=OXIDE_GEMM_CENSUS_BUILD_NVCC_VERSION_HEX={nvcc_version_hex}");
    println!("cargo:rustc-env=OXIDE_GEMM_CENSUS_BUILD_CUDA_COMPUTE_CAP={compute_cap}");
    println!("cargo:rustc-env=OXIDE_GEMM_CENSUS_BUILD_CUDA_ARCH={cuda_arch}");
}
