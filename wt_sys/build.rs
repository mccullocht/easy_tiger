use std::env;
use std::fs::remove_file;
use std::path::{Path, PathBuf};
use std::process::Command;

const GITHUB_WT_TAGS_URI: &str = "https://github.com/wiredtiger/wiredtiger/archive/refs/tags";
const WT_VERSION: &str = "11.2.0";

fn download_source() -> PathBuf {
    let uri = format!("{GITHUB_WT_TAGS_URI}/{WT_VERSION}.tar.gz");
    let out_dir = env::var("OUT_DIR").unwrap();
    Command::new("wget")
        .arg(uri)
        .arg("-P")
        .arg(out_dir.clone())
        .output()
        .expect("Failed to download source");
    PathBuf::from_iter([out_dir, format!("{WT_VERSION}.tar.gz")])
}

fn extract_source(tar_path: &Path) -> PathBuf {
    let mut src_path = tar_path.to_path_buf();
    src_path.pop();
    Command::new("tar")
        .arg("-xvf")
        .arg(tar_path)
        .arg("-C")
        .arg(&src_path)
        .output()
        .expect("Failed to extract source");
    src_path.push(format!("wiredtiger-{WT_VERSION}"));
    src_path
}

fn build_wt(src_path: &Path) -> PathBuf {
    let have_diagnostic = env::var("PROFILE").unwrap() == "debug";
    let build_path = cmake::Config::new(src_path)
        .define("ENABLE_STATIC", "1")
        .define("HAVE_DIAGNOSTIC", if have_diagnostic { "1" } else { "0" })
        // Overidde C_FLAGS and CXX_FLAGS to keep cmake from passing both --target and -mmacosx-version-min
        .define("CMAKE_C_FLAGS", "")
        .define("CMAKE_CXX_FLAGS", "")
        .no_build_target(false)
        .build();
    PathBuf::from_iter([build_path, PathBuf::from("build")])
}

fn main() {
    let tar_path = download_source();
    let src_path = extract_source(&tar_path);
    let build_path = build_wt(&src_path);
    remove_file(tar_path).expect("failed to cleanup source tar");

    println!("cargo::rerun-if-changed=build.rs");

    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search={}", build_path.display());

    // Tell cargo to tell rustc to statically link with the wiredtiger library.
    // This requires that WT was configured with the -DENABLE_STATIC=1 option to cmake.
    println!("cargo:rustc-link-lib=static=wiredtiger");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header(format!("{}/include/wiredtiger.h", build_path.display()))
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
