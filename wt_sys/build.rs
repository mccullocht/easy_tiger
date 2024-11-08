use std::env;
use std::fs::remove_file;
use std::path::PathBuf;
use std::process::Command;

const GITHUB_WT_TAGS_URI: &str = "https://github.com/wiredtiger/wiredtiger/archive/refs/tags";
const WT_VERSION: &str = "11.2.0";

fn download_source() {
    let uri = format!("{GITHUB_WT_TAGS_URI}/{WT_VERSION}.tar.gz");
    Command::new("wget")
        .arg(uri)
        .arg("-P")
        .arg(env::var("OUT_DIR").unwrap())
        .output()
        .expect("Failed to download source");
}

fn extract_source() {
    let out_dir = env::var("OUT_DIR").unwrap();
    Command::new("tar")
        .arg("-xvf")
        .arg(format!("{out_dir}/{WT_VERSION}.tar.gz"))
        .arg("-C")
        .arg(out_dir)
        .output()
        .expect("Failed to extract source");
}

fn build_wt() {
    // TODO: switch to the CMake crate.
    // TODO: use PROFILE={release,debug} to set HAVE_DIAGNOSTIC
    // TODO: use NUM_JOBS to set parallelism.
    let src_dir = format!("{}/wiredtiger-{WT_VERSION}", env::var("OUT_DIR").unwrap());
    let build_dir = format!("{src_dir}/build");
    Command::new("cmake")
        .arg("-DENABLE_STATIC=1")
        .arg("-DHAVE_DIAGNOSTIC=0")
        .arg("-S")
        .arg(src_dir)
        .arg("-B")
        .arg(build_dir.clone())
        .output()
        .expect("Failed to generate build files");
    Command::new("cmake")
        .arg("--build")
        .arg(build_dir)
        .arg("-j16")
        .output()
        .expect("Failed to build WiredTiger");
}

fn cleanup() {
    let path = format!("{}/{WT_VERSION}.tar.gz", env::var("OUT_DIR").unwrap());
    remove_file(path).expect("Failed to cleanup");
}

fn main() {
    download_source();
    extract_source();
    build_wt();
    cleanup();

    let wt_build_dir = format!(
        "{}/wiredtiger-{WT_VERSION}/build",
        env::var("OUT_DIR").unwrap()
    );

    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search={wt_build_dir}");

    // Tell cargo to tell rustc to statically link with the wiredtiger library.
    // This requires that WT was configured with the -DENABLE_STATIC=1 option to cmake.
    println!("cargo:rustc-link-lib=static=wiredtiger");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header(format!("{wt_build_dir}/include/wiredtiger.h"))
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
