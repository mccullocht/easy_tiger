use std::env;
use std::path::PathBuf;

fn build_wt() -> PathBuf {
    let have_diagnostic = env::var("PROFILE").unwrap() == "debug";
    let jobs = env::var("NUM_JOBS").unwrap_or("1".to_string());
    let build_path = cmake::Config::new("wiredtiger")
        .define("ENABLE_STATIC", "1")
        .define("HAVE_DIAGNOSTIC", if have_diagnostic { "1" } else { "0" })
        .define("ENABLE_PYTHON", "0")
        .define("HAVE_UNITTEST", "0")
        // We have quasi-vendored WT and won't change upstream source to handle errors from new compiler versions.
        .cflag("-Wno-everything")
        .cflag("-w")
        // CMake crate is not doing this correctly for whatever reason.
        .build_arg(format!("-j{jobs}"))
        .build_target("wiredtiger_static")
        .build();
    PathBuf::from_iter([build_path, PathBuf::from("build")])
}

fn main() {
    let build_path = build_wt();

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=wiredtiger");

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
