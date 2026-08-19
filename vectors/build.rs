fn main() {
    println!("cargo::rerun-if-changed=src/lvq.c");
    println!("cargo::rerun-if-changed=src/quiver.c");
    cc::Build::new()
        .file("src/lvq.c")
        .file("src/quiver.c")
        .opt_level(3)
        .compile("vectors");
}
