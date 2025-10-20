fn main() {
    println!("cargo::rerun-if-changed=src/float16.c");
    println!("cargo::rerun-if-changed=src/lvq.c");
    cc::Build::new()
        .file("src/float16.c")
        .file("src/lvq.c")
        .opt_level(3)
        .compile("vectors");
}
