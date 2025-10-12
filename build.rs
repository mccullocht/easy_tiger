fn main() {
    println!("cargo::rerun-if-changed=src/vectors/float16.c");
    println!("cargo::rerun-if-changed=src/vectors/lvq.c");
    cc::Build::new()
        .file("src/vectors/float16.c")
        .file("src/vectors/lvq.c")
        .opt_level(3)
        .compile("vectors");
}
