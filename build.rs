fn main() {
    println!("cargo::rerun-if-changed=src/vector/float16.c");
    cc::Build::new()
        .file("src/vectors/float16.c")
        .compile("float16");
}
