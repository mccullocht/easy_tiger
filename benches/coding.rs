use criterion::{criterion_group, criterion_main, Criterion};
use easy_tiger::vectors::{F32VectorCoding, VectorSimilarity};
use rand::{Rng, SeedableRng};

fn generate_test_vector(dim: usize) -> Vec<f32> {
    let rng = rand_xoshiro::Xoroshiro128PlusPlus::seed_from_u64(0x455A_5469676572);
    rng.random_iter::<f32>()
        .by_ref()
        .take(dim)
        .collect::<Vec<_>>()
}

fn benchmark_coding(
    c: &mut Criterion,
    format: F32VectorCoding,
    similarity: Option<VectorSimilarity>,
) {
    let vector = generate_test_vector(1024);
    let coder = format.new_coder(similarity.unwrap_or(VectorSimilarity::Dot));
    let mut out = vec![0u8; coder.byte_len(vector.len())];
    let id = if let Some(s) = similarity {
        format!("{format}/coding/{s}")
    } else {
        format!("{format}/coding")
    };
    c.bench_function(&id, |b| {
        b.iter(|| coder.encode_to(&vector, std::hint::black_box(&mut out)))
    });
}

fn float32_benchmarks(c: &mut Criterion) {
    for sim in VectorSimilarity::all() {
        benchmark_coding(c, F32VectorCoding::F32, Some(sim));
    }
}

fn float16_benchmarks(c: &mut Criterion) {
    for sim in VectorSimilarity::all() {
        benchmark_coding(c, F32VectorCoding::F16, Some(sim));
    }
}

fn i4_scaled_uniform_benchmarks(c: &mut Criterion) {
    benchmark_coding(c, F32VectorCoding::I4ScaledUniformQuantized, None);
}

fn i8_scaled_uniform_benchmarks(c: &mut Criterion) {
    benchmark_coding(c, F32VectorCoding::I8ScaledUniformQuantized, None);
}

fn i16_scaled_uniform_benchmarks(c: &mut Criterion) {
    benchmark_coding(c, F32VectorCoding::I16ScaledUniformQuantized, None);
}

fn binary_benchmarks(c: &mut Criterion) {
    benchmark_coding(c, F32VectorCoding::BinaryQuantized, None);
}

fn lvq1x1_benchmarks(c: &mut Criterion) {
    benchmark_coding(c, F32VectorCoding::LVQ1x1, None);
}

fn lvq1x4_benchmarks(c: &mut Criterion) {
    benchmark_coding(c, F32VectorCoding::LVQ1x4, None);
}

fn lvq1x8_benchmarks(c: &mut Criterion) {
    benchmark_coding(c, F32VectorCoding::LVQ1x8, None);
}

fn lvq2x1x8_benchmarks(c: &mut Criterion) {
    benchmark_coding(c, F32VectorCoding::LVQ2x1x8, None);
}

fn lvq2x4x4_benchmarks(c: &mut Criterion) {
    benchmark_coding(c, F32VectorCoding::LVQ2x4x4, None);
}

fn lvq2x4x8_benchmarks(c: &mut Criterion) {
    benchmark_coding(c, F32VectorCoding::LVQ2x4x8, None);
}

fn lvq2x8x8_benchmarks(c: &mut Criterion) {
    benchmark_coding(c, F32VectorCoding::LVQ2x8x8, None);
}

criterion_group!(
    benches,
    float32_benchmarks,
    float16_benchmarks,
    i4_scaled_uniform_benchmarks,
    i8_scaled_uniform_benchmarks,
    i16_scaled_uniform_benchmarks,
    binary_benchmarks,
    lvq1x1_benchmarks,
    lvq1x4_benchmarks,
    lvq1x8_benchmarks,
    lvq2x1x8_benchmarks,
    lvq2x4x4_benchmarks,
    lvq2x4x8_benchmarks,
    lvq2x8x8_benchmarks,
);
criterion_main!(benches);
