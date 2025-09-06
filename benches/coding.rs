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

fn scaled_uniform_benchmarks(c: &mut Criterion) {
    for format in [
        F32VectorCoding::I4ScaledUniformQuantized,
        F32VectorCoding::I8ScaledUniformQuantized,
        F32VectorCoding::I16ScaledUniformQuantized,
    ] {
        benchmark_coding(c, format, None);
    }
}

fn binary_benchmarks(c: &mut Criterion) {
    benchmark_coding(c, F32VectorCoding::BinaryQuantized, None);
}

fn lvq_benchmarks(c: &mut Criterion) {
    for format in [
        F32VectorCoding::LVQ1x1,
        F32VectorCoding::LVQ1x4,
        F32VectorCoding::LVQ1x8,
        F32VectorCoding::LVQ2x1x8,
        F32VectorCoding::LVQ2x1x12,
        F32VectorCoding::LVQ2x1x16,
        F32VectorCoding::LVQ2x4x4,
        F32VectorCoding::LVQ2x4x8,
        F32VectorCoding::LVQ2x8x8,
    ] {
        benchmark_coding(c, format, None);
    }
}

criterion_group!(
    benches,
    float32_benchmarks,
    float16_benchmarks,
    scaled_uniform_benchmarks,
    binary_benchmarks,
    lvq_benchmarks,
);
criterion_main!(benches);
