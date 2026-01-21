use criterion::{Criterion, criterion_group, criterion_main};
use rand::{Rng, SeedableRng};
use vectors::{F32VectorCoding, VectorSimilarity};

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

    if similarity.unwrap_or(VectorSimilarity::Dot) == VectorSimilarity::Dot {
        let mut decoded = vec![0f32; vector.len()];
        c.bench_function(&format!("{format}/decode"), |b| {
            b.iter(|| coder.decode_to(&out, std::hint::black_box(&mut decoded)))
        });
    }
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

fn binary_benchmarks(c: &mut Criterion) {
    benchmark_coding(c, F32VectorCoding::BinaryQuantized, None);
}

fn lvq_benchmarks(c: &mut Criterion) {
    for format in [
        F32VectorCoding::TLVQ1,
        F32VectorCoding::TLVQ2,
        F32VectorCoding::TLVQ4,
        F32VectorCoding::TLVQ8,
        F32VectorCoding::TLVQ1x8,
        F32VectorCoding::TLVQ2x8,
        F32VectorCoding::TLVQ4x8,
        F32VectorCoding::TLVQ8x8,
    ] {
        benchmark_coding(c, format, None);
    }
}

criterion_group!(
    benches,
    float32_benchmarks,
    float16_benchmarks,
    binary_benchmarks,
    lvq_benchmarks,
);
criterion_main!(benches);
