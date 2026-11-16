use criterion::{Criterion, criterion_group, criterion_main};
use rand::{RngExt, SeedableRng};
use vectors::rotate::{Kernel, Rotator};
use vectors::{F32VectorCoding, VectorSimilarity, prepare_vector};

const SEED: u64 = 0x455A_5469676572;
const DIMENSIONS: usize = 1024;

fn generate_test_vector(dim: usize) -> Vec<f32> {
    let rng = rand_xoshiro::Xoroshiro128PlusPlus::seed_from_u64(SEED);
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
    benchmark_coding_vector(c, format, &generate_test_vector(DIMENSIONS), similarity)
}

fn benchmark_coding_vector(
    c: &mut Criterion,
    format: F32VectorCoding,
    vector: &[f32],
    similarity: Option<VectorSimilarity>,
) {
    let coder = format.coder();
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
    benchmark_coding(c, F32VectorCoding::RaBitQ, None);
    benchmark_coding(c, F32VectorCoding::QuIVer, None);
}

fn lvq_benchmarks(c: &mut Criterion) {
    for format in [
        F32VectorCoding::TLVQ1,
        F32VectorCoding::TLVQ2,
        F32VectorCoding::TLVQ4,
        F32VectorCoding::TLVQ8,
    ] {
        benchmark_coding(c, format, None);
    }
}

fn turbo_quant_benchmarks(c: &mut Criterion) {
    // TurboQuant assumes that vectors have been randomly rotated and, for angular distance,
    // normalized before encoding. Rotation is benchmarked separately below.
    let rotator = Rotator::with_kernel(DIMENSIONS, SEED, Kernel::default());
    let vector = prepare_vector(generate_test_vector(DIMENSIONS), Some(&rotator), true, None);
    for format in [
        F32VectorCoding::TurboQuantM1,
        F32VectorCoding::TurboQuantM2,
        F32VectorCoding::TurboQuantM3,
        F32VectorCoding::TurboQuantM4,
        F32VectorCoding::TurboQuantM8,
    ] {
        benchmark_coding_vector(c, format, &vector, None);
    }
}

fn rotator_benchmarks(c: &mut Criterion) {
    for kernel in Kernel::all() {
        for dim in [512usize, 768, 1024, 1536, 2048] {
            let vector = generate_test_vector(dim);
            let rotator = Rotator::with_kernel(dim, SEED, kernel);
            c.bench_function(&format!("rotate/{kernel}/forward/{dim}"), |b| {
                b.iter(|| rotator.rotate_copy(std::hint::black_box(&vector)))
            });
        }
    }
}

criterion_group!(
    benches,
    float32_benchmarks,
    float16_benchmarks,
    binary_benchmarks,
    lvq_benchmarks,
    turbo_quant_benchmarks,
    rotator_benchmarks,
);
criterion_main!(benches);
