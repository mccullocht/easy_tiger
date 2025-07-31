use criterion::{criterion_group, criterion_main, Criterion};
use easy_tiger::vectors::{F32VectorCoding, VectorSimilarity};
use rand::{Rng, SeedableRng};

fn generate_test_vectors(dim: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = rand_xoshiro::Xoroshiro128PlusPlus::seed_from_u64(0x455A_5469676572);
    let a = (&mut rng)
        .random_iter::<f32>()
        .by_ref()
        .take(dim)
        .collect::<Vec<_>>();
    let b = (&mut rng)
        .random_iter::<f32>()
        .take(dim)
        .collect::<Vec<_>>();
    (a, b)
}

fn benchmark_distance(
    name: &'static str,
    x: &[f32],
    y: &[f32],
    coding: F32VectorCoding,
    similarity: VectorSimilarity,
    c: &mut Criterion,
) {
    let coder = coding.new_coder();
    let x = coder.encode(x);
    let y = coder.encode(y);
    let dist = coding.new_symmetric_vector_distance(similarity).unwrap();
    c.bench_function(name, |b| {
        b.iter(|| std::hint::black_box(dist.distance(&x, &y)))
    });
}

pub fn float32_benchmarks(c: &mut Criterion) {
    let (a, b) = generate_test_vectors(1024);

    benchmark_distance(
        "f32/dot",
        &a,
        &b,
        F32VectorCoding::RawL2Normalized,
        VectorSimilarity::Dot,
        c,
    );
    benchmark_distance(
        "f32/l2",
        &a,
        &b,
        F32VectorCoding::Raw,
        VectorSimilarity::Euclidean,
        c,
    );
}

criterion_group!(benches, float32_benchmarks);
criterion_main!(benches);
