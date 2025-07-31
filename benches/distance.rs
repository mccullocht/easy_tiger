use criterion::{criterion_group, criterion_main, Criterion};
use easy_tiger::vectors::{new_query_vector_distance_f32, F32VectorCoding, VectorSimilarity};
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

fn benchmark_query_distance(
    name: &'static str,
    x: &[f32],
    y: &[f32],
    coding: F32VectorCoding,
    similarity: VectorSimilarity,
    c: &mut Criterion,
) {
    let y = coding.new_coder().encode(y);
    let dist = new_query_vector_distance_f32(x, similarity, coding);
    c.bench_function(name, |b| b.iter(|| std::hint::black_box(dist.distance(&y))));
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

pub fn i8_scaled_uniform_benchmarks(c: &mut Criterion) {
    let (x, y) = generate_test_vectors(1024);

    benchmark_distance(
        "i8-scaled-uniform/doc/dot",
        &x,
        &y,
        F32VectorCoding::I8ScaledUniformQuantized,
        VectorSimilarity::Dot,
        c,
    );
    benchmark_distance(
        "i8-scaled-uniform/doc/l2",
        &x,
        &y,
        F32VectorCoding::I8ScaledUniformQuantized,
        VectorSimilarity::Euclidean,
        c,
    );

    benchmark_query_distance(
        "i8-scaled-uniform/query/dot",
        &x,
        &y,
        F32VectorCoding::I8ScaledUniformQuantized,
        VectorSimilarity::Dot,
        c,
    );
    benchmark_query_distance(
        "i8-scaled-uniform/query/l2",
        &x,
        &y,
        F32VectorCoding::I8ScaledUniformQuantized,
        VectorSimilarity::Euclidean,
        c,
    );
}

pub fn i4_scaled_uniform_benchmarks(c: &mut Criterion) {
    let (x, y) = generate_test_vectors(1024);

    benchmark_distance(
        "i4-scaled-uniform/doc/dot",
        &x,
        &y,
        F32VectorCoding::I4ScaledUniformQuantized,
        VectorSimilarity::Dot,
        c,
    );
    benchmark_distance(
        "i4-scaled-uniform/doc/l2",
        &x,
        &y,
        F32VectorCoding::I4ScaledUniformQuantized,
        VectorSimilarity::Euclidean,
        c,
    );

    benchmark_query_distance(
        "i4-scaled-uniform/query/dot",
        &x,
        &y,
        F32VectorCoding::I4ScaledUniformQuantized,
        VectorSimilarity::Dot,
        c,
    );
    benchmark_query_distance(
        "i4-scaled-uniform/query/l2",
        &x,
        &y,
        F32VectorCoding::I4ScaledUniformQuantized,
        VectorSimilarity::Euclidean,
        c,
    );
}

pub fn u1_benchmarks(c: &mut Criterion) {
    let (x, y) = generate_test_vectors(1024);

    benchmark_distance(
        "u1/hamming",
        &x,
        &y,
        F32VectorCoding::BinaryQuantized,
        VectorSimilarity::Dot, // doesn't really matter here
        c,
    );
}

criterion_group!(
    benches,
    float32_benchmarks,
    i8_scaled_uniform_benchmarks,
    i4_scaled_uniform_benchmarks,
    u1_benchmarks,
);
criterion_main!(benches);
