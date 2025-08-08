use criterion::{criterion_group, criterion_main, Criterion};
use easy_tiger::vectors::{
    new_query_vector_distance_f32, F32VectorCoding, NonUniformQuantizedDimensions, VectorSimilarity,
};
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
    name: &str,
    x: &[f32],
    y: &[f32],
    coding: F32VectorCoding,
    similarity: VectorSimilarity,
    c: &mut Criterion,
) {
    let coder = coding.new_coder(similarity);
    let x = coder.encode(x);
    let y = coder.encode(y);
    let dist = coding.new_symmetric_vector_distance(similarity).unwrap();
    c.bench_function(name, |b| {
        b.iter(|| std::hint::black_box(dist.distance(&x, &y)))
    });
}

fn benchmark_query_distance(
    name: &str,
    x: &[f32],
    y: &[f32],
    coding: F32VectorCoding,
    similarity: VectorSimilarity,
    c: &mut Criterion,
) {
    let y = coding.new_coder(similarity).encode(y);
    let dist = new_query_vector_distance_f32(x, similarity, coding);
    c.bench_function(name, |b| b.iter(|| std::hint::black_box(dist.distance(&y))));
}

pub fn float32_benchmarks(c: &mut Criterion) {
    let (a, b) = generate_test_vectors(1024);

    for sim in VectorSimilarity::all() {
        benchmark_distance(&format!("f32/{sim}"), &a, &b, F32VectorCoding::F32, sim, c);
    }
}

pub fn float16_benchmarks(c: &mut Criterion) {
    query_and_doc_benchmarks(c, F32VectorCoding::F16);
}

pub fn i16_scaled_uniform_benchmarks(c: &mut Criterion) {
    query_and_doc_benchmarks(c, F32VectorCoding::I16ScaledUniformQuantized);
}

pub fn i8_scaled_uniform_benchmarks(c: &mut Criterion) {
    query_and_doc_benchmarks(c, F32VectorCoding::I8ScaledUniformQuantized);
}

pub fn i4_scaled_uniform_benchmarks(c: &mut Criterion) {
    query_and_doc_benchmarks(c, F32VectorCoding::I4ScaledUniformQuantized);
}

pub fn i8_scaled_non_uniform_benchmarks(c: &mut Criterion) {
    let format = F32VectorCoding::I8ScaledNonUniformQuantized(
        NonUniformQuantizedDimensions::try_from([256, 512].as_slice()).unwrap(),
    );
    query_and_doc_benchmarks(c, format);
}

fn query_and_doc_benchmarks(c: &mut Criterion, format: F32VectorCoding) {
    let (x, y) = generate_test_vectors(1024);
    for similarity in [VectorSimilarity::Dot, VectorSimilarity::Euclidean] {
        benchmark_distance(
            &format!("{format}/doc/{similarity}"),
            &x,
            &y,
            format,
            similarity,
            c,
        );
        benchmark_query_distance(
            &format!("{format}/query/{similarity}"),
            &x,
            &y,
            format,
            similarity,
            c,
        );
    }
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
    float16_benchmarks,
    i16_scaled_uniform_benchmarks,
    i8_scaled_uniform_benchmarks,
    i4_scaled_uniform_benchmarks,
    i8_scaled_non_uniform_benchmarks,
    u1_benchmarks,
);
criterion_main!(benches);
