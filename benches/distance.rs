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
    let dist = coding.new_vector_distance(similarity);
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

fn query_and_doc_benchmarks(
    c: &mut Criterion,
    format: F32VectorCoding,
    similarities: impl ExactSizeIterator<Item = VectorSimilarity>,
) {
    let (x, y) = generate_test_vectors(1024);
    for similarity in similarities {
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

pub fn float16_benchmarks(c: &mut Criterion) {
    query_and_doc_benchmarks(c, F32VectorCoding::F16, VectorSimilarity::all());
}

pub fn i16_scaled_uniform_benchmarks(c: &mut Criterion) {
    query_and_doc_benchmarks(
        c,
        F32VectorCoding::I16ScaledUniformQuantized,
        VectorSimilarity::all(),
    );
}

pub fn i8_scaled_uniform_benchmarks(c: &mut Criterion) {
    query_and_doc_benchmarks(
        c,
        F32VectorCoding::I8ScaledUniformQuantized,
        VectorSimilarity::all(),
    );
}

pub fn i4_scaled_uniform_benchmarks(c: &mut Criterion) {
    query_and_doc_benchmarks(
        c,
        F32VectorCoding::I4ScaledUniformQuantized,
        VectorSimilarity::all(),
    );
}

pub fn i8_scaled_non_uniform_benchmarks(c: &mut Criterion) {
    let format = F32VectorCoding::I8ScaledNonUniformQuantized(
        NonUniformQuantizedDimensions::try_from([256, 512].as_slice()).unwrap(),
    );
    query_and_doc_benchmarks(c, format, VectorSimilarity::all());
}

pub fn i1_benchmarks(c: &mut Criterion) {
    let (x, y) = generate_test_vectors(1024);

    benchmark_distance(
        "i1/hamming",
        &x,
        &y,
        F32VectorCoding::BinaryQuantized,
        VectorSimilarity::Dot, // doesn't really matter here
        c,
    );
    benchmark_query_distance(
        "i1/queryi8",
        &x,
        &y,
        F32VectorCoding::BinaryQuantized,
        VectorSimilarity::Dot,
        c,
    );
}

fn angular_similarities() -> impl ExactSizeIterator<Item = VectorSimilarity> {
    [VectorSimilarity::Dot, VectorSimilarity::Cosine].into_iter()
}

fn lvq2x8x8_benchmarks(c: &mut Criterion) {
    query_and_doc_benchmarks(c, F32VectorCoding::LVQ2x8x8, angular_similarities());
}

fn lvq2x4x8_benchmarks(c: &mut Criterion) {
    query_and_doc_benchmarks(c, F32VectorCoding::LVQ2x4x8, angular_similarities());
}

fn lvq2x4x4_benchmarks(c: &mut Criterion) {
    query_and_doc_benchmarks(c, F32VectorCoding::LVQ2x4x4, angular_similarities());
}

fn lvq2x1x8_benchmarks(c: &mut Criterion) {
    query_and_doc_benchmarks(c, F32VectorCoding::LVQ2x1x8, angular_similarities());
}

fn lvq1x8_benchmarks(c: &mut Criterion) {
    query_and_doc_benchmarks(c, F32VectorCoding::LVQ1x8, angular_similarities());
}

fn lvq1x4_benchmarks(c: &mut Criterion) {
    query_and_doc_benchmarks(c, F32VectorCoding::LVQ1x4, angular_similarities());
}

fn lvq1x1_benchmarks(c: &mut Criterion) {
    query_and_doc_benchmarks(c, F32VectorCoding::LVQ1x1, angular_similarities());
}

criterion_group!(
    benches,
    float32_benchmarks,
    float16_benchmarks,
    i16_scaled_uniform_benchmarks,
    i8_scaled_uniform_benchmarks,
    i4_scaled_uniform_benchmarks,
    i8_scaled_non_uniform_benchmarks,
    i1_benchmarks,
    lvq2x8x8_benchmarks,
    lvq2x4x8_benchmarks,
    lvq2x4x4_benchmarks,
    lvq2x1x8_benchmarks,
    lvq1x8_benchmarks,
    lvq1x4_benchmarks,
    lvq1x1_benchmarks,
);
criterion_main!(benches);
