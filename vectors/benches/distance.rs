use criterion::{Criterion, criterion_group, criterion_main};
use rand::{Rng, SeedableRng};
use vectors::{F32VectorCoding, VectorSimilarity};

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
    let dist = coding.query_vector_distance_f32(x, similarity);
    c.bench_function(name, |b| b.iter(|| std::hint::black_box(dist.distance(&y))));
}

pub fn float32_benchmarks(c: &mut Criterion) {
    let (a, b) = generate_test_vectors(1024);

    for sim in VectorSimilarity::all() {
        benchmark_distance(&format!("f32/{sim}"), &a, &b, F32VectorCoding::F32, sim, c);
    }
}

fn query_and_doc_benchmarks<I>(c: &mut Criterion, format: F32VectorCoding, similarities: I)
where
    I: IntoIterator<Item = VectorSimilarity>,
{
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

fn lvq_benchmarks(c: &mut Criterion) {
    // Regardless of the similarity type all of the implementations use dot product internally and
    // then adjust using stored hyper parameters.
    let similarities = [VectorSimilarity::Dot];
    query_and_doc_benchmarks(c, F32VectorCoding::LVQ2x1x8, similarities);
    query_and_doc_benchmarks(c, F32VectorCoding::LVQ2x4x4, similarities);
    query_and_doc_benchmarks(c, F32VectorCoding::LVQ2x4x8, similarities);
    query_and_doc_benchmarks(c, F32VectorCoding::LVQ2x8x8, similarities);
    query_and_doc_benchmarks(c, F32VectorCoding::TLVQ1, similarities);
    query_and_doc_benchmarks(c, F32VectorCoding::TLVQ2, similarities);
    query_and_doc_benchmarks(c, F32VectorCoding::TLVQ4, similarities);
    query_and_doc_benchmarks(c, F32VectorCoding::TLVQ8, similarities);
    query_and_doc_benchmarks(c, F32VectorCoding::TLVQ1x8, similarities);
    query_and_doc_benchmarks(c, F32VectorCoding::TLVQ2x8, similarities);
    query_and_doc_benchmarks(c, F32VectorCoding::TLVQ4x8, similarities);
    query_and_doc_benchmarks(c, F32VectorCoding::TLVQ8x8, similarities);
}

criterion_group!(
    benches,
    float32_benchmarks,
    float16_benchmarks,
    i1_benchmarks,
    lvq_benchmarks,
);
criterion_main!(benches);
