use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use half::f16;
use rand::{Rng, SeedableRng};
use vectors::{F32VectorCoding, VectorSimilarity};

const DIMENSIONS: usize = 2048;
const BULK_VECTORS: usize = 16;

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

pub fn float32_benchmarks(c: &mut Criterion) {
    let (a, b) = generate_test_vectors(DIMENSIONS);

    let mut group = c.benchmark_group("f32");
    group.throughput(Throughput::ElementsAndBytes {
        elements: 1,
        bytes: (a.len() * std::mem::size_of::<f32>()) as u64,
    });
    for sim in VectorSimilarity::all() {
        let coder = F32VectorCoding::F32.new_coder(sim);
        let x = coder.encode(&a);
        let y = coder.encode(&b);
        let dist = F32VectorCoding::F32.new_vector_distance(sim);
        group.bench_function(sim.to_string(), |b| {
            b.iter(|| std::hint::black_box(dist.distance(&x, &y)))
        });
    }
}

pub fn float16_benchmarks(c: &mut Criterion) {
    let (a, b) = generate_test_vectors(DIMENSIONS);

    let mut group = c.benchmark_group("f16");
    group.throughput(Throughput::ElementsAndBytes {
        elements: 1,
        bytes: (a.len() * std::mem::size_of::<f16>()) as u64,
    });

    for sim in VectorSimilarity::all() {
        let coder = F32VectorCoding::F16.new_coder(sim);
        let x = coder.encode(&a);
        let y = coder.encode(&b);
        let dist = F32VectorCoding::F16.new_vector_distance(sim);
        group.bench_function(&format!("doc/{sim}"), |b| {
            b.iter(|| std::hint::black_box(dist.distance(&x, &y)))
        });

        let query_dist = F32VectorCoding::F16.query_vector_distance_f32(&a, sim);
        group.bench_function(&format!("query/{sim}"), |b| {
            b.iter(|| std::hint::black_box(query_dist.distance(&y)))
        });
    }
}

pub fn quantized_normalized_benchmarks(c: &mut Criterion) {
    // Every type in this list is effectively l2 normalized at encoding time and uses a trivial
    // scalar transform of a dot product to calculate distance. For these we only compute dot
    // product similarity since other types are unlikely to vary all that much.
    let encodings = [
        F32VectorCoding::BinaryQuantized,
        F32VectorCoding::TLVQ1,
        F32VectorCoding::TLVQ2,
        F32VectorCoding::TLVQ4,
        F32VectorCoding::TLVQ8,
        F32VectorCoding::TLVQ1x8,
        F32VectorCoding::TLVQ2x8,
        F32VectorCoding::TLVQ4x8,
        F32VectorCoding::TLVQ8x8,
    ];
    let sim = VectorSimilarity::Dot;
    let (a, b) = generate_test_vectors(DIMENSIONS);

    for encoding in encodings {
        let coder = encoding.new_coder(sim);
        let x = coder.encode(&a);
        let y = coder.encode(&b);

        let mut group = c.benchmark_group(format!("{encoding}/distance"));
        group.throughput(Throughput::ElementsAndBytes {
            elements: 1,
            bytes: x.len() as u64,
        });
        let dist = encoding.new_vector_distance(sim);
        group.bench_function(&format!("doc/{sim}"), |b| {
            b.iter(|| std::hint::black_box(dist.distance(&x, &y)))
        });

        let query_dist = encoding.query_vector_distance_f32(&a, sim);
        group.bench_function(&format!("query/{sim}"), |b| {
            b.iter(|| std::hint::black_box(query_dist.distance(&y)))
        });
        drop(group);

        let mut group = c.benchmark_group(format!("{encoding}/bulk_distance"));
        group.throughput(Throughput::ElementsAndBytes {
            elements: BULK_VECTORS as u64,
            bytes: (x.len() * BULK_VECTORS) as u64,
        });
        let mut bulk_docs_storage = vec![];
        bulk_docs_storage.resize_with(BULK_VECTORS, || y.clone());
        let bulk_docs = bulk_docs_storage
            .iter()
            .map(|x| x.as_slice())
            .collect::<Vec<_>>();
        let mut bulk_out = vec![0.0; BULK_VECTORS];
        group.bench_function(&format!("doc/{sim}"), |b| {
            b.iter(|| {
                std::hint::black_box({
                    dist.bulk_distance(&x, &bulk_docs, &mut bulk_out);
                    bulk_out[0]
                })
            })
        });
        group.bench_function(&format!("query/{sim}"), |b| {
            b.iter(|| {
                std::hint::black_box({
                    query_dist.bulk_distance(&bulk_docs, &mut bulk_out);
                    bulk_out[0]
                })
            })
        });
    }
}

criterion_group!(
    benches,
    float32_benchmarks,
    float16_benchmarks,
    quantized_normalized_benchmarks,
);
criterion_main!(benches);
