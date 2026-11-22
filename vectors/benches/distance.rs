use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use half::f16;
use rand::{RngExt, SeedableRng};
use vectors::rotate::{Kernel, Rotator};
use vectors::{F32VectorCoding, VectorSimilarity, int8, prepare_vector};

const SEED: u64 = 0x455A_5469676572;
const DIMENSIONS: usize = 2048;
const BULK_VECTORS: usize = 16;

fn generate_test_vectors(dim: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = rand_xoshiro::Xoroshiro128PlusPlus::seed_from_u64(SEED);
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

fn generate_test_vectors_i8(dim: usize) -> (Vec<i8>, Vec<i8>) {
    let mut rng = rand_xoshiro::Xoroshiro128PlusPlus::seed_from_u64(SEED);
    let a = (&mut rng).random_iter::<i8>().take(dim).collect::<Vec<_>>();
    let b = (&mut rng).random_iter::<i8>().take(dim).collect::<Vec<_>>();
    (a, b)
}

pub fn float32_benchmarks(c: &mut Criterion) {
    let (a, b) = generate_test_vectors(DIMENSIONS);

    let mut group = c.benchmark_group("f32/distance");
    group.throughput(Throughput::ElementsAndBytes {
        elements: 1,
        bytes: (a.len() * std::mem::size_of::<f32>()) as u64,
    });
    for sim in VectorSimilarity::all() {
        let coder = F32VectorCoding::F32.coder();
        let x = coder.encode(&a);
        let y = coder.encode(&b);
        let dist = F32VectorCoding::F32
            .distance_symmetric(sim)
            .expect("supports symmetric distance");
        group.bench_function(sim.to_string(), |b| {
            b.iter(|| std::hint::black_box(dist.distance(&x, &y)))
        });
    }
    drop(group);

    let mut group = c.benchmark_group("f32/bulk_distance");
    group.throughput(Throughput::ElementsAndBytes {
        elements: BULK_VECTORS as u64,
        bytes: (a.len() * std::mem::size_of::<f32>() * BULK_VECTORS) as u64,
    });
    for sim in VectorSimilarity::all() {
        let coder = F32VectorCoding::F32.coder();
        let x = coder.encode(&a);
        let y = coder.encode(&b);
        let mut bulk_docs_storage = vec![];
        bulk_docs_storage.resize_with(BULK_VECTORS, || y.clone());
        let bulk_docs = bulk_docs_storage
            .iter()
            .map(|x| x.as_slice())
            .collect::<Vec<_>>();
        let mut bulk_out = vec![0.0; BULK_VECTORS];
        let dist = F32VectorCoding::F32
            .distance_symmetric(sim)
            .expect("supports symmetric distance");
        group.bench_function(sim.to_string(), |b| {
            b.iter(|| {
                std::hint::black_box({
                    dist.bulk_distance(&x, &bulk_docs, &mut bulk_out);
                    bulk_out[0]
                })
            })
        });
    }
}

pub fn float16_benchmarks(c: &mut Criterion) {
    let (a, b) = generate_test_vectors(DIMENSIONS);

    let mut group = c.benchmark_group("f16/distance");
    group.throughput(Throughput::ElementsAndBytes {
        elements: 1,
        bytes: (a.len() * std::mem::size_of::<f16>()) as u64,
    });

    for sim in VectorSimilarity::all() {
        let coder = F32VectorCoding::F16.coder();
        let x = coder.encode(&a);
        let y = coder.encode(&b);
        let dist = F32VectorCoding::F16
            .distance_symmetric(sim)
            .expect("supports symmetric distance");
        group.bench_function(&format!("doc/{sim}"), |b| {
            b.iter(|| std::hint::black_box(dist.distance(&x, &y)))
        });

        let query_dist = F32VectorCoding::F16.query_distance_asymmetric(sim, &a);
        group.bench_function(&format!("query/{sim}"), |b| {
            b.iter(|| std::hint::black_box(query_dist.distance(&y)))
        });
    }
    drop(group);

    let mut group = c.benchmark_group("f16/bulk_distance");
    group.throughput(Throughput::ElementsAndBytes {
        elements: BULK_VECTORS as u64,
        bytes: (a.len() * std::mem::size_of::<f16>() * BULK_VECTORS) as u64,
    });

    for sim in VectorSimilarity::all() {
        let coder = F32VectorCoding::F16.coder();
        let x = coder.encode(&a);
        let y = coder.encode(&b);
        let mut bulk_docs_storage = vec![];
        bulk_docs_storage.resize_with(BULK_VECTORS, || y.clone());
        let bulk_docs = bulk_docs_storage
            .iter()
            .map(|x| x.as_slice())
            .collect::<Vec<_>>();
        let mut bulk_out = vec![0.0; BULK_VECTORS];

        let dist = F32VectorCoding::F16
            .distance_symmetric(sim)
            .expect("supports symmetric distance");
        group.bench_function(&format!("doc/{sim}"), |b| {
            b.iter(|| {
                std::hint::black_box({
                    dist.bulk_distance(&x, &bulk_docs, &mut bulk_out);
                    bulk_out[0]
                })
            })
        });

        let query_dist = F32VectorCoding::F16.query_distance_asymmetric(sim, &a);
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

pub fn quantized_normalized_benchmarks(c: &mut Criterion) {
    // Every type in this list is effectively l2 normalized at encoding time and uses a trivial
    // scalar transform of a dot product to calculate distance. For these we only compute dot
    // product similarity since other types are unlikely to vary all that much.
    let encodings = [
        F32VectorCoding::RaBitQ,
        F32VectorCoding::QuIVer,
        F32VectorCoding::TLVQ1,
        F32VectorCoding::TLVQ2,
        F32VectorCoding::TLVQ4,
        F32VectorCoding::TLVQ8,
    ];
    let sim = VectorSimilarity::Dot;
    let (a, b) = generate_test_vectors(DIMENSIONS);

    for encoding in encodings {
        let coder = encoding.coder();
        let x = coder.encode(&a);
        let y = coder.encode(&b);

        let mut group = c.benchmark_group(format!("{encoding}/distance"));
        group.throughput(Throughput::ElementsAndBytes {
            elements: 1,
            bytes: x.len() as u64,
        });
        let dist = encoding
            .distance_symmetric(sim)
            .expect("supports symmetric distance");
        group.bench_function(&format!("doc/{sim}"), |b| {
            b.iter(|| std::hint::black_box(dist.distance(&x, &y)))
        });

        let query_dist = encoding.query_distance_asymmetric(sim, &a);
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

fn turbo_quant_benchmarks(c: &mut Criterion) {
    // TurboQuant requires inputs to be randomly rotated before encoding, and only supports
    // asymmetric (query) distance. There is no symmetric "doc" distance to benchmark.
    let rotator = Rotator::with_kernel(DIMENSIONS, SEED, Kernel::default());
    let formats = [
        F32VectorCoding::TurboQuantM1,
        F32VectorCoding::TurboQuantM2,
        F32VectorCoding::TurboQuantM4,
        F32VectorCoding::TurboQuantM8,
    ];

    for encoding in formats {
        for sim in VectorSimilarity::all() {
            let (a, b) = generate_test_vectors(DIMENSIONS);
            let a = prepare_vector(a, Some(&rotator), sim.angular(), None);
            let b = prepare_vector(b, Some(&rotator), sim.angular(), None);

            let coder = encoding.coder();
            let y = coder.encode(&b);
            let query_dist = encoding.query_distance_asymmetric(sim, &a);

            let mut group = c.benchmark_group(format!("{encoding}/distance"));
            group.throughput(Throughput::ElementsAndBytes {
                elements: 1,
                bytes: y.len() as u64,
            });
            group.bench_function(format!("query/{sim}"), |b| {
                b.iter(|| std::hint::black_box(query_dist.distance(&y)))
            });
            drop(group);

            let mut group = c.benchmark_group(format!("{encoding}/bulk_distance"));
            group.throughput(Throughput::ElementsAndBytes {
                elements: BULK_VECTORS as u64,
                bytes: (y.len() * BULK_VECTORS) as u64,
            });
            let bulk_docs_storage = vec![y.clone(); BULK_VECTORS];
            let bulk_docs = bulk_docs_storage
                .iter()
                .map(|x| x.as_slice())
                .collect::<Vec<_>>();
            let mut bulk_out = vec![0.0; BULK_VECTORS];
            group.bench_function(format!("query/{sim}"), |b| {
                b.iter(|| {
                    std::hint::black_box({
                        query_dist.bulk_distance(&bulk_docs, &mut bulk_out);
                        bulk_out[0]
                    })
                })
            });
        }
    }
}

fn int8_benchmarks(c: &mut Criterion) {
    // i8 vectors are encoded from i8 input, and there is no asymmetric f32 query distance, so
    // query benchmarks use an encoded i8 query. Both similarities are benchmarked since they use
    // the stored norms differently.
    let (a, b) = generate_test_vectors_i8(DIMENSIONS);
    let coder = int8::VectorCoder::default();
    let x = coder.encode(&a);
    let y = coder.encode(&b);

    let mut group = c.benchmark_group("i8/distance");
    group.throughput(Throughput::ElementsAndBytes {
        elements: 1,
        bytes: y.len() as u64,
    });
    for sim in VectorSimilarity::all() {
        let dist = F32VectorCoding::I8
            .distance_symmetric(sim)
            .expect("supports symmetric distance");
        group.bench_function(format!("doc/{sim}"), |b| {
            b.iter(|| std::hint::black_box(dist.distance(&x, &y)))
        });

        let query_dist = F32VectorCoding::I8
            .query_distance_symmetric(sim, x.as_slice())
            .expect("supports symmetric distance");
        group.bench_function(format!("query/{sim}"), |b| {
            b.iter(|| std::hint::black_box(query_dist.distance(&y)))
        });
    }
    drop(group);

    let mut group = c.benchmark_group("i8/bulk_distance");
    group.throughput(Throughput::ElementsAndBytes {
        elements: BULK_VECTORS as u64,
        bytes: (y.len() * BULK_VECTORS) as u64,
    });
    let bulk_docs_storage = vec![y.clone(); BULK_VECTORS];
    let bulk_docs = bulk_docs_storage
        .iter()
        .map(|x| x.as_slice())
        .collect::<Vec<_>>();
    let mut bulk_out = vec![0.0; BULK_VECTORS];
    for sim in VectorSimilarity::all() {
        let dist = F32VectorCoding::I8
            .distance_symmetric(sim)
            .expect("supports symmetric distance");
        group.bench_function(format!("doc/{sim}"), |b| {
            b.iter(|| {
                std::hint::black_box({
                    dist.bulk_distance(&x, &bulk_docs, &mut bulk_out);
                    bulk_out[0]
                })
            })
        });

        let query_dist = F32VectorCoding::I8
            .query_distance_symmetric(sim, x.as_slice())
            .expect("supports symmetric distance");
        group.bench_function(format!("query/{sim}"), |b| {
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
    turbo_quant_benchmarks,
    int8_benchmarks,
);
criterion_main!(benches);
