use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use easy_tiger::token::{
    CentroidParentFilterDocIterator, DOC_ITER_SENTINEL, DocFormat, DocIterator,
    DocIteratorBlockMut, PayloadFormat, TermDocIterator, doc_block_key,
};
use rand::prelude::*;
use wt_mdb::{
    Connection,
    connection::{CreateOptionsBuilder, OptionsBuilder},
};

const TABLE: &str = "postings";
const FIELD: &str = "f";
const ASSIGNMENT_TERM: &[u8] = b"assignments";
const NUM_DOCS: usize = 1_000_000;
const DOCS_PER_CENTROID: usize = 700;
const MIN_CENTROID_LEN: usize = 100;
const MAX_CENTROID_LEN: usize = 1000;
const BLOCK_LEN: usize = 128;
const INPUT_LENS: &[usize] = &[1_000, 10_000, 100_000, 1_000_000];

/// Generate a centroid assignment for each of [`NUM_DOCS`] docs.
///
/// Centroid lengths are drawn from a triangular distribution on
/// [`MIN_CENTROID_LEN`, `MAX_CENTROID_LEN`] with mode `MAX_CENTROID_LEN`, which has a mean of
/// [`DOCS_PER_CENTROID`]. Lengths are then nudged so that they sum to [`NUM_DOCS`] and docs are
/// randomly assigned to centroids.
fn generate_assignments(rng: &mut impl Rng) -> (Vec<u64>, u32) {
    let num_centroids = NUM_DOCS / DOCS_PER_CENTROID;
    let range = (MAX_CENTROID_LEN - MIN_CENTROID_LEN) as f64;
    let mut lens: Vec<usize> = (0..num_centroids)
        .map(|_| MIN_CENTROID_LEN + (range * rng.random::<f64>().sqrt()) as usize)
        .collect();
    let mut total: usize = lens.iter().sum();
    while total != NUM_DOCS {
        let len = &mut lens[rng.random_range(0..num_centroids)];
        if total < NUM_DOCS && *len < MAX_CENTROID_LEN {
            *len += 1;
            total += 1;
        } else if total > NUM_DOCS && *len > MIN_CENTROID_LEN {
            *len -= 1;
            total -= 1;
        }
    }

    let mut assignments: Vec<u64> = lens
        .iter()
        .enumerate()
        .flat_map(|(c, &len)| std::iter::repeat_n(c as u64, len))
        .collect();
    assignments.shuffle(rng);
    (assignments, num_centroids as u32 - 1)
}

/// Write the posting for `term` to `cursor` in blocks of [`BLOCK_LEN`] entries.
fn write_posting(
    cursor: &mut wt_mdb::IndexCursorGuard<'_>,
    term: &[u8],
    entries: impl ExactSizeIterator<Item = (i64, u64)>,
    payload_format: PayloadFormat,
) {
    let num_blocks = entries.len().div_ceil(BLOCK_LEN);
    let mut entries = entries.peekable();
    for i in 0..num_blocks {
        let mut block = DocIteratorBlockMut::new(DocFormat::LEB128, payload_format);
        let mut limit = DOC_ITER_SENTINEL;
        for (doc, payload) in entries.by_ref().take(BLOCK_LEN) {
            block.insert(doc, payload);
            limit = doc;
        }
        if i + 1 == num_blocks {
            limit = DOC_ITER_SENTINEL;
        }
        cursor
            .set(&doc_block_key(FIELD, term, limit), &block.serialize())
            .unwrap();
    }
}

fn input_term(len: usize) -> Vec<u8> {
    format!("input{len}").into_bytes()
}

fn centroid_parent_filter_benchmarks(c: &mut Criterion) {
    let dir = tempfile::TempDir::new().unwrap();
    let conn: Arc<Connection> = Connection::open(
        dir.path().to_str().unwrap(),
        Some(OptionsBuilder::default().create().into()),
    )
    .unwrap();
    conn.create_table(
        TABLE,
        Some(
            CreateOptionsBuilder::default()
                .key_format::<Vec<u8>>()
                .value_format::<Vec<u8>>()
                .into(),
        ),
    )
    .unwrap();

    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0x455A_5469676572);
    let (assignments, max_centroid_id) = generate_assignments(&mut rng);
    {
        let txn = conn.begin_transaction(None).unwrap();
        {
            let mut cursor = txn.open_index_cursor(TABLE).unwrap();
            write_posting(
                &mut cursor,
                ASSIGNMENT_TERM,
                assignments.iter().enumerate().map(|(d, &c)| (d as i64, c)),
                PayloadFormat::LEB128,
            );
            for &len in INPUT_LENS {
                let mut docs = rand::seq::index::sample(&mut rng, NUM_DOCS, len).into_vec();
                docs.sort_unstable();
                write_posting(
                    &mut cursor,
                    &input_term(len),
                    docs.into_iter().map(|d| (d as i64, 0)),
                    PayloadFormat::None,
                );
            }
        }
        txn.commit(None).unwrap();
    }

    let txn = conn.begin_transaction(None).unwrap();
    let open_iterators = |term: &[u8], len: usize| {
        (
            TermDocIterator::new(&txn, TABLE, FIELD, term, len as u64).unwrap(),
            TermDocIterator::new(&txn, TABLE, FIELD, ASSIGNMENT_TERM, NUM_DOCS as u64).unwrap(),
        )
    };
    let mut group = c.benchmark_group("centroid_filter");
    for &len in INPUT_LENS {
        let term = input_term(len);
        group.throughput(Throughput::Elements(len as u64));
        group.bench_function(BenchmarkId::new("parent_filter", len), |b| {
            b.iter(|| {
                let (input, assignments) = open_iterators(&term, len);
                let mut it = CentroidParentFilterDocIterator::new(
                    Box::new(input),
                    Box::new(assignments),
                    max_centroid_id,
                    max_centroid_id + 1,
                );
                let mut count = 0usize;
                while it.doc() != DOC_ITER_SENTINEL {
                    count += 1;
                    it.advance();
                }
                it.error().unwrap();
                std::hint::black_box(count)
            })
        });
        group.bench_function(BenchmarkId::new("hit_counts", len), |b| {
            b.iter(|| {
                let (mut input, mut assignments) = open_iterators(&term, len);
                std::hint::black_box(centroid_hit_counts(
                    &mut input,
                    &mut assignments,
                    max_centroid_id,
                ))
            })
        });
    }
    group.finish();
}

/// Count the number of docs in `input` assigned to each centroid in `assignments`.
///
/// This is an alternative to [`CentroidParentFilterDocIterator`] that visits every input doc
/// rather than yielding one doc per centroid.
fn centroid_hit_counts(
    input: &mut dyn DocIterator,
    assignments: &mut dyn DocIterator,
    max_centroid_id: u32,
) -> Vec<u32> {
    let mut counts = vec![0u32; max_centroid_id as usize + 1];
    let mut doc = input.doc();
    while doc != DOC_ITER_SENTINEL {
        let mut assigned = assignments.doc();
        if assigned < doc {
            assigned = assignments.advance_to(doc);
        }
        if assigned == doc {
            counts[assignments.payload() as usize] += 1;
            doc = input.advance();
        } else {
            // No assignment for doc; skip to the next doc that has one.
            doc = input.advance_to(assigned);
        }
    }
    input.error().unwrap();
    assignments.error().unwrap();
    counts
}

criterion_group!(benches, centroid_parent_filter_benchmarks);
criterion_main!(benches);
