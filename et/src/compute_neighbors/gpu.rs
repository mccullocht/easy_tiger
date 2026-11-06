use std::{
    io,
    ops::Range,
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

use bytemuck::{Pod, Zeroable};
use easy_tiger::{
    Neighbor,
    input::{DerefVectorStore, VectorStore},
};
use memmap2::Mmap;
use rayon::prelude::*;
use vectors::{VectorSimilarity, f16};

use crate::{neighbor_util::TopNeighbors, ui::progress_bar};

use super::{ComputeNeighborsArgs, write_neighbors};

/// WGSL compute shader for exact pairwise distance computation with on-GPU candidate filtering.
///
/// Vectors are uploaded as f16 with each row zero-padded to a multiple of `DIM_TILE` dims (zero
/// padding changes neither dot product nor L2 distance) and read as `u32` f16 pairs, so the
/// shader needs no `f16` support. All arithmetic is carried out in f32.
///
/// Each 16x16 workgroup computes a 64 query x 64 doc block of distances; each thread owns a 4x4
/// sub-block held in registers. Per stage of `DIM_TILE` dims the workgroup stages the block's
/// query and doc rows into workgroup memory, converted to f32 once and transposed so that a
/// thread reads its 4 query values and 4 doc values for one dim as two `vec4` loads that then
/// feed 16 FMAs. The next stage's global loads are issued before the current stage's math so
/// their latency is hidden.
///
/// Rather than writing the full distance matrix, a pair is only emitted when its distance is no
/// greater than `thresholds[q]`, the current k-th smallest distance known for that query (an
/// upper bound on the final k-th distance, so no true neighbor is ever dropped). Emitted pairs
/// are appended to `candidates` through an atomic counter; the counter keeps counting past the
/// buffer's capacity so the host can detect overflow and recompute that range.
///
/// Similarity:
///   Euclidean: squared L2.
///   Dot: (1 - dot) / 2; assumes pre-normalized vectors.
const SHADER: &str = r#"
struct Params {
    query_count: u32,
    doc_count: u32,
    q_offset: u32,
    d_offset: u32,
}

struct Candidate {
    query: u32,
    doc: u32,
    distance: f32,
}

struct Candidates {
    count: atomic<u32>,
    items: array<Candidate>,
}

@group(0) @binding(0) var<uniform> params: Params;
// Each u32 holds two f16s (little-endian: low half is the even element).
@group(0) @binding(1) var<storage, read> query_vectors: array<u32>;
@group(0) @binding(2) var<storage, read> doc_vectors: array<u32>;
@group(0) @binding(3) var<storage, read> thresholds: array<f32>;
@group(0) @binding(4) var<storage, read_write> candidates: Candidates;

// u32s per (padded) row; baked in at runtime so loop bounds are compile-time constants.
const ROW_U32: u32 = __ROW_U32__u;
// Dims staged in workgroup memory per stage, and the matching number of u32s per row.
const TK: u32 = __DIM_TILE__u;
const TK_U32: u32 = TK / 2u;
const STAGES: u32 = ROW_U32 / TK_U32;
// Rows per workgroup block on each axis (16 threads x 4 rows per thread).
const TQ: u32 = 64u;
const TD: u32 = 64u;
// vec4 stride between dims in the staged tiles: 16 vec4s (64 rows) + 1 to spread LDS banks.
const S4: u32 = 17u;

// Staged tiles, transposed: element [k * S4 + g] holds rows 4g..4g+3 at dim k of the stage.
var<workgroup> q_s: array<vec4<f32>, TK * S4>;
var<workgroup> d_s: array<vec4<f32>, TK * S4>;

// Load this thread's share of one stage: one u32 (2 dims) from each of 4 consecutive rows.
// Rows past the end of the batch re-read the last valid row; their results are discarded.
fn fetch_stage(stage: u32, staging_query: bool, row0: u32, last: u32, col: u32) -> vec4<u32> {
    let off = stage * TK_U32 + col;
    let r0 = min(row0, last) * ROW_U32 + off;
    let r1 = min(row0 + 1u, last) * ROW_U32 + off;
    let r2 = min(row0 + 2u, last) * ROW_U32 + off;
    let r3 = min(row0 + 3u, last) * ROW_U32 + off;
    if staging_query {
        return vec4<u32>(query_vectors[r0], query_vectors[r1], query_vectors[r2], query_vectors[r3]);
    }
    return vec4<u32>(doc_vectors[r0], doc_vectors[r1], doc_vectors[r2], doc_vectors[r3]);
}

// Convert a fetched 4 row x 2 dim block to f32 and store it transposed (one vec4 per dim).
fn store_stage(v: vec4<u32>, staging_query: bool, col: u32, rg: u32) {
    let a = unpack2x16float(v.x);
    let b = unpack2x16float(v.y);
    let c = unpack2x16float(v.z);
    let d = unpack2x16float(v.w);
    let i0 = 2u * col * S4 + rg;
    let i1 = i0 + S4;
    if staging_query {
        q_s[i0] = vec4<f32>(a.x, b.x, c.x, d.x);
        q_s[i1] = vec4<f32>(a.y, b.y, c.y, d.y);
    } else {
        d_s[i0] = vec4<f32>(a.x, b.x, c.x, d.x);
        d_s[i1] = vec4<f32>(a.y, b.y, c.y, d.y);
    }
}

fn to_distance(acc: vec4<f32>) -> vec4<f32> {
    return __DISTANCE__;
}

// Append each pair in a 1 query x 4 doc row that passes the query's threshold.
fn emit_row(q: u32, d0: u32, acc: vec4<f32>) {
    if q >= params.query_count {
        return;
    }
    let threshold = thresholds[q];
    let dist = to_distance(acc);
    let capacity = arrayLength(&candidates.items);
    for (var j = 0u; j < 4u; j++) {
        let d = d0 + j;
        if d < params.doc_count && dist[j] <= threshold {
            let idx = atomicAdd(&candidates.count, 1u);
            if idx < capacity {
                candidates.items[idx] = Candidate(q, d, dist[j]);
            }
        }
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(local_invocation_id) lid3: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let qx = lid3.x;
    let dy = lid3.y;
    let qt0 = params.q_offset + wid.x * TQ;
    let dt0 = params.d_offset + wid.y * TD;

    // Staging assignment: threads 0..127 stage query rows and 128..255 stage doc rows (a
    // wave-uniform split). Within each half, 8 consecutive threads read 8 consecutive u32s of
    // the same row (coalesced), and each thread covers 4 consecutive rows.
    let staging_query = lid < 128u;
    let t = lid & 127u;
    let col = t & 7u;
    let rg = t >> 3u;
    let row0 = select(dt0, qt0, staging_query) + rg * 4u;
    let last = select(params.doc_count, params.query_count, staging_query) - 1u;

    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    var next = fetch_stage(0u, staging_query, row0, last, col);
    for (var s = 0u; s < STAGES; s++) {
        workgroupBarrier();
        store_stage(next, staging_query, col, rg);
        workgroupBarrier();
        if s + 1u < STAGES {
            next = fetch_stage(s + 1u, staging_query, row0, last, col);
        }
        for (var k = 0u; k < TK; k++) {
            let a = q_s[k * S4 + qx];
            let b = d_s[k * S4 + dy];
__ACCUM__
        }
    }

    let q0 = qt0 + qx * 4u;
    let d0 = dt0 + dy * 4u;
    emit_row(q0, d0, acc0);
    emit_row(q0 + 1u, d0, acc1);
    emit_row(q0 + 2u, d0, acc2);
    emit_row(q0 + 3u, d0, acc3);
}
"#;

const DOT_ACCUM: &str = r#"            acc0 = fma(vec4<f32>(a.x), b, acc0);
            acc1 = fma(vec4<f32>(a.y), b, acc1);
            acc2 = fma(vec4<f32>(a.z), b, acc2);
            acc3 = fma(vec4<f32>(a.w), b, acc3);"#;
const DOT_DISTANCE: &str = "(vec4<f32>(1.0) - acc) * 0.5";

const EUCLID_ACCUM: &str = r#"            let e0 = vec4<f32>(a.x) - b;
            let e1 = vec4<f32>(a.y) - b;
            let e2 = vec4<f32>(a.z) - b;
            let e3 = vec4<f32>(a.w) - b;
            acc0 = fma(e0, e0, acc0);
            acc1 = fma(e1, e1, acc1);
            acc2 = fma(e2, e2, acc2);
            acc3 = fma(e3, e3, acc3);"#;
const EUCLID_DISTANCE: &str = "acc";

/// Dims staged per stage in the shader. Rows are zero-padded to a multiple of this. With the
/// 64 + 64 staged rows (padded to 17 vec4s per dim) this is 2 * 16 * 17 * 16 B = ~8.5 KiB of
/// workgroup memory, small enough that several workgroups stay resident per core on hardware
/// with 32 KiB of threadgroup memory.
const DIM_TILE: usize = 16;

/// Queries and docs covered by one workgroup.
const WG_TILE_Q: usize = 64;
const WG_TILE_D: usize = 64;

/// Target (query, doc) pairs per batch. Each batch is one submission and one readback, and
/// thresholds are refreshed between batches; this keeps batches large enough to amortize the
/// per-submission cost without making threshold refreshes too infrequent.
const TARGET_BATCH_PAIRS: usize = 1 << 29;

/// Upper bound on scalar multiply-add operations (~query * doc * dims) issued by a single
/// dispatch. Unbounded dispatches can run long enough to trip a driver watchdog (e.g. Windows
/// TDR, ~2s), which kills the device. At even 1 TFLOPS a dispatch of this size finishes in
/// ~35ms, well within the watchdog, while being large enough (16M pairs at 1024 dims) that each
/// dispatch keeps the whole GPU busy; dispatches within a pass are separated by barriers since
/// they all append to the candidate buffer, so small dispatches would leave the GPU draining.
const MAX_DISPATCH_OPS: usize = 1 << 34;

/// Lower bound on candidate buffer capacity.
const MIN_CANDIDATE_CAPACITY: usize = 1 << 20;
/// Upper bound on the candidate buffer size in bytes (there are three buffers this size).
const MAX_CANDIDATE_BYTES: usize = 256 << 20;

/// Instantiate the shader template for this run's row width and similarity function.
fn build_shader_source(padded_dims: usize, similarity: VectorSimilarity) -> String {
    let (accum, distance) = match similarity {
        VectorSimilarity::Euclidean => (EUCLID_ACCUM, EUCLID_DISTANCE),
        VectorSimilarity::Dot => (DOT_ACCUM, DOT_DISTANCE),
    };
    SHADER
        .replace("__ROW_U32__", &(padded_dims / 2).to_string())
        .replace("__DIM_TILE__", &DIM_TILE.to_string())
        .replace("__ACCUM__", accum)
        .replace("__DISTANCE__", distance)
}

/// Uniform buffer layout for the distance shader.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuParams {
    query_count: u32,
    doc_count: u32,
    q_offset: u32,
    d_offset: u32,
}

/// A candidate pair emitted by the shader. `query` and `doc` are relative to the batch start.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Candidate {
    query: u32,
    doc: u32,
    distance: f32,
}

/// Size of the candidate count header that precedes the candidates in the candidate buffer.
const CANDIDATE_HEADER_BYTES: usize = std::mem::size_of::<u32>();

/// Per-phase wall-clock totals across all batches, in microseconds. Phase attribution per batch
/// cycle:
///   - `upload`:   main thread copying vector bytes and thresholds into queue staging
///   - `submit`:   main thread recording/submitting the sub-dispatches and result copy
///   - `wait`:     main thread blocked waiting for the batch's result copy
///   - `accumulate`: rayon feeding candidates into the top-k accumulators
///
/// Batches are pipelined two-deep, so `accumulate` overlaps GPU work on the next batch; `wait`
/// is the time the GPU is the bottleneck. `overflows` counts batches whose candidates did not
/// fit and were recomputed in halves; a few are expected at the start of each query batch,
/// before thresholds are established.
#[derive(Default)]
struct BatchTimings {
    upload_us: AtomicU64,
    submit_us: AtomicU64,
    wait_us: AtomicU64,
    accumulate_us: AtomicU64,
    batches: AtomicU64,
    sub_dispatches: AtomicU64,
    candidates: AtomicU64,
    overflows: AtomicU64,
}

impl BatchTimings {
    fn report(&self, label: &str) {
        let batches = self.batches.load(Ordering::Relaxed);
        if batches == 0 {
            return;
        }
        let us = |v: &AtomicU64| v.load(Ordering::Relaxed) / batches;
        tracing::info!(
            "{}: {} batches ({} overflowed), {} sub-dispatches; per batch: upload {} ms, submit {} ms, wait {} ms, accumulate {} ms, candidates {}",
            label,
            batches,
            self.overflows.load(Ordering::Relaxed),
            self.sub_dispatches.load(Ordering::Relaxed),
            us(&self.upload_us) as f64 / 1e3,
            us(&self.submit_us) as f64 / 1e3,
            us(&self.wait_us) as f64 / 1e3,
            us(&self.accumulate_us) as f64 / 1e3,
            self.candidates.load(Ordering::Relaxed) / batches,
        );
    }
}

/// A (query, doc) range scheduled for computation.
#[derive(Clone, Copy, PartialEq, Eq)]
struct BatchRange {
    q_start: usize,
    q_end: usize,
    d_start: usize,
    d_end: usize,
}

impl BatchRange {
    fn queries(&self) -> Range<usize> {
        self.q_start..self.q_end
    }

    fn docs(&self) -> Range<usize> {
        self.d_start..self.d_end
    }

    fn pairs(&self) -> u64 {
        (self.queries().len() * self.docs().len()) as u64
    }

    /// Split the doc range in half, or `None` if it covers a single doc.
    fn split(&self) -> Option<(Self, Self)> {
        let len = self.docs().len();
        if len < 2 {
            return None;
        }
        let mid = self.d_start + len / 2;
        Some((
            Self {
                d_end: mid,
                ..*self
            },
            Self {
                d_start: mid,
                ..*self
            },
        ))
    }
}

/// A batch whose compute and result copy have been submitted to the GPU and which is awaiting
/// CPU-side consumption. Batches are pipelined two-deep; `slot` selects which of the two staging
/// buffers holds this batch's result copy.
struct InFlight {
    slot: usize,
    /// Submission containing the copy into the staging buffer. Waiting on it also completes
    /// every earlier submission, but not the next batch's compute, which was submitted after.
    copy_submission: wgpu::SubmissionIndex,
    batch: BatchRange,
}

/// GPU resources and per-batch scheduling.
struct BatchRunner<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    pipeline: &'a wgpu::ComputePipeline,
    bind_group: &'a wgpu::BindGroup,
    query_buffer: &'a wgpu::Buffer,
    doc_buffer: &'a wgpu::Buffer,
    params_buffer: &'a wgpu::Buffer,
    thresholds_buffer: &'a wgpu::Buffer,
    candidates_buffer: &'a wgpu::Buffer,
    staging_buffers: &'a [wgpu::Buffer; 2],
    query_vectors: &'a DerefVectorStore<f16, Mmap>,
    doc_vectors: &'a DerefVectorStore<f16, Mmap>,
    results: &'a [TopNeighbors],
    pb: &'a indicatif::ProgressBar,
    padded_dims: usize,
    candidate_capacity: usize,
    /// Sub-dispatch tile along the query/doc axes; computed once from `MAX_DISPATCH_OPS`.
    dispatch_q: usize,
    dispatch_d: usize,
    /// Stride between per-sub-dispatch param blocks: `GpuParams` padded up to
    /// `min_uniform_buffer_offset_alignment` for dynamic-offset binding.
    params_stride: usize,
    /// Query range most recently uploaded to `query_buffer`, in queue order.
    resident_queries: Option<Range<usize>>,
    /// Scratch for padding rows when dims is not a multiple of `DIM_TILE`.
    upload_scratch: Vec<f16>,
    /// Scratch for the batch's per-sub-dispatch params, staged into `params_buffer` in one write.
    params_scratch: Vec<u8>,
    thresholds_scratch: Vec<f32>,
    timings: &'a BatchTimings,
}

impl BatchRunner<'_> {
    /// Upload the batch's vectors and its queries' current thresholds, submit its
    /// (watchdog-bounded) compute dispatches and the copy of the candidates into staging buffer
    /// `slot`, and return the in-flight state. Only submits; never waits, so the GPU keeps
    /// working while the caller consumes the previous batch.
    ///
    /// All uploads are queue-ordered behind earlier submissions, so overwriting the shared
    /// input buffers here never affects work that was already submitted.
    fn submit(&mut self, batch: BatchRange, slot: usize) -> InFlight {
        let current_q = batch.queries().len();
        let current_d = batch.docs().len();

        let upload_start = Instant::now();
        if self.resident_queries.as_ref() != Some(&batch.queries()) {
            write_vectors(
                self.queue,
                self.query_buffer,
                self.query_vectors,
                batch.queries(),
                self.padded_dims,
                &mut self.upload_scratch,
            );
            self.resident_queries = Some(batch.queries());
        }
        write_vectors(
            self.queue,
            self.doc_buffer,
            self.doc_vectors,
            batch.docs(),
            self.padded_dims,
            &mut self.upload_scratch,
        );
        // f64::MAX (no threshold yet) becomes +inf, which admits every pair.
        self.thresholds_scratch.clear();
        self.thresholds_scratch.extend(
            self.results[batch.queries()]
                .iter()
                .map(|r| r.max_competitive_distance() as f32),
        );
        self.queue.write_buffer(
            self.thresholds_buffer,
            0,
            bytemuck::cast_slice(&self.thresholds_scratch),
        );
        self.timings
            .upload_us
            .fetch_add(upload_start.elapsed().as_micros() as u64, Ordering::Relaxed);

        // Split the batch into smaller dispatches so no single dispatch runs long enough to trip
        // a driver watchdog (see MAX_DISPATCH_OPS). The sub-dispatches cover disjoint (q, d)
        // sub-ranges of the batch via q_offset/d_offset and are all recorded into a single
        // compute pass; per-sub-dispatch params are staged in one write and selected with
        // dynamic uniform offsets. The whole batch (counter reset, compute, result copy) is one
        // command buffer: one wgpu submit costs on the order of a millisecond.
        let submit_start = Instant::now();

        let sub_q = current_q.div_ceil(self.dispatch_q);
        let sub_d = current_d.div_ceil(self.dispatch_d);
        let n_subs = sub_q * sub_d;
        self.params_scratch.clear();
        self.params_scratch.resize(n_subs * self.params_stride, 0);
        for i in 0..sub_q {
            for j in 0..sub_d {
                let block = &mut self.params_scratch[(i * sub_d + j) * self.params_stride..]
                    [..std::mem::size_of::<GpuParams>()];
                block.copy_from_slice(bytemuck::bytes_of(&GpuParams {
                    query_count: current_q as u32,
                    doc_count: current_d as u32,
                    q_offset: (i * self.dispatch_q) as u32,
                    d_offset: (j * self.dispatch_d) as u32,
                }));
            }
        }
        self.queue
            .write_buffer(self.params_buffer, 0, &self.params_scratch);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compute_encoder"),
            });
        // Reset the candidate count; the candidates themselves need no clearing.
        encoder.clear_buffer(
            self.candidates_buffer,
            0,
            Some(CANDIDATE_HEADER_BYTES as u64),
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("distance_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.pipeline);
            for i in 0..sub_q {
                for j in 0..sub_d {
                    let sq_start = i * self.dispatch_q;
                    let sd_start = j * self.dispatch_d;
                    let sq_end = (sq_start + self.dispatch_q).min(current_q);
                    let sd_end = (sd_start + self.dispatch_d).min(current_d);
                    pass.set_bind_group(
                        0,
                        self.bind_group,
                        &[((i * sub_d + j) * self.params_stride) as u32],
                    );
                    pass.dispatch_workgroups(
                        (sq_end - sq_start).div_ceil(WG_TILE_Q) as u32,
                        (sd_end - sd_start).div_ceil(WG_TILE_D) as u32,
                        1,
                    );
                }
            }
        }

        // Commands in an encoder execute in order, so this runs after the pass above.
        encoder.copy_buffer_to_buffer(
            self.candidates_buffer,
            0,
            &self.staging_buffers[slot],
            0,
            self.candidates_buffer.size(),
        );
        let copy_submission = self.queue.submit([encoder.finish()]);

        self.timings
            .submit_us
            .fetch_add(submit_start.elapsed().as_micros() as u64, Ordering::Relaxed);
        self.timings
            .sub_dispatches
            .fetch_add(n_subs as u64, Ordering::Relaxed);

        InFlight {
            slot,
            copy_submission,
            batch,
        }
    }

    /// Wait for `inflight` and feed its candidates into the top-k accumulators. Returns false
    /// without adding anything if the candidates overflowed the buffer, in which case the
    /// batch's range must be recomputed.
    fn consume(&self, inflight: &InFlight) -> io::Result<bool> {
        let staging = &self.staging_buffers[inflight.slot];
        let (tx, rx) = mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        let wait_start = Instant::now();
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(inflight.copy_submission.clone()),
                timeout: None,
            })
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        rx.recv()
            .unwrap()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        self.timings
            .wait_us
            .fetch_add(wait_start.elapsed().as_micros() as u64, Ordering::Relaxed);
        self.timings.batches.fetch_add(1, Ordering::Relaxed);

        let accumulate_start = Instant::now();
        let fits = {
            let mapped = staging
                .slice(..)
                .get_mapped_range()
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
            let count = u32::from_le_bytes(
                mapped[..CANDIDATE_HEADER_BYTES].try_into().unwrap(),
            ) as usize;
            self.timings
                .candidates
                .fetch_add(count as u64, Ordering::Relaxed);
            let fits = count <= self.candidate_capacity;
            if fits {
                let candidates: &[Candidate] = bytemuck::cast_slice(
                    &mapped[CANDIDATE_HEADER_BYTES
                        ..CANDIDATE_HEADER_BYTES + count * std::mem::size_of::<Candidate>()],
                );
                let batch = inflight.batch;
                candidates.par_iter().with_min_len(4096).for_each(|c| {
                    self.results[batch.q_start + c.query as usize].add(Neighbor::new(
                        (batch.d_start + c.doc as usize) as i64,
                        c.distance as f64,
                    ));
                });
            }
            fits
        };
        staging.unmap();
        self.timings.accumulate_us.fetch_add(
            accumulate_start.elapsed().as_micros() as u64,
            Ordering::Relaxed,
        );

        if fits {
            self.pb.inc(inflight.batch.pairs());
        } else {
            self.timings.overflows.fetch_add(1, Ordering::Relaxed);
        }
        Ok(fits)
    }

    /// Consume `inflight`; if its candidates overflowed, recompute its range in halves.
    ///
    /// Candidates from an overflowed attempt are discarded entirely, so recomputing never adds
    /// a pair twice. Each half is run synchronously against thresholds refreshed by everything
    /// consumed so far, and split further if it still overflows. This terminates because a
    /// single-doc range emits at most `q_batch <= candidate_capacity` candidates. Any batch
    /// submitted after `inflight` is unaffected: recovery work is queued behind it and uses the
    /// staging slot `inflight` just released.
    fn finish(&mut self, inflight: InFlight) -> io::Result<()> {
        if self.consume(&inflight)? {
            return Ok(());
        }
        self.recompute_halves(inflight.batch, inflight.slot)
    }

    fn recompute_halves(&mut self, batch: BatchRange, slot: usize) -> io::Result<()> {
        let (a, b) = batch.split().ok_or_else(|| {
            io::Error::other("candidate buffer overflowed on a single-doc range")
        })?;
        for half in [a, b] {
            let inflight = self.submit(half, slot);
            if !self.consume(&inflight)? {
                self.recompute_halves(half, slot)?;
            }
        }
        Ok(())
    }
}

/// Try to obtain a high-performance GPU adapter. Returns `None` if no suitable adapter is found.
pub fn try_adapter() -> Option<wgpu::Adapter> {
    let instance = wgpu::Instance::default();
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .ok()
}

pub fn run(adapter: wgpu::Adapter, args: &ComputeNeighborsArgs) -> io::Result<()> {
    let query_vectors: DerefVectorStore<f16, Mmap> =
        DerefVectorStore::from_file(&args.query_vectors)?;
    let query_limit = args
        .query_limit
        .unwrap_or(query_vectors.len())
        .min(query_vectors.len());

    let doc_vectors: DerefVectorStore<f16, Mmap> = DerefVectorStore::from_file(&args.doc_vectors)?;
    let doc_limit = args
        .doc_limit
        .unwrap_or(doc_vectors.len())
        .min(doc_vectors.len());

    if query_vectors.elem_stride() != doc_vectors.elem_stride() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "query and doc vectors must have the same dimensionality ({} vs {})",
                query_vectors.elem_stride(),
                doc_vectors.elem_stride()
            ),
        ));
    }
    let dims = query_vectors.elem_stride();
    let padded_dims = dims.next_multiple_of(DIM_TILE);
    let k = super::top_k(args);

    let info = adapter.get_info();
    tracing::info!("using GPU: {} ({:?})", info.name, info.backend);

    // Request the maximum buffer limits the adapter supports so we can use the largest
    // possible batches on this hardware.
    let adapter_limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("compute_neighbors"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits {
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            max_buffer_size: adapter_limits.max_buffer_size,
            ..wgpu::Limits::default()
        },
        memory_hints: wgpu::MemoryHints::Performance,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        trace: wgpu::Trace::Off,
    }))
    .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

    // Batch sizing. Every query in a batch is compared against all docs before the next query
    // batch starts, so make query batches as large as the query buffer allows: one pass over
    // the docs per query batch. Doc batches then target TARGET_BATCH_PAIRS pairs.
    let device_limits = device.limits();
    let max_buf_bytes = (device_limits.max_buffer_size as usize)
        .min(device_limits.max_storage_buffer_binding_size as usize);
    let row_bytes = padded_dims * std::mem::size_of::<f16>();
    let max_vecs = (max_buf_bytes / row_bytes).max(1);
    let q_batch = query_limit.min(max_vecs).max(1);
    let d_batch = (TARGET_BATCH_PAIRS / q_batch)
        .clamp(1, max_vecs)
        .min(doc_limit)
        .max(1);

    // Candidate capacity. Once thresholds are established a batch emits well under k
    // candidates per query, so 4x that absorbs the early batches; anything larger overflows and
    // is recomputed in halves. Capacity must be at least q_batch so that a single doc's
    // candidates always fit, which guarantees recomputation terminates.
    let candidate_capacity = (4 * q_batch * k)
        .max(MIN_CANDIDATE_CAPACITY)
        .min((MAX_CANDIDATE_BYTES.min(max_buf_bytes) - CANDIDATE_HEADER_BYTES) / std::mem::size_of::<Candidate>())
        .max(q_batch);
    tracing::info!(
        "batch sizes: q_batch={} d_batch={} candidate_capacity={} (max_buf: {} MiB)",
        q_batch,
        d_batch,
        candidate_capacity,
        max_buf_bytes / (1024 * 1024),
    );

    // --- Shader and pipeline ---
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("distance_shader"),
        source: wgpu::ShaderSource::Wgsl(
            build_shader_source(padded_dims, args.similarity).into(),
        ),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("distance_bgl"),
        entries: &[
            // Params are bound with a dynamic offset: one buffer holds the per-sub-dispatch
            // blocks for the whole batch (see BatchRunner::submit).
            bgl_entry(0, wgpu::BufferBindingType::Uniform, true),
            bgl_entry(
                1,
                wgpu::BufferBindingType::Storage { read_only: true },
                false,
            ),
            bgl_entry(
                2,
                wgpu::BufferBindingType::Storage { read_only: true },
                false,
            ),
            bgl_entry(
                3,
                wgpu::BufferBindingType::Storage { read_only: true },
                false,
            ),
            bgl_entry(
                4,
                wgpu::BufferBindingType::Storage { read_only: false },
                false,
            ),
        ],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("distance_pipeline"),
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("distance_pl"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            }),
        ),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // --- Buffers ---

    // Query and doc buffers are refilled per batch. Rows are padded to a multiple of DIM_TILE
    // f16s, so every upload is a multiple of wgpu::COPY_BUFFER_ALIGNMENT.
    let query_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("query_buffer"),
        size: (q_batch * row_bytes) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let doc_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("doc_buffer"),
        size: (d_batch * row_bytes) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let thresholds_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("thresholds_buffer"),
        size: (q_batch * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Sub-dispatch geometry (bounded by MAX_DISPATCH_OPS), aligned to workgroup tiles so only
    // the batch's last sub-dispatch on each axis has partial tiles. The params layout for one
    // batch has one aligned block per sub-dispatch, selected in-shader via dynamic uniform
    // offsets.
    let dispatch_pairs = (MAX_DISPATCH_OPS / padded_dims).max(1);
    let dispatch_side = ((dispatch_pairs as f64).sqrt() as usize / WG_TILE_Q * WG_TILE_Q)
        .max(WG_TILE_Q);
    let dispatch_q = dispatch_side.min(q_batch);
    let dispatch_d = dispatch_side.min(d_batch);
    let params_align = device_limits.min_uniform_buffer_offset_alignment as usize;
    let params_stride = std::mem::size_of::<GpuParams>().div_ceil(params_align) * params_align;
    let max_subs = q_batch.div_ceil(dispatch_q) * d_batch.div_ceil(dispatch_d);
    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params_buffer"),
        size: (max_subs * params_stride) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Candidates are appended on GPU then copied to staging buffers for CPU readback. Two
    // staging buffers let readback of batch N overlap with compute of batch N+1: a MAP_READ
    // buffer stays CPU-mapped for the whole consumption phase, so it cannot also receive the
    // next batch's copy. The single shared candidates_buffer is safe because the copy for batch
    // N executes (by queue ordering) before batch N+1 resets and overwrites it.
    let candidates_size =
        (CANDIDATE_HEADER_BYTES + candidate_capacity * std::mem::size_of::<Candidate>()) as u64;
    let candidates_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("candidates_buffer"),
        size: candidates_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let staging_buffers: [wgpu::Buffer; 2] = std::array::from_fn(|slot| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("staging_buffer_{slot}")),
            size: candidates_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("distance_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &params_buffer,
                    offset: 0,
                    size: Some(
                        std::num::NonZero::new(std::mem::size_of::<GpuParams>() as u64).unwrap(),
                    ),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: query_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: doc_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: thresholds_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: candidates_buffer.as_entire_binding(),
            },
        ],
    });

    // --- CPU-side top-k accumulators, one per query ---
    let results: Vec<TopNeighbors> = (0..query_limit).map(|_| TopNeighbors::new(k)).collect();

    // Progress is measured in (query, doc) pairs.
    let pb = progress_bar(query_limit * doc_limit, "computing distances");
    pb.enable_steady_tick(Duration::from_millis(1000));

    let timings = BatchTimings::default();
    let mut runner = BatchRunner {
        device: &device,
        queue: &queue,
        pipeline: &pipeline,
        bind_group: &bind_group,
        query_buffer: &query_buffer,
        doc_buffer: &doc_buffer,
        params_buffer: &params_buffer,
        thresholds_buffer: &thresholds_buffer,
        candidates_buffer: &candidates_buffer,
        staging_buffers: &staging_buffers,
        query_vectors: &query_vectors,
        doc_vectors: &doc_vectors,
        results: &results,
        pb: &pb,
        padded_dims,
        candidate_capacity,
        dispatch_q,
        dispatch_d,
        params_stride,
        resident_queries: None,
        upload_scratch: Vec::new(),
        params_scratch: Vec::new(),
        thresholds_scratch: Vec::with_capacity(q_batch),
        timings: &timings,
    };

    // --- Nested loop: outer over query batches, inner over doc batches ---
    //
    // Batches are pipelined two-deep: each batch is submitted *before* waiting on the previous
    // batch's results, so the GPU keeps working while the CPU reads back candidates and updates
    // the top-k accumulators. The cost is that each batch's thresholds lag by one batch.
    let mut batches = (0..query_limit).step_by(q_batch).flat_map(move |q_start| {
        let q_end = (q_start + q_batch).min(query_limit);
        (0..doc_limit).step_by(d_batch).map(move |d_start| {
            let d_end = (d_start + d_batch).min(doc_limit);
            BatchRange {
                q_start,
                q_end,
                d_start,
                d_end,
            }
        })
    });

    if let Some(first) = batches.next() {
        let mut inflight = runner.submit(first, 0);
        for batch in batches {
            let next = runner.submit(batch, 1 - inflight.slot);
            runner.finish(inflight)?;
            inflight = next;
        }
        runner.finish(inflight)?;
    }

    pb.finish();

    timings.report("batch phase timings");

    // --- Write output ---
    drop(runner);
    write_neighbors(args, results)
}

/// Upload the vectors at `rows` of `store` into `buffer` at offset 0, each row zero-padded to
/// `padded_dims` elements. When no padding is needed the rows are written straight from the
/// store's backing memory.
fn write_vectors(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    store: &DerefVectorStore<f16, Mmap>,
    rows: Range<usize>,
    padded_dims: usize,
    scratch: &mut Vec<f16>,
) {
    let dims = store.elem_stride();
    if dims == padded_dims {
        let payload = store.flat_slice(rows.start * dims, rows.end * dims);
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(payload));
    } else {
        scratch.clear();
        scratch.resize(rows.len() * padded_dims, f16::default());
        for (dst, row) in scratch.chunks_exact_mut(padded_dims).zip(rows) {
            dst[..dims].copy_from_slice(&store[row]);
        }
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(scratch));
    }
}

fn bgl_entry(
    binding: u32,
    ty: wgpu::BufferBindingType,
    has_dynamic_offset: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset,
            min_binding_size: None,
        },
        count: None,
    }
}
