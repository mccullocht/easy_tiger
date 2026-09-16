use std::{io, sync::mpsc, time::Duration};

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

/// WGSL compute shader for pairwise distance computation.
///
/// Each thread computes the distance between one (query, doc) pair. Query and doc vectors are
/// stored as f16 to match the input format, but every accumulation is carried out in f32 for
/// precision. The shader is instantiated at runtime from the templates below with `dims` and the
/// similarity function baked into the source, since both are fixed for the whole run: loop bounds
/// become compile-time constants the driver compiler can unroll, and the runtime similarity
/// switch disappears.
///
/// Two load variants are generated:
///   - `SHADER_SCALAR` (odd dims): scalar `array<f16>` loads, `enable f16`.
///   - `SHADER_PACKED` (even dims): f16 pairs are read as `u32`s and unpacked with
///     `unpack2x16float`, so all threads load fully coalesced 4-byte elements instead of
///     strided 2-byte ones. Requires no f16 shader features.
///
/// Similarity bodies:
///   0 = Euclidean (squared L2)
///   1 = Dot product distance – assumes pre-normalized vectors: (-dot + 1) / 2
///
/// (Cosine is Dot over normalized vectors; normalize the inputs first.)
///
/// Output distances are written row-major as `distances[q * doc_count + d]`.
const SHADER_SCALAR: &str = r#"
enable f16;

struct Params {
    query_count: u32,
    doc_count: u32,
    dimensions: u32,
    q_offset: u32,
    d_offset: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> query_vectors: array<f16>;
@group(0) @binding(2) var<storage, read> doc_vectors: array<f16>;
@group(0) @binding(3) var<storage, read_write> distances: array<f32>;

// Baked in at runtime.
const DIMS: u32 = __DIMS__u;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q = gid.x + params.q_offset;
    let d = gid.y + params.d_offset;

    if q >= params.query_count || d >= params.doc_count {
        return;
    }

    let q_base = q * DIMS;
    let d_base = d * DIMS;
__BODY__
    distances[q * params.doc_count + d] = result;
}
"#;

const SHADER_PACKED: &str = r#"
struct Params {
    query_count: u32,
    doc_count: u32,
    dimensions: u32,
    q_offset: u32,
    d_offset: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Each u32 holds two f16s (little-endian: low half is the even element).
@group(0) @binding(1) var<storage, read> query_vectors: array<u32>;
@group(0) @binding(2) var<storage, read> doc_vectors: array<u32>;
@group(0) @binding(3) var<storage, read_write> distances: array<f32>;

// Baked in at runtime. DIMS is always even in this variant, so every row is a whole number of
// u32s and no scalar tail handling is needed.
const DIMS: u32 = __DIMS__u;
const HALF_DIMS: u32 = DIMS / 2u;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q = gid.x + params.q_offset;
    let d = gid.y + params.d_offset;

    if q >= params.query_count || d >= params.doc_count {
        return;
    }

    let q_base = q * HALF_DIMS;
    let d_base = d * HALF_DIMS;
__BODY__
    distances[q * params.doc_count + d] = result;
}
"#;

const EUCLID_SCALAR_BODY: &str = r#"
    var acc: f32 = 0.0;
    for (var i: u32 = 0u; i < DIMS; i++) {
        let diff = f32(query_vectors[q_base + i]) - f32(doc_vectors[d_base + i]);
        acc = fma(diff, diff, acc);
    }
    let result = acc;
"#;

const DOT_SCALAR_BODY: &str = r#"
    var acc: f32 = 0.0;
    for (var i: u32 = 0u; i < DIMS; i++) {
        acc = fma(f32(query_vectors[q_base + i]), f32(doc_vectors[d_base + i]), acc);
    }
    let result = (-acc + 1.0) / 2.0;
"#;

const EUCLID_PACKED_BODY: &str = r#"
    var acc = vec2<f32>(0.0);
    for (var i: u32 = 0u; i < HALF_DIMS; i++) {
        let pq = unpack2x16float(query_vectors[q_base + i]);
        let pd = unpack2x16float(doc_vectors[d_base + i]);
        let diff = pq - pd;
        acc += diff * diff;
    }
    let result = acc.x + acc.y;
"#;

const DOT_PACKED_BODY: &str = r#"
    var acc = vec2<f32>(0.0);
    for (var i: u32 = 0u; i < HALF_DIMS; i++) {
        let pq = unpack2x16float(query_vectors[q_base + i]);
        let pd = unpack2x16float(doc_vectors[d_base + i]);
        acc += pq * pd;
    }
    let result = (-acc.x - acc.y + 1.0) / 2.0;
"#;

/// Instantiate the shader templates for this run's dimensionality and similarity function.
fn build_shader_source(dims: usize, similarity: VectorSimilarity) -> String {
    let packed = dims.is_multiple_of(2);
    let template = if packed { SHADER_PACKED } else { SHADER_SCALAR };
    let body = match (packed, similarity) {
        (true, VectorSimilarity::Euclidean) => EUCLID_PACKED_BODY,
        (true, VectorSimilarity::Dot) => DOT_PACKED_BODY,
        (false, VectorSimilarity::Euclidean) => EUCLID_SCALAR_BODY,
        (false, VectorSimilarity::Dot) => DOT_SCALAR_BODY,
    };
    template
        .replace("__DIMS__", &dims.to_string())
        .replace("__BODY__", body)
}

/// Uniform buffer layout for the distance shader.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuParams {
    query_count: u32,
    doc_count: u32,
    dimensions: u32,
    q_offset: u32,
    d_offset: u32,
}

const WG_Q: usize = 16;
const WG_D: usize = 16;

/// Upper bound on scalar multiply-add operations (~query * doc * dims) issued by a single
/// dispatch. The memory-driven batch sizes below can produce dispatches with hundreds of
/// millions of threads, each running an O(dims) loop; on a single command buffer that can run
/// long enough to trip a driver watchdog (e.g. Windows TDR, or Linux GPU hangcheck), which kills
/// the device ("device lost") well before the GPU is actually short on memory. Splitting a
/// memory-sized batch into several dispatches bounded by this budget keeps each individual
/// dispatch short, independent of how much buffer memory is available.
const MAX_DISPATCH_OPS: usize = 1 << 30;

/// A (query, doc) range scheduled for computation.
#[derive(Clone, Copy)]
struct BatchRange {
    q_start: usize,
    q_end: usize,
    d_start: usize,
    d_end: usize,
}

/// A batch whose compute and result copy have been submitted to the GPU and which is awaiting
/// CPU-side consumption. Batches are pipelined two-deep; `slot` selects which of the two staging
/// buffers holds this batch's result copy.
#[derive(Clone)]
struct InFlight {
    slot: usize,
    /// Submission containing the copy into the staging buffer. Waiting on it also completes
    /// every earlier submission, but not the next batch's compute, which was submitted after.
    copy_submission: wgpu::SubmissionIndex,
    copy_bytes: usize,
    batch: BatchRange,
}

/// GPU resources and per-batch scheduling shared by `prepare_batch`/`consume_batch`.
struct BatchRunner<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    pipeline: &'a wgpu::ComputePipeline,
    bind_group: &'a wgpu::BindGroup,
    query_buffer: &'a wgpu::Buffer,
    doc_buffer: &'a wgpu::Buffer,
    params_buffer: &'a wgpu::Buffer,
    distances_buffer: &'a wgpu::Buffer,
    staging_buffers: &'a [wgpu::Buffer; 2],
    dims: usize,
    /// Scratch for padding upload payloads when the byte length is not
    /// COPY_BUFFER_ALIGNMENT-aligned (odd dims only).
    upload_scratch: Vec<f16>,
}

impl BatchRunner<'_> {
    /// Upload the batch's vectors, submit its (watchdog-bounded) compute dispatches and the copy
    /// of the results into staging buffer `slot`, and return the in-flight state. Only submits;
    /// never waits, so the GPU keeps working while the caller consumes the previous batch.
    fn prepare_batch(
        &mut self,
        batch: BatchRange,
        slot: usize,
        upload_query: bool,
        query_vectors: &DerefVectorStore<f16, Mmap>,
        doc_vectors: &DerefVectorStore<f16, Mmap>,
    ) -> InFlight {
        let current_q = batch.q_end - batch.q_start;
        let current_d = batch.d_end - batch.d_start;

        if upload_query {
            write_vector_payload(
                self.queue,
                self.query_buffer,
                query_vectors.flat_slice(batch.q_start * self.dims, batch.q_end * self.dims),
                &mut self.upload_scratch,
            );
        }
        write_vector_payload(
            self.queue,
            self.doc_buffer,
            doc_vectors.flat_slice(batch.d_start * self.dims, batch.d_end * self.dims),
            &mut self.upload_scratch,
        );

        // Split this memory-sized batch into smaller dispatches so no single command buffer
        // runs long enough to trip a driver watchdog. Query/doc buffers already hold the
        // full batch, so sub-dispatches just cover different (q, d) sub-ranges of it via
        // q_offset/d_offset; only the distances buffer copy needs to wait for all of them.
        let dispatch_pairs = (MAX_DISPATCH_OPS / self.dims).max(1);
        let dispatch_side = (dispatch_pairs as f64).sqrt() as usize;
        let dispatch_q = dispatch_side.min(current_q).max(1);
        let dispatch_d = dispatch_side.min(current_d).max(1);

        let mut sq_start = 0usize;
        while sq_start < current_q {
            let sq_end = (sq_start + dispatch_q).min(current_q);
            let mut sd_start = 0usize;
            while sd_start < current_d {
                let sd_end = (sd_start + dispatch_d).min(current_d);

                self.queue.write_buffer(
                    self.params_buffer,
                    0,
                    bytemuck::bytes_of(&GpuParams {
                        query_count: current_q as u32,
                        doc_count: current_d as u32,
                        dimensions: self.dims as u32,
                        q_offset: sq_start as u32,
                        d_offset: sd_start as u32,
                    }),
                );

                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("compute_encoder"),
                        });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("distance_pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(self.pipeline);
                    pass.set_bind_group(0, self.bind_group, &[]);
                    pass.dispatch_workgroups(
                        (sq_end - sq_start).div_ceil(WG_Q) as u32,
                        (sd_end - sd_start).div_ceil(WG_D) as u32,
                        1,
                    );
                }
                self.queue.submit([encoder.finish()]);

                sd_start = sd_end;
            }
            sq_start = sq_end;
        }

        // Copy the full batch of results to staging once all sub-dispatches have completed.
        let copy_bytes = (current_q * current_d * std::mem::size_of::<f32>()) as u64;
        let mut copy_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("copy_encoder"),
                });
        copy_encoder.copy_buffer_to_buffer(
            self.distances_buffer,
            0,
            &self.staging_buffers[slot],
            0,
            copy_bytes,
        );
        let copy_submission = self.queue.submit([copy_encoder.finish()]);

        InFlight {
            slot,
            copy_submission,
            copy_bytes: copy_bytes as usize,
            batch,
        }
    }

    /// Wait for `inflight`'s result copy, feed the distances into the per-query top-k
    /// accumulators, and release the staging slot. Blocks only on this batch's copy; the next
    /// batch's compute (submitted before this call) runs concurrently.
    fn consume_batch(
        &self,
        inflight: InFlight,
        results: &[TopNeighbors],
        pb: &indicatif::ProgressBar,
    ) -> io::Result<()> {
        let staging = &self.staging_buffers[inflight.slot];
        let (tx, rx) = mpsc::channel();
        staging
            .slice(..inflight.copy_bytes as u64)
            .map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(inflight.copy_submission.clone()),
                timeout: None,
            })
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        rx.recv()
            .unwrap()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

        // Feed GPU distances into the per-query TopNeighbors accumulators.
        //
        // The shader writes distances row-major as distances[q_local * current_d + d_local].
        // Parallelize at row granularity: one rayon task per query row, with the inner loop over
        // that row's distances kept serial.
        {
            let current_d = inflight.batch.d_end - inflight.batch.d_start;
            let mapped = staging
                .slice(..inflight.copy_bytes as u64)
                .get_mapped_range()
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
            let distances: &[f32] = bytemuck::cast_slice(&mapped);
            results[inflight.batch.q_start..inflight.batch.q_end]
                .par_iter()
                .zip(distances.par_chunks_exact(current_d))
                .for_each(|(top, row)| {
                    for (d_local, &dist) in row.iter().enumerate() {
                        top.add(Neighbor::new(
                            (inflight.batch.d_start + d_local) as i64,
                            dist as f64,
                        ));
                    }
                });
        }
        staging.unmap();

        pb.inc(
            ((inflight.batch.q_end - inflight.batch.q_start)
                * (inflight.batch.d_end - inflight.batch.d_start)) as u64,
        );
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

/// Return true if `adapter` can run WGSL shaders that use the `f16` type, which this module
/// requires since input vectors are stored as f16.
pub fn supports_f16(adapter: &wgpu::Adapter) -> bool {
    adapter.features().contains(wgpu::Features::SHADER_F16)
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
    let k = super::top_k(args);

    let info = adapter.get_info();
    tracing::info!("using GPU: {} ({:?})", info.name, info.backend);

    // Request the maximum buffer limits the adapter supports so we can use the largest
    // possible batches on this hardware.
    let adapter_limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("compute_neighbors"),
        required_features: wgpu::Features::SHADER_F16,
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

    // Compute batch sizes from the limits actually granted by the device. Three buffers are
    // constrained:
    //   query buffer:    q_batch * dims * 2  <= max_buf_bytes
    //   doc buffer:      d_batch * dims * 2  <= max_buf_bytes
    //   distances buffer: q_batch * d_batch * 4 <= max_buf_bytes
    //
    // To maximise GPU utilisation we size batches symmetrically: q_batch ≈ d_batch ≈
    // sqrt(max_pairs) so the distances buffer fills the allowed budget. Both vector buffers
    // are also checked against max_vecs = max_buf_bytes / (dims * 2).
    let device_limits = device.limits();
    let max_buf_bytes = (device_limits.max_buffer_size as usize)
        .min(device_limits.max_storage_buffer_binding_size as usize);
    let (q_batch, d_batch) = if query_limit == 0 || doc_limit == 0 {
        (1, 1) // buffers are created but the processing loop won't execute
    } else {
        let max_vecs = max_buf_bytes / (dims * std::mem::size_of::<f16>());
        let max_pairs = max_buf_bytes / std::mem::size_of::<f32>();
        let sq = (max_pairs as f64).sqrt() as usize;
        let q = sq.min(max_vecs).min(query_limit).max(1);
        let d = (max_pairs / q).min(max_vecs).min(doc_limit).max(1);
        (q, d)
    };
    tracing::info!(
        "batch sizes: q_batch={} d_batch={} (max_buf: {} MiB)",
        q_batch,
        d_batch,
        max_buf_bytes / (1024 * 1024),
    );

    // --- Shader and pipeline ---
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("distance_shader"),
        source: wgpu::ShaderSource::Wgsl(build_shader_source(dims, args.similarity).into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("distance_bgl"),
        entries: &[
            bgl_entry(0, wgpu::BufferBindingType::Uniform),
            bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
            bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
            bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
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

    // Query and doc buffers are refilled each batch; both need COPY_DST. Sizes are rounded up to
    // wgpu::COPY_BUFFER_ALIGNMENT (4 bytes) since f16 elements are only 2 bytes wide and
    // write_buffer/copy_buffer_to_buffer require 4-byte aligned sizes; uploads from odd-dims
    // inputs are padded via a scratch buffer to match.
    let query_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("query_buffer"),
        size: round_up_copy_alignment(q_batch * dims * std::mem::size_of::<f16>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let doc_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("doc_buffer"),
        size: round_up_copy_alignment(d_batch * dims * std::mem::size_of::<f16>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Params are re-written each dispatch.
    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params_buffer"),
        size: std::mem::size_of::<GpuParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Distances are computed on GPU then copied to staging buffers for CPU readback. Two staging
    // buffers let readback of batch N overlap with compute of batch N+1: a MAP_READ buffer stays
    // CPU-mapped for the whole consumption phase, so it cannot also receive the next batch's
    // copy. The single shared distances_buffer is safe because the copy for batch N is submitted
    // (and, by queue ordering, executes) before batch N+1's dispatches overwrite it.
    let dist_buffer_size = (q_batch * d_batch * std::mem::size_of::<f32>()) as u64;
    let distances_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("distances_buffer"),
        size: dist_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffers: [wgpu::Buffer; 2] = std::array::from_fn(|slot| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("staging_buffer_{slot}")),
            size: dist_buffer_size,
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
                resource: params_buffer.as_entire_binding(),
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
                resource: distances_buffer.as_entire_binding(),
            },
        ],
    });

    let mut runner = BatchRunner {
        device: &device,
        queue: &queue,
        pipeline: &pipeline,
        bind_group: &bind_group,
        query_buffer: &query_buffer,
        doc_buffer: &doc_buffer,
        params_buffer: &params_buffer,
        distances_buffer: &distances_buffer,
        staging_buffers: &staging_buffers,
        dims,
        upload_scratch: Vec::new(),
    };

    // --- CPU-side top-k accumulators, one per query ---
    let results: Vec<TopNeighbors> = (0..query_limit).map(|_| TopNeighbors::new(k)).collect();

    // Progress is measured in (query, doc) pairs.
    let pb = progress_bar(query_limit * doc_limit, "computing distances");
    pb.enable_steady_tick(Duration::from_millis(1000));

    // --- Nested loop: outer over query batches, inner over doc batches ---
    //
    // Batches are pipelined two-deep: each batch's upload and compute are submitted *before*
    // waiting on the previous batch's result copy, so the GPU keeps working while the CPU reads
    // back distances and updates the top-k accumulators.
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

    let Some(mut inflight) = batches
        .next()
        .map(|batch| runner.prepare_batch(batch, 0, true, &query_vectors, &doc_vectors))
    else {
        // Nothing to compute (empty inputs); skip straight to writing results.
        return write_neighbors(args, results);
    };

    // The query buffer only changes when the q batch rolls over, so skip its re-upload within
    // one q batch. (Safe to defer: batch N+1's upload is queue-ordered after batch N's compute.)
    let mut prev_q_start = inflight.batch.q_start;
    for batch in batches {
        let upload_query = batch.q_start != prev_q_start;
        prev_q_start = batch.q_start;
        let slot = 1 - inflight.slot;
        let next = runner.prepare_batch(batch, slot, upload_query, &query_vectors, &doc_vectors);
        runner.consume_batch(inflight, &results, &pb)?;
        inflight = next;
    }
    runner.consume_batch(inflight, &results, &pb)?;

    // --- Write output ---
    write_neighbors(args, results)
}

/// Round `bytes` up to `wgpu::COPY_BUFFER_ALIGNMENT` (4 bytes), the minimum granularity for
/// `write_buffer`/`copy_buffer_to_buffer` calls.
fn round_up_copy_alignment(bytes: usize) -> usize {
    bytes.div_ceil(wgpu::COPY_BUFFER_ALIGNMENT as usize) * wgpu::COPY_BUFFER_ALIGNMENT as usize
}

/// Upload `payload` (contiguous f16 elements) into `buffer` at offset 0. The payload is written
/// straight from its backing storage; when its byte length is not a multiple of
/// `wgpu::COPY_BUFFER_ALIGNMENT` (only possible for odd dims), it is first copied into `scratch`
/// and padded with zeros. The padding elements sit past the last index any dispatch reads
/// (bounded by `params.query_count`/`params.doc_count`), so they never affect results.
fn write_vector_payload(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    payload: &[f16],
    scratch: &mut Vec<f16>,
) {
    let byte_len = std::mem::size_of_val(payload);
    if byte_len.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT as usize) {
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(payload));
    } else {
        scratch.clear();
        scratch.extend_from_slice(payload);
        let padded_elems = round_up_copy_alignment(byte_len) / std::mem::size_of::<f16>();
        scratch.resize(padded_elems, f16::default());
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(scratch));
    }
}

fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
