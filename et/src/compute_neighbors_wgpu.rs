use std::{
    fs::File,
    io::{self, BufWriter, Write},
    num::NonZero,
    path::PathBuf,
    sync::mpsc,
    time::Duration,
};

use bytemuck::{Pod, Zeroable};
use clap::Args;
use easy_tiger::{
    Neighbor,
    input::{DerefVectorStore, VectorStore},
};
use memmap2::Mmap;
use vectors::VectorSimilarity;

use crate::{neighbor_util::TopNeighbors, ui::progress_bar};

/// WGSL compute shader for pairwise distance computation.
///
/// Each thread computes the distance between one (query, doc) pair. Three distance functions are
/// supported, selected at runtime via the `similarity` uniform:
///   0 = Euclidean (squared L2)
///   1 = Dot product distance – assumes pre-normalized vectors: (-dot + 1) / 2
///   2 = Cosine distance – full cosine with per-pair normalization
///
/// Output distances are written row-major as `distances[q * doc_count + d]`.
const SHADER: &str = r#"
struct Params {
    query_count: u32,
    doc_count: u32,
    dimensions: u32,
    similarity: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> query_vectors: array<f32>;
@group(0) @binding(2) var<storage, read> doc_vectors: array<f32>;
@group(0) @binding(3) var<storage, read_write> distances: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q = gid.x;
    let d = gid.y;

    if q >= params.query_count || d >= params.doc_count {
        return;
    }

    let q_base = q * params.dimensions;
    let d_base = d * params.dimensions;
    var result: f32;

    switch params.similarity {
        case 0u: {
            // Euclidean: squared L2 distance
            var acc: f32 = 0.0;
            for (var i: u32 = 0u; i < params.dimensions; i++) {
                let diff = query_vectors[q_base + i] - doc_vectors[d_base + i];
                acc = fma(diff, diff, acc);
            }
            result = acc;
        }
        case 1u: {
            // Dot product distance (vectors assumed to be normalized): (-dot + 1) / 2
            var acc: f32 = 0.0;
            for (var i: u32 = 0u; i < params.dimensions; i++) {
                acc = fma(query_vectors[q_base + i], doc_vectors[d_base + i], acc);
            }
            result = (-acc + 1.0) / 2.0;
        }
        default: {
            // Cosine distance: full computation without assuming normalization
            var dot_qd: f32 = 0.0;
            var norm_q: f32 = 0.0;
            var norm_d: f32 = 0.0;
            for (var i: u32 = 0u; i < params.dimensions; i++) {
                let qv = query_vectors[q_base + i];
                let dv = doc_vectors[d_base + i];
                dot_qd = fma(qv, dv, dot_qd);
                norm_q = fma(qv, qv, norm_q);
                norm_d = fma(dv, dv, norm_d);
            }
            result = (-dot_qd / sqrt(norm_q * norm_d) + 1.0) / 2.0;
        }
    }

    distances[q * params.doc_count + d] = result;
}
"#;

/// Uniform buffer layout for the distance shader.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuParams {
    query_count: u32,
    doc_count: u32,
    dimensions: u32,
    similarity: u32,
}

const WG_Q: usize = 16;
const WG_D: usize = 16;

#[derive(Args)]
pub struct ComputeNeighborsWgpuArgs {
    /// Path to numpy formatted little-endian float vectors.
    #[arg(long)]
    query_vectors: PathBuf,
    /// Maximum number of query vectors to process.
    #[arg(long)]
    query_limit: Option<usize>,
    /// Path to numpy formatted little-endian float vectors.
    #[arg(long)]
    doc_vectors: PathBuf,
    /// Maximum number of doc vectors to process.
    #[arg(long)]
    doc_limit: Option<usize>,

    /// Number of dimensions for both query and doc vectors.
    #[arg(short, long)]
    dimensions: NonZero<usize>,
    /// Similarity function to use.
    #[arg(short, long)]
    similarity: VectorSimilarity,

    /// Path to neighbors file to write.
    ///
    /// The output file will contain one row for each vector in query_vectors. Within each row
    /// there will be neighbors_len entries of Neighbor, an (i64, f64) tuple.
    #[arg(short, long)]
    neighbors: PathBuf,
    /// Number of neighbors for each query in the neighbors file.
    #[arg(long, default_value_t = NonZero::new(100).unwrap())]
    neighbors_len: NonZero<usize>,
}

pub fn compute_neighbors_wgpu(args: ComputeNeighborsWgpuArgs) -> io::Result<()> {
    let query_vectors: DerefVectorStore<f32, Mmap> = DerefVectorStore::new(
        unsafe { Mmap::map(&File::open(&args.query_vectors)?)? },
        args.dimensions,
    )?;
    let query_limit = args
        .query_limit
        .unwrap_or(query_vectors.len())
        .min(query_vectors.len());

    let doc_vectors: DerefVectorStore<f32, Mmap> = DerefVectorStore::new(
        unsafe { Mmap::map(&File::open(&args.doc_vectors)?)? },
        args.dimensions,
    )?;
    let doc_limit = args
        .doc_limit
        .unwrap_or(doc_vectors.len())
        .min(doc_vectors.len());

    let dims = args.dimensions.get();
    let k = args.neighbors_len.get();

    let similarity_code: u32 = match args.similarity {
        VectorSimilarity::Euclidean => 0,
        VectorSimilarity::Dot => 1,
        VectorSimilarity::Cosine => 2,
    };

    // --- GPU setup ---
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

    let info = adapter.get_info();
    tracing::info!("using GPU: {} ({:?})", info.name, info.backend);

    // Request the maximum buffer limits the adapter supports so we can use the largest
    // possible batches on this hardware.
    let adapter_limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
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
        },
    ))
    .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

    // Compute batch sizes from the limits actually granted by the device. Three buffers are
    // constrained:
    //   query buffer:    q_batch * dims * 4  <= max_buf_bytes
    //   doc buffer:      d_batch * dims * 4  <= max_buf_bytes
    //   distances buffer: q_batch * d_batch * 4 <= max_buf_bytes
    //
    // To maximise GPU utilisation we size batches symmetrically: q_batch ≈ d_batch ≈
    // sqrt(max_pairs) so the distances buffer fills the allowed budget. Both vector buffers
    // are also checked against max_vecs = max_buf_bytes / (dims * 4).
    let device_limits = device.limits();
    let max_buf_bytes = (device_limits.max_buffer_size as usize)
        .min(device_limits.max_storage_buffer_binding_size as usize);
    let (q_batch, d_batch) = if query_limit == 0 || doc_limit == 0 {
        (1, 1) // buffers are created but the processing loop won't execute
    } else {
        let max_vecs = max_buf_bytes / (dims * std::mem::size_of::<f32>());
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
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
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

    // Query and doc buffers are refilled each batch; both need COPY_DST.
    let query_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("query_buffer"),
        size: (q_batch * dims * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let doc_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("doc_buffer"),
        size: (d_batch * dims * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Params are re-written each batch.
    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params_buffer"),
        size: std::mem::size_of::<GpuParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Distances are computed on GPU then copied to a staging buffer for CPU readback.
    let dist_buffer_size = (q_batch * d_batch * std::mem::size_of::<f32>()) as u64;
    let distances_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("distances_buffer"),
        size: dist_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_buffer"),
        size: dist_buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
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

    // --- CPU-side top-k accumulators, one per query ---
    let results: Vec<TopNeighbors> = (0..query_limit).map(|_| TopNeighbors::new(k)).collect();

    // Progress is measured in (query, doc) pairs.
    let pb = progress_bar(query_limit * doc_limit, "computing distances");
    pb.enable_steady_tick(Duration::from_millis(1000));

    // --- Nested loop: outer over query batches, inner over doc batches ---
    let mut q_start = 0usize;
    while q_start < query_limit {
        let q_end = (q_start + q_batch).min(query_limit);
        let current_q = q_end - q_start;

        let query_data: Vec<f32> = (q_start..q_end)
            .flat_map(|q| query_vectors[q].iter().copied())
            .collect();
        queue.write_buffer(&query_buffer, 0, bytemuck::cast_slice(&query_data));

        let mut d_start = 0usize;
        while d_start < doc_limit {
            let d_end = (d_start + d_batch).min(doc_limit);
            let current_d = d_end - d_start;

            let doc_data: Vec<f32> = (d_start..d_end)
                .flat_map(|d| doc_vectors[d].iter().copied())
                .collect();
            queue.write_buffer(&doc_buffer, 0, bytemuck::cast_slice(&doc_data));

            queue.write_buffer(
                &params_buffer,
                0,
                bytemuck::bytes_of(&GpuParams {
                    query_count: current_q as u32,
                    doc_count: current_d as u32,
                    dimensions: dims as u32,
                    similarity: similarity_code,
                }),
            );

            // Dispatch compute and copy results to staging.
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compute_encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("distance_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(
                    current_q.div_ceil(WG_Q) as u32,
                    current_d.div_ceil(WG_D) as u32,
                    1,
                );
            }
            let copy_bytes = (current_q * current_d * std::mem::size_of::<f32>()) as u64;
            encoder.copy_buffer_to_buffer(&distances_buffer, 0, &staging_buffer, 0, copy_bytes);
            queue.submit([encoder.finish()]);

            // Map the staging buffer and block until GPU work is done.
            let (tx, rx) = mpsc::channel();
            staging_buffer
                .slice(..copy_bytes)
                .map_async(wgpu::MapMode::Read, move |result| {
                    tx.send(result).unwrap();
                });
            device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
            rx.recv()
                .unwrap()
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

            // Feed GPU distances into the per-query TopNeighbors accumulators.
            //
            // The shader writes distances row-major as distances[q_local * current_d + d_local].
            {
                let mapped = staging_buffer.slice(..copy_bytes).get_mapped_range();
                let distances: &[f32] = bytemuck::cast_slice(&mapped);
                for q_local in 0..current_q {
                    let row = &distances[q_local * current_d..(q_local + 1) * current_d];
                    for (d_local, &dist) in row.iter().enumerate() {
                        results[q_start + q_local]
                            .add(Neighbor::new((d_start + d_local) as i64, dist as f64));
                    }
                }
            }
            staging_buffer.unmap();

            pb.inc((current_q * current_d) as u64);
            d_start = d_end;
        }

        q_start = q_end;
    }

    // --- Write output ---
    let mut writer = BufWriter::new(File::create(&args.neighbors)?);
    for neighbors in results.into_iter().map(|r| r.into_neighbors()) {
        for n in neighbors.into_iter().take(k) {
            writer.write_all(&<[u8; 16]>::from(n))?;
        }
    }

    Ok(())
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
