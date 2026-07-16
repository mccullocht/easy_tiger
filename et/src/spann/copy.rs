use std::{io, sync::Arc};

use clap::Args;
use easy_tiger::{
    posting_block::{encode_f32, leaf_page_max, PostingBlock},
    spann::{centroid_stats::CentroidCounts, CentroidAssignment, TableIndex},
};
use vectors::F32VectorCoding;
use wt_mdb::{
    connection::CreateOptionsBuilder,
    session::Formatted,
    Connection, Error,
};

use crate::ui::progress_spinner;

#[derive(Args)]
pub struct CopyArgs {
    /// Name of the destination index to create and copy into.
    ///
    /// The destination tables must not already exist.
    #[arg(short = 'd', long)]
    dest_index_name: String,
    /// If set, re-encode each posting vector into this format instead of copying blocks verbatim.
    ///
    /// This changes the posting vector length (and therefore the leaf page sizing) of the
    /// destination index, and records the new format in the destination index config.
    #[arg(long, value_enum)]
    postings_format: Option<F32VectorCoding>,
}

pub fn copy(connection: Arc<Connection>, index_name: &str, args: CopyArgs) -> io::Result<()> {
    let source = TableIndex::from_db(&connection, index_name)?;
    let head_config = source.head_config().config().clone();
    let source_spann_config = *source.config();
    // The destination config differs from the source only if we are re-encoding postings.
    let spann_config = match args.postings_format {
        Some(posting_coder) => easy_tiger::spann::IndexConfig {
            posting_coder,
            ..source_spann_config
        },
        None => source_spann_config,
    };
    // Compute destination table names without touching the db.
    let dest =
        TableIndex::from_init(&args.dest_index_name, head_config.clone(), spann_config);

    // --- Head (vamana) tables ---
    let src_head = source.head_config();
    let dst_head = dest.head_config();

    // The graph table carries the GraphConfig as app_metadata.
    copy_table::<i64, Vec<u8>>(
        &connection,
        src_head.graph_table_name(),
        dst_head.graph_table_name(),
        Some(CreateOptionsBuilder::default().app_metadata(&serde_json::to_string(&head_config)?)),
    )?;
    copy_table::<i64, Vec<u8>>(
        &connection,
        src_head.nav_table().name(),
        dst_head.nav_table().name(),
        None,
    )?;
    if let Some(src_rerank) = src_head.rerank_table() {
        let dst_rerank = dst_head
            .rerank_table()
            .expect("destination rerank table derived from same config");
        copy_table::<i64, Vec<u8>>(
            &connection,
            src_rerank.name(),
            dst_rerank.name(),
            None,
        )?;
    }

    // --- SPANN tables ---
    copy_table::<i64, CentroidAssignment>(
        &connection,
        source.centroid_assignments_table_name(),
        dest.centroid_assignments_table_name(),
        None,
    )?;
    copy_table::<i64, Vec<u8>>(
        &connection,
        source.raw_vectors_table_name(),
        dest.raw_vectors_table_name(),
        None,
    )?;
    copy_table::<u32, CentroidCounts>(
        &connection,
        source.centroid_stats_table_name(),
        dest.centroid_stats_table_name(),
        None,
    )?;

    // The postings table carries the IndexConfig as app_metadata and uses larger leaf pages sized
    // to hold a full centroid worth of posting vectors. Both the app_metadata and the page sizing
    // reflect the (possibly re-encoded) destination posting format.
    let dimensions = head_config.dimensions.get();
    let similarity = head_config.similarity;
    let posting_vector_len = spann_config
        .posting_coder
        .coder(similarity, None)
        .byte_len(dimensions);
    let leaf_page_size =
        leaf_page_max(spann_config.max_centroid_len, posting_vector_len, 4096) as u32;
    let postings_options = CreateOptionsBuilder::default()
        .app_metadata(&serde_json::to_string(&spann_config)?)
        .leaf_page_max(leaf_page_size)
        .leaf_value_max(leaf_page_size);
    if args.postings_format.is_some() {
        let src_coder = source_spann_config
            .posting_coder
            .coder(similarity, None);
        let dst_coder = spann_config.posting_coder.coder(similarity, None);
        copy_postings_reencoded(
            &connection,
            source.postings_table_name(),
            dest.postings_table_name(),
            postings_options,
            src_coder.as_ref(),
            dst_coder.as_ref(),
            src_coder.byte_len(dimensions),
            dimensions,
        )?;
    } else {
        copy_table::<u32, Vec<u8>>(
            &connection,
            source.postings_table_name(),
            dest.postings_table_name(),
            Some(postings_options),
        )?;
    }

    Ok(())
}

/// Copy the postings table, re-encoding every vector in each block from `src_coder`'s format into
/// `dst_coder`'s format.
///
/// Each source block is decoded to f32 vectors and re-encoded, so the destination blocks use the
/// new (possibly different length) posting vector encoding.
#[allow(clippy::too_many_arguments)]
fn copy_postings_reencoded(
    connection: &Arc<Connection>,
    source_table: &str,
    dest_table: &str,
    dest_options: CreateOptionsBuilder,
    src_coder: &dyn vectors::F32VectorCoder,
    dst_coder: &dyn vectors::F32VectorCoder,
    src_vector_len: usize,
    dimensions: usize,
) -> io::Result<()> {
    let progress = progress_spinner(format!("re-encode {source_table} -> {dest_table}"));
    let txn = connection.begin_transaction(None)?;
    let cursor = txn.open_cursor::<u32, Vec<u8>>(source_table)?;
    let mut bulk =
        connection.new_bulk_load_cursor::<u32, Vec<u8>>(dest_table, Some(dest_options))?;
    for item in cursor {
        let (centroid_id, data) = item?;
        let block = PostingBlock::new(&data, src_vector_len)
            .ok_or_else(|| Error::wired_tiger(wt_mdb::WiredTigerError::Generic))?;
        let reencoded = encode_f32(
            block.iter().map(|(id, encoded)| (id, src_coder.decode(encoded))),
            dst_coder,
            dimensions,
        );
        bulk.append(centroid_id, reencoded.as_slice())?;
        progress.inc(1);
    }
    progress.finish();
    Ok(())
}

/// Copy every row of `source_table` into a freshly created `dest_table` using a bulk load cursor.
///
/// Source rows are scanned in ascending key order, satisfying the bulk cursor's requirement that
/// keys be appended in order. `dest_options` supplies any table-specific create options (app
/// metadata, page sizing); the key and value formats are added by the bulk cursor.
fn copy_table<K, V>(
    connection: &Arc<Connection>,
    source_table: &str,
    dest_table: &str,
    dest_options: Option<CreateOptionsBuilder>,
) -> io::Result<()>
where
    K: Formatted,
    V: Formatted,
{
    let progress = progress_spinner(format!("copy {source_table} -> {dest_table}"));
    let txn = connection.begin_transaction(None)?;
    let cursor = txn.open_cursor::<K, V>(source_table)?;
    let mut bulk = connection.new_bulk_load_cursor::<K, V>(dest_table, dest_options)?;
    for item in cursor {
        let (key, value) = item?;
        bulk.append(key.to_formatted_ref(), value.to_formatted_ref())?;
        progress.inc(1);
    }
    progress.finish();
    Ok(())
}
