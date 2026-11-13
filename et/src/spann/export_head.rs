use std::{
    fs::File,
    io::{self, BufWriter, Write},
    path::PathBuf,
    sync::Arc,
};

use clap::Args;
use easy_tiger::spann::{TableIndex, TransactionIndex};
use wt_mdb::Connection;

#[derive(Args)]
pub struct ExportHeadArgs {
    /// Output file to write centroid vectors as little-endian f32 values.
    #[arg(short, long)]
    output: PathBuf,
}

pub fn export_head(
    connection: Arc<Connection>,
    index_name: &str,
    args: ExportHeadArgs,
) -> io::Result<()> {
    let index = Arc::new(TableIndex::from_db(&connection, index_name)?);
    let txn_idx = TransactionIndex::new(&index, connection.begin_transaction(None)?);

    let hf_table = txn_idx.head().index().high_fidelity_table();
    let coder = hf_table.new_coder();
    let table_name = hf_table.name().to_owned();

    let cursor = txn_idx
        .head()
        .transaction()
        .open_record_cursor(&table_name)?;

    let mut out = BufWriter::new(File::create(&args.output)?);
    for item in cursor {
        let (key, encoded) = item?;
        if key < 0 {
            // Skip the entry point record stored at key -1.
            continue;
        }
        let decoded = coder.decode(&encoded);
        for &x in decoded.iter() {
            out.write_all(&x.to_le_bytes())?;
        }
    }

    Ok(())
}
