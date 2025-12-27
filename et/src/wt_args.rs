use std::{num::NonZero, sync::Arc};

use clap::Args;
use wt_mdb::{connection::OptionsBuilder, Connection, Result, Statistics};

#[derive(Args)]
pub struct WiredTigerArgs {
    /// Path to the WiredTiger database.
    #[arg(long)]
    wiredtiger_db_path: String,
    /// Size of the WiredTiger disk cache, in MB.
    #[arg(long, default_value_t = NonZero::new(1024).unwrap())]
    wiredtiger_cache_size_mb: NonZero<usize>,
    /// If true, create the WiredTiger database if it does not exist.
    #[arg(long, default_value_t = false)]
    wiredtiger_create_db: bool,

    /// Name of the index, used to derive table names in WiredTiger.
    #[arg(short, long)]
    index_name: String,
}

impl WiredTigerArgs {
    pub fn open_connection(&self) -> Result<Arc<Connection>> {
        // TODO: Connection.filename should accept &Path. This will likely be very annoying to plumb to CString.
        let mut connection_options = OptionsBuilder::default()
            .cache_size_mb(self.wiredtiger_cache_size_mb)
            .statistics(Statistics::Fast)
            .checkpoint_log_size(128 << 20);
        if self.wiredtiger_create_db {
            connection_options = connection_options.create();
        }
        Connection::open(&self.wiredtiger_db_path, Some(connection_options.into()))
    }

    pub fn index_name(&self) -> &str {
        &self.index_name
    }
}
