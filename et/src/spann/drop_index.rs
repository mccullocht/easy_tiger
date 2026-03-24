use std::{io, sync::Arc};

use easy_tiger::spann::TableIndex;
use wt_mdb::{Connection, connection::DropOptionsBuilder};

pub fn drop_index(connection: Arc<Connection>, index_name: &str) -> io::Result<()> {
    TableIndex::drop_tables(
        &connection,
        index_name,
        &Some(DropOptionsBuilder::default().set_force().into()),
    )
    .map_err(|e| e.into())
}
