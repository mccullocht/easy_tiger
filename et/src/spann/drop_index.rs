use std::{io, sync::Arc};

use easy_tiger::spann::TableIndex;
use wt_mdb::{options::DropOptionsBuilder, Connection};

pub fn drop_index(connection: Arc<Connection>, index_name: &str) -> io::Result<()> {
    TableIndex::drop_tables(
        &connection.open_session()?,
        index_name,
        &Some(DropOptionsBuilder::default().set_force().into()),
    )
    .map_err(|e| e.into())
}
