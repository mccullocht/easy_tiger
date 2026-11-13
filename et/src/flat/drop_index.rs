use std::{io, sync::Arc};

use easy_tiger::flat;
use wt_mdb::{Connection, connection::DropOptionsBuilder};

pub fn drop_index(connection: Arc<Connection>, index_name: &str) -> io::Result<()> {
    flat::drop_index(
        &connection,
        index_name,
        Some(DropOptionsBuilder::default().set_force().into()),
    )
}
