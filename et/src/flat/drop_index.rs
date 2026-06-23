use std::{io, sync::Arc};

use wt_mdb::{Connection, connection::DropOptionsBuilder};

use super::flat_table_name;

pub fn drop_index(connection: Arc<Connection>, index_name: &str) -> io::Result<()> {
    connection
        .drop_table(
            &flat_table_name(index_name),
            Some(DropOptionsBuilder::default().set_force().into()),
        )
        .map_err(io::Error::from)
}
