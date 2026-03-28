use std::{io, sync::Arc};

use easy_tiger::vamana::wt::TableGraphVectorIndex;
use wt_mdb::{connection::DropOptionsBuilder, Connection};

pub fn drop_index(connection: Arc<Connection>, index_name: &str) -> io::Result<()> {
    for table_name in TableGraphVectorIndex::generate_table_names(index_name) {
        connection.drop_table(
            &table_name,
            Some(DropOptionsBuilder::default().set_force().into()),
        )?;
    }
    Ok(())
}
