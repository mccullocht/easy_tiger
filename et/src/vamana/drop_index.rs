use std::{io, sync::Arc};

use easy_tiger::vamana::wt::TableGraphVectorIndex;
use wt_mdb::{session::DropOptionsBuilder, Connection};

pub fn drop_index(connection: Arc<Connection>, index_name: &str) -> io::Result<()> {
    let session = connection.open_session()?;
    for table_name in TableGraphVectorIndex::generate_table_names(index_name) {
        session.drop_table(
            &table_name,
            Some(DropOptionsBuilder::default().set_force().into()),
        )?;
    }
    Ok(())
}
