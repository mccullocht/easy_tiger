use std::{io, sync::Arc};

use easy_tiger::wt::TableGraphVectorIndex;
use wt_mdb::{options::DropOptionsBuilder, Connection};

pub fn drop_index(connection: Arc<Connection>, index_name: &str) -> io::Result<()> {
    let session = connection.open_session()?;
    for table_name in TableGraphVectorIndex::generate_table_names(index_name) {
        session.drop_record_table(
            &table_name,
            Some(DropOptionsBuilder::default().set_force().into()),
        )?;
    }
    Ok(())
}
