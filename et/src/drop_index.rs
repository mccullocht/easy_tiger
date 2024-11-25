use std::{io, sync::Arc};

use easy_tiger::wt::WiredTigerGraphVectorIndex;
use wt_mdb::{options::DropOptionsBuilder, Connection};

pub fn drop_index(connection: Arc<Connection>, index_name: &str) -> io::Result<()> {
    let index = WiredTigerGraphVectorIndex::from_db(&connection, index_name)?;
    let session = connection.open_session()?;
    session.drop_record_table(
        index.graph_table_name(),
        Some(DropOptionsBuilder::default().set_force().into()),
    )?;
    session.drop_record_table(
        index.nav_table_name(),
        Some(DropOptionsBuilder::default().set_force().into()),
    )?;
    Ok(())
}
