use std::{io, ops::Range, str::FromStr, sync::Arc};

use clap::Args;
use easy_tiger::vamana::wt::TableGraphVectorIndex;
use indicatif::ProgressIterator;
use wt_mdb::{Connection, RecordCursor};

use crate::ui::progress_spinner;

#[derive(Debug, Clone)]
struct KeyRange(Range<i64>);

impl FromStr for KeyRange {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts = s.split("-").collect::<Vec<_>>();
        if parts.len() != 2 {
            return Err("must have exactly two parts");
        }
        let start = parts[0]
            .parse::<i64>()
            .map_err(|_| "failed to parse start key")?;
        if start < 0 {
            return Err("start must be >= 0");
        }
        let end = parts[1]
            .parse::<i64>()
            .map_err(|_| "failed to parse end key")?;
        Ok(Self(start..end))
    }
}

#[derive(Args)]
pub struct DeleteArgs {
    /// First key to delete from the index.
    #[arg(long)]
    keys: KeyRange,
}

fn delete_all(
    id_cursor: RecordCursor<'_>,
    mut graph_cursor: RecordCursor<'_>,
    mut nav_cursor: RecordCursor<'_>,
) -> wt_mdb::Result<()> {
    for record in id_cursor.progress_with(progress_spinner("deleting vectors")) {
        let key = record?.0;
        graph_cursor.remove(key)?;
        nav_cursor.remove(key)?;
    }
    Ok(())
}

pub fn delete(connection: Arc<Connection>, index_name: &str, args: DeleteArgs) -> io::Result<()> {
    let index = Arc::new(TableGraphVectorIndex::from_db(&connection, index_name)?);
    let session = connection.open_session()?;
    session.begin_transaction(None)?;

    // Use the nav table to find the ids in range to delete. This is probably cheaper than iterating over the graph table.
    let mut id_cursor = session.open_record_cursor(index.nav_table().name())?;
    id_cursor.set_bounds(args.keys.0.clone())?;
    match delete_all(
        id_cursor,
        session.open_record_cursor(index.graph_table_name())?,
        session.open_record_cursor(index.nav_table().name())?,
    ) {
        Ok(_) => {
            println!("Deleted {:?}", args.keys.0);
            session.commit_transaction(None)?;
        }
        Err(e) => {
            println!("Delete failed {e}");
            session.rollback_transaction(None)?;
        }
    }

    Ok(())
}
