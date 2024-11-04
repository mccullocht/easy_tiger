use std::borrow::Cow;

use wt_mdb::{RecordCursor, RecordView, Result};

use crate::graph::NavVectorStore;

pub struct WiredTigerNavVectoreStore<'a> {
    cursor: RecordCursor<'a>,
}

impl<'a> WiredTigerNavVectoreStore<'a> {
    pub fn new(cursor: RecordCursor<'a>) -> Self {
        Self { cursor }
    }
}

impl<'a> NavVectorStore for WiredTigerNavVectoreStore<'a> {
    fn get(&mut self, node: i64) -> Option<Result<Cow<'_, [u8]>>> {
        Some(unsafe { self.cursor.seek_exact_unsafe(node)? }.map(RecordView::into_inner_value))
    }
}
