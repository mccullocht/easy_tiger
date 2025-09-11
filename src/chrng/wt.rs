use std::{io, iter::FusedIterator, sync::Arc};

use vectors::QueryVectorDistance;
use wt_mdb::{Connection, Error, IndexCursorGuard, RecordCursorGuard, Result, Session};

use crate::{
    chrng::ClusterKey,
    vamana::{
        wt::{Leb128EdgeIterator, TableGraphVectorIndex, ENTRY_POINT_KEY},
        GraphConfig,
    },
};

pub struct HeadGraphCursor<'c>(RecordCursorGuard<'c>);

impl super::HeadGraphCursor for HeadGraphCursor<'_> {
    type EdgeIterator<'a>
        = Leb128EdgeIterator<'a>
    where
        Self: 'a;

    fn entry_point(&mut self) -> Result<i64> {
        let i64_bytes = unsafe {
            self.0
                .seek_exact_unsafe(ENTRY_POINT_KEY)
                .unwrap_or(Err(Error::not_found_error()))?
        };
        Ok(i64::from_le_bytes(i64_bytes.try_into().unwrap()))
    }

    fn edges(&mut self, vertex_id: i64) -> Result<Self::EdgeIterator<'_>> {
        let encoded_edges = unsafe {
            self.0
                .seek_exact_unsafe(vertex_id)
                .unwrap_or(Err(Error::not_found_error()))?
        };
        Ok(Leb128EdgeIterator::new(encoded_edges))
    }
}

pub struct HeadVectorDistanceCursor<'c> {
    cursor: RecordCursorGuard<'c>,
    dist: Box<dyn QueryVectorDistance + 'static>,
}

impl super::HeadVectorDistanceCursor for HeadVectorDistanceCursor<'_> {
    fn distance(&mut self, vertex_id: i64) -> Result<f64> {
        let vector = unsafe {
            self.cursor
                .seek_exact_unsafe(vertex_id)
                .unwrap_or(Err(Error::not_found_error()))?
        };
        Ok(self.dist.distance(vector))
    }
}

pub struct ClusterKeyEdgeIter<'a> {
    data: &'a [u8],
    prev: ClusterKey,
}

impl<'a> ClusterKeyEdgeIter<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            prev: ClusterKey {
                cluster_id: 0,
                vector_id: 0,
            },
        }
    }
}

impl Iterator for ClusterKeyEdgeIter<'_> {
    type Item = ClusterKey;

    fn next(&mut self) -> Option<Self::Item> {
        let cluster_delta = leb128::read::unsigned(&mut self.data).ok()?;
        let vector_delta = leb128::read::signed(&mut self.data).ok()?;
        if cluster_delta == 0 {
            self.prev.vector_id += vector_delta;
        } else {
            self.prev.cluster_id += cluster_delta as u32;
            self.prev.vector_id = vector_delta;
        }
        Some(self.prev)
    }
}

impl FusedIterator for ClusterKeyEdgeIter<'_> {}

pub struct TailGraphCursor<'c>(IndexCursorGuard<'c>);

impl super::TailGraphCursor for TailGraphCursor<'_> {
    type EdgeIterator<'a>
        = ClusterKeyEdgeIter<'a>
    where
        Self: 'a;

    fn edges(&mut self, vertex_id: ClusterKey) -> Result<Self::EdgeIterator<'_>> {
        let encoded_key = vertex_id.to_key_bytes();
        let encoded_edges = unsafe {
            self.0
                .seek_exact_unsafe(&encoded_key)
                .unwrap_or(Err(Error::not_found_error()))?
        };
        Ok(ClusterKeyEdgeIter::new(encoded_edges))
    }
}

pub struct TailVectorDistanceIter<'a, 'c, F: FnMut(ClusterKey) -> bool> {
    cursor: &'a mut TailVectorDistanceCursor<'c>,
    cluster_id: u32,
    filter_fn: F,
    bounds_set: bool,
}

impl<'a, 'c, F: FnMut(ClusterKey) -> bool> TailVectorDistanceIter<'a, 'c, F> {
    fn new(cursor: &'a mut TailVectorDistanceCursor<'c>, cluster_id: u32, filter: F) -> Self {
        Self {
            cursor,
            cluster_id,
            filter_fn: filter,
            bounds_set: false,
        }
    }
}

impl<F: FnMut(ClusterKey) -> bool> Iterator for TailVectorDistanceIter<'_, '_, F> {
    type Item = Result<(ClusterKey, f64)>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.bounds_set {
            self.cursor.cursor.reset().unwrap();
            self.bounds_set = true;
            let start = ClusterKey {
                cluster_id: self.cluster_id,
                vector_id: 0,
            };
            let end = ClusterKey {
                cluster_id: self.cluster_id,
                vector_id: i64::MAX,
            };
            if let Err(e) = self
                .cursor
                .cursor
                .set_bounds(start.to_key_bytes().as_slice()..end.to_key_bytes().as_slice())
            {
                return Some(Err(e));
            }
        }

        while let Some(record) = unsafe { self.cursor.cursor.next_unsafe() } {
            let (raw_key, vector) = match record {
                Ok((raw_key, vector)) => (raw_key, vector),
                Err(e) => return Some(Err(e)),
            };
            let key = ClusterKey::try_from(raw_key).expect("12 byte key");
            if (self.filter_fn)(key) {
                continue;
            }

            let dist = self.cursor.dist.distance(vector);
            return Some(Ok((key, dist)));
        }

        None
    }
}

impl<F: FnMut(ClusterKey) -> bool> Drop for TailVectorDistanceIter<'_, '_, F> {
    fn drop(&mut self) {
        if self.bounds_set {
            self.cursor.cursor.set_bounds(..).expect("unset bounds");
        }
    }
}

pub struct TailVectorDistanceCursor<'c> {
    cursor: IndexCursorGuard<'c>,
    dist: Box<dyn QueryVectorDistance + 'static>,
}

impl<'c> super::TailVectorDistanceCursor for TailVectorDistanceCursor<'c> {
    type VectorDistanceIter<'a, F: FnMut(ClusterKey) -> bool>
        = TailVectorDistanceIter<'a, 'c, F>
    where
        Self: 'a;

    fn distance(&mut self, vertex_id: ClusterKey) -> Result<f64> {
        let vertex_id = vertex_id.to_key_bytes();
        let vector = unsafe {
            self.cursor
                .seek_exact_unsafe(&vertex_id)
                .unwrap_or(Err(Error::not_found_error()))?
        };
        Ok(self.dist.distance(vector))
    }

    fn cluster_distance<F: FnMut(ClusterKey) -> bool>(
        &mut self,
        cluster_id: u32,
        filter: F,
    ) -> Self::VectorDistanceIter<'_, F> {
        TailVectorDistanceIter::new(self, cluster_id, filter)
    }
}

pub struct VectorIndex {
    index_name: String,
    head: TableGraphVectorIndex,
    tail: TableGraphVectorIndex,
}

impl VectorIndex {
    pub fn from_db(connection: &Arc<Connection>, index_name: &str) -> io::Result<Self> {
        Ok(Self {
            index_name: index_name.to_owned(),
            head: TableGraphVectorIndex::from_db(connection, &format!("{index_name}.head"))?,
            tail: TableGraphVectorIndex::from_db(connection, &format!("{index_name}.tail"))?,
        })
    }

    pub fn index_name(&self) -> &str {
        &self.index_name
    }

    pub fn config(&self) -> &GraphConfig {
        self.head.config()
    }
}

pub struct SessionIndexReader {
    session: Session,
    index: Arc<VectorIndex>,
}

impl SessionIndexReader {
    pub fn new(session: Session, index: Arc<VectorIndex>) -> Self {
        Self { session, index }
    }

    pub fn session(&self) -> &Session {
        &self.session
    }

    pub fn into_session(self) -> Session {
        self.session
    }
}

impl super::IndexReader for SessionIndexReader {
    type HeadGraphCursor<'a>
        = HeadGraphCursor<'a>
    where
        Self: 'a;
    type HeadVectorDistanceCursor<'a>
        = HeadVectorDistanceCursor<'a>
    where
        Self: 'a;
    type TailGraphCursor<'a>
        = TailGraphCursor<'a>
    where
        Self: 'a;
    type TailVectorDistanceCursor<'a>
        = TailVectorDistanceCursor<'a>
    where
        Self: 'a;

    fn head_graph_cursor(&self) -> Result<Self::HeadGraphCursor<'_>> {
        self.session
            .get_record_cursor(self.index.head.graph_table_name())
            .map(HeadGraphCursor)
    }

    fn head_vector_distance_cursor(
        &self,
        query: impl Into<Vec<f32>>,
    ) -> Result<Self::HeadVectorDistanceCursor<'_>> {
        let dist = self
            .index
            .config()
            .nav_format
            .query_vector_distance_f32(query.into(), self.index.config().similarity);
        self.session
            .get_record_cursor(self.index.head.nav_table().name())
            .map(|cursor| HeadVectorDistanceCursor { cursor, dist })
    }

    fn tail_graph_cursor(&self) -> Result<Self::TailGraphCursor<'_>> {
        self.session
            .get_index_cursor(self.index.tail.graph_table_name())
            .map(TailGraphCursor)
    }

    fn tail_vector_distance_cursor(
        &self,
        query: impl Into<Vec<f32>>,
    ) -> Result<Self::TailVectorDistanceCursor<'_>> {
        let dist = self
            .index
            .config()
            .nav_format
            .query_vector_distance_f32(query.into(), self.index.config().similarity);
        self.session
            .get_index_cursor(self.index.tail.nav_table().name())
            .map(|cursor| TailVectorDistanceCursor { cursor, dist })
    }
}
