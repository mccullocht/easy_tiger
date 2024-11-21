use std::sync::Arc;

use thread_local::ThreadLocal;
use threadpool::ThreadPool;
use wt_mdb::{Connection, Session};

struct SessionSupplier {
    connection: Arc<Connection>,
    session: ThreadLocal<Session>,
}

impl SessionSupplier {
    fn new(connection: Arc<Connection>) -> Self {
        Self {
            connection,
            session: ThreadLocal::new(),
        }
    }

    fn get(&self) -> &Session {
        // XXX bare unwrap of session womp womp.
        self.session
            .get_or(|| self.connection.open_session().unwrap())
    }
}

// XXX docos
// XXX should this be in the wt_mdb crate?
#[derive(Clone)]
pub struct WorkerPool {
    thread_pool: ThreadPool,
    supplier: Arc<SessionSupplier>,
}

// XXX I would have a clone of one of these in my PGVIR implementation.
// on lookup() I would create a new closure that accepts the vertex_id and some
// graph metadata, create a scoped cursor, do the lookup, invoke the user's
// callback. in my concurrent searcher this would just queue the result in
// a channel that I read from later.
impl WorkerPool {
    pub fn new(thread_pool: ThreadPool, connection: Arc<Connection>) -> Self {
        Self {
            thread_pool,
            supplier: Arc::new(SessionSupplier::new(connection)),
        }
    }

    pub fn execute<C>(&self, closure: C)
    where
        C: FnOnce(&Session) + Send + Sync + 'static,
    {
        let supplier = self.supplier.clone();
        self.thread_pool.execute(move || closure(supplier.get()));
    }
}
