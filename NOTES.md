# Modeling WiredTiger in Rust

## Core Objects

The core objects are `WT_CONNECTION`, `WT_SESSION`, and `WT_CURSOR`, with analogous structs in the
Rust API that wrap these object-oriented C structs. The first two objects act as factories: a
`Connection` can create `Session`s, and a `Session` can create `Cursor`s.

In general the design attempts to ensure that the Rust struct is not dropped (and underlying C
struct is not closed) until there are no outstanding references. This means that a `Session` or
`Connection` will not be dropped until all corresponding `Cursor` or `Session` objects have been
dropped. To this end we do not expose `close()` as part of the public APIs as it would be difficult
to work with -- on failure it would have to return the object.

`Connection` does not have many common APIs and is thread-safe, so it is relatively simple to model
as a collection of immutable methods on the struct and is marked `Send + Sync`. Public APIs largely
work with `Arc<Connection>` to ensure that it lives as long as there are any outstanding `Session`s.

`Session` and `Cursor` are trickier to model. WiredTiger documentation suggests that these are not
safe for concurrent access by multiple threads (`!Sync) but should be fine to share across threads
(`Send`). In practice mutating a `Cursor` may change the underlying `Session`(!) so they are a unit.
These structs may be `Send` but should not be `Sync` -- additional synchronization is required.

`Cursor` contains a bunch of mutable methods for accessing data in the table and can even implement
`Iterator` easily enough. Accessing a `Cursor` does cause mutations to its parent `Session` so
ideally it would hold a mutable reference back to the `Session`. This is very difficult to use
though, as you can't call any `Session` methods or access a second `Cursor` until the first `Cursor`
is dropped. Using a reference count back to the `Session` is also problematic as that reference
needs to be immutable.

`Session` also contains a bunch of mutable methods for managing transactions and creating additional
cursors. Since `Connection` is thread safe we can keep an `Arc<Connection>` reference. Again,
having mutable methods on this struct makes it more difficult to work with cursors in a natural
way.

Working around all the likely requires `Session` methods to be immutable as otherwise we cannot
have multiple outstanding cursor simulataneously which isn't tenable. Beyond that we should
decide if we want `Session` and `Cursor` to be `Send` or not since that will affect possible
solutions:

`Send`: it is tempting to try to eliminate `Cursor`s reference to `Session` but that will lead
to unsound properties like allowing a `Session` and its `Cursor` to be sent to different threads.
In this case `Cursor` should maintain an immutable reference back to `Session`. Note that in
practice it would be very difficult to `Send` a `Cursor` since the borrowed `Session` value won't
live long enough in almost all cases.

`!Send`: in this case you could actually eliminate the `Cursor -> Session` reference through
refcounting if that makes coding abstractions any easier. In practice attempts to parallelize
work will involve having a thread-local `Session` object that we create `Cursor`s on.

## Configuration Strings/Options

To make APIs safer we provide structs that wrap configuration parameters, only exposing settings
that are documented. Typically there is a builder struct that allows setting parameters that can
be transformed `Into` an options struct that wraps an `Option<CString>` suitable for passing to
WiredTiger FFI. This reduces the odds of a fat-finger error.

# Modeling Concurrent Graph Search on WiredTiger

In DiskANN one approach to lower query latency is to issue the graph/raw vector reads concurrently:

1. Pop the top N candidates off the queue.
2. Issue disk reads for these candidates.
3. Process the responses and update result/candidate queues.
4. Repeat loop.

This is challenging to do on top of WiredTiger. Ideally you would do this within a single session
in order to get a consistent view. Unfortunately you perform cursor reads in two different
threads linked back to the same session -- these reads mutate the session. To do this the work
needs to be distributed to sessions on other threads.

# WiredTiger-related implementation mistakes

* Default WT build sets `-DHAVE_DIAGNOSTICS=1`, which introduces random sleeps.
  Used the `cmake` crate to propagate profile from `cargo` and switch this off for non-debug builds.
* Transaction is bulk build. Without transactions each read is implicitly a separate transaction.
  Each nav vector read results in a read lock acquire/release for the transaction snapshot timestamp.
  Beginning a transaction before each search resulted in a 10x speedup.
* Misunderstanding multi-threading. I read that `WT_SESSION` and `WT_CURSOR` cannot be access
  concurrently by multiple threads. I modeled WT in such a way that both sessions and cursors are
  `Send + Sync` but most methods are mutable so can really only be accessed by one thread at a time.
  I then wrote an abstraction that would generate cursors on a single session but use them from
  multiple threads. This segfaults due to cursors concurrently accessing session data.
