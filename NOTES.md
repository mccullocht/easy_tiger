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

