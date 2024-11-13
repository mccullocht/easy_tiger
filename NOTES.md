# WiredTiger-related implementation mistakes

* Default WT build sets HAVE_DIAGNOSTICS, which introduces random sleeps.
  Used the `cmake` crate to propagate profile from `cargo` and switch this off for non-debug builds.
* Transaction is bulk build. Without transactions each read is implicitly a separate transaction.
  Each nav vector read results in a read lock acquire/release for the transaction snapshot timestamp.
  Beginning a transaction before each search resulted in a 10x speedup.