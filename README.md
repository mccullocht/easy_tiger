# EasyTiger

A DiskANN/Vamana implementation using WiredTiger as the storage layer.

The vector index is modeled as two tables parallel to an existing collection, where one table
contains the raw float vector and graph edges and the other contains quantized vectors -- binary
quantized vectors in our case. This structure should allow us to serve larger than memory indices
efficiently and with relatively low latency. See the [design](docs/design.md) for more details.

This implementation will build the WiredTiger library and provides a crude Rust wrapper around
WT in the `wt_mdb` crate. See [implementation notes](docs/implementation_notes.md) if you'd like
to know more about the WT wrapper.

## Performance Testing

Tests were run using a [Qdrant OpenAI dbpedia dataset](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M) of 1M vectors with 3072 dimensions.
The raw vectors for this index occupy about 12GB.

Preliminary performance numbers on an M2 Mac with 32GB RAM searching for 128 candidates:

| Configuration                                      | Avg Latency | Recall@10 |
| -------------------------------------------------- | ----------- | --------- |
| EasyTiger, 16GB cache                              | 3.4 ms      | 0.967     |
| EasyTiger, 4GB cache                               | 10.4 ms     | 0.967     |
| EasyTiger, 4GB cache, read concurrency 4           | 7.2 ms      | 0.971     |
| Lucene 9.12, Float HNSW, 8 segments                | 15 ms       | 0.960     |
| Lucene 9.12, Float HNSW, 8 segments, intersegment  | 2.6 ms      | 0.969     |
| Lucene 9.12, Float HNSW, 1 segment                 | 2.6 ms      | 0.965     |

Results were compared using float vectors to maximize recall figures, Lucene has other
configurations that trade accuracy for memory consumption. At the moment there are no
framework-level components for re-ranking using float vectors even when they are available
on disk.

EasyTiger is competitive on accuracy and latency when the entire index fits in the WiredTiger
cache. When the cache size is reduced latency is still reasonable as the cache will often contain
vertices closer to the entry point. Some of the latency effects can be mitigated by using a read
concurrency feature that speculatively reads multiple vertices in parallel. This also increases
recall as we tend to perform more vector comparisons in this case. A significant chunk of the CPU
time is spent looking up vectors in the quantized vector (nav) table since we might do thousands
of these lookups per query. This could be mitigated by denormalizing the quantized vectors into
the graph edges -- basically make the graph vertices much larger (increase index size 2-4x) but
reduce the number of reads.

Lucene is much slower in the default configuration as it applies the 128 candidate budget to
the underlying graph in each segment, and performs many more vector comparisons as a result.
This can't be mitigated with statistical approaches to reducing budget as the distribution of
vectors to segments is not ~random. The latency impact can be mitigated by using intersegment
concurrency, but the cost of this query will still be roughly 15ms of CPU time. The additional
vector comparisons also mean that a query over a larger than memory index will suffer more due
to additional vector reads; testing this would require running the setup in a memory container.
Lucene performance is great with 1 segment -- it is a more efficient implementation than the WT
version -- but this isn't representative of how Lucene is typically used.