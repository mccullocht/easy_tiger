# Design

An EasyTiger index for a vector field is modeled as two tables parallel to an existing collection
sharing the same keys. Vector indices do not have any inherent order so it does not make sense to
model them like traditional BTree indices.

A Vamana/DiskANN graph index structure is used, placing full fidelity vectors and graph edges in
one table and quantized vectors in another. Quantized vectors are used for navigating the graph and
are expected to be in memory. Full fidelity vectors and graph edges are in a second table and are
expected to be mostly on disk. This would allow you to search a vector data set using about
`num_dimensions * num_vectors / 8` bytes of RAM, plus some additional for caching graph data.

This implementation allows you to choose either Euclidean (L2) or dot product (~cos) distance
functions as well as all the hyper parameters mentioned in the DiskANN paper.

## Search

On top of this we implement a Vamana/DiskANN index -- a flat graph where each vertex is a vector and
edges are selected based on the distance between vertices. During a search we visit a vertex by
reading the vector value and edges from the graph table, then examine the outgoing edges and score
them against the query using the quantized vectors to populate a candidate priority queue. When the
search is complete we use the normalized vectors to compute high fidelity scores and rerank the
result set.

The search is greedy -- when a vertex is visited we process outgoing edges and immediately insert
them in the candidate queue before picking the next vertex to visit. This can be very slow if most
of the data set is on disk. This implementation allows vertex reads to occur in a thread pool, with
a concurrency limit set per-query. When concurrency is on the algorithm will pop candidates until
we reach the concurrency limit, issue concurrent reads, then process any results and update the
candidate list. This reduces latency by parallelizing reads but also results in more work -- more
vertices are read/visisted and more candidates are scored.

## Indexing

Unlike in a typical Vamana graph our edges are undirected, so all edges have a reciprocal link.
This simplifies processing of mutation operations (particularly deletions) but at the cost of
pruning more edges from the graph when we can't satisfty the undirected property. It is recommended
that higher settings of `beam_width` and `max_edges` be used with this implementation than you might
use with a directed graph implementation.

Insertion works as described in the paper. For deletions the directed graph property ensures that we
can find and remove all in-links to the deleted vertex. During deletions we may also use the edge
set from the deleted vertex to seed more links to maintain connectivity. Updates can be modeled as
a delete and an insertion on the same row, although we may be able to avoid the deletion work if the
updated vector is very similar to the existing vector.

We can also support "bulk" indexing where the graph is built entirely in memory and uploaded to
WiredTiger when indexing is complete. This only works for bootstrapping a new vector index when the
tables are empty or do not exist at all. This is significantly faster than the regular insertion
path because we don't have to traverse the BTree to locate vector data.

# Caveats

* This implementation assumes that each key in the collection contains at most 1 vector. Many
  applications are likely to model as 1-to-many. I think this could be addressed with a compound
  key -- 64 bits of row key + 64 bits of vector hash so that it is still possible to perform
  predicate joins easily, although it will be difficult to estimate cardinality.
* Quantization is inflexible -- only binary quantization is supported. Stateful quantization
  algorithms are more difficult to maintain on mutable indices.
* This implementation does not allow pre-quantized inputs for the "full fidelity" vectors or the
  quantized vectors, even though some model may produce quantized vectors.
* There are no provisions to optimize the graph entry point during continuous operations.
* Reachability of any given vector is not guaranteed. The graph is undirected so it is easy to
  identify vertices that have no edges and are completely unreachable, but if we end up with
  subgraphs that is more difficult to detect.

# Future Work

Quantized vector lookup is a significant cost during search. This work could be parallelized, or
we could denormalize quantized vectors into the graph table. Denormalizing these vectors would
likely come at a cost of 2-4x storage requirements, but would also reduce lookup count by a factor
of 10x or more and also significantly reduce memory requirements.
