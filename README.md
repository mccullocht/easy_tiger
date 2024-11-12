# EasyTiger

A DiskANN/Vamana implementation using WiredTiger as the storage layer.

# Design

An EasyTiger index for a vector field is modeled as two tables parallel to
the existing collection -- meaning that they share common keys. This column
grouping allows for more efficient cache usage, and prevents us from leaking
implementation details to users.

On top of this we implement a Vamana/DiskANN index -- a flat graph between
the indexed vectors that may be searched. Like in a typical Vamana index
we search the graph using quantized vectors (in our case bit vectors) to
navigate the graph during search, and re-rank using raw vectors read
alongside the graph edge data. Unlike a typical Vamana graph edges are
undirected -- all links are reciprocal -- which facilitates vertex updates
and deletes.

# Bulk Ingestion

The `bulk_load` tool accepts numpy formatted little-endian float vectors and
configuration parameters and upload this into WiredTiger tables. This tool
builds the graph outside of the database to speed ingestion as conflicts can
be resolved entirely in-memory.