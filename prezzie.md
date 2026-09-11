# Quantization Bound Estimates

Quantizers perform lossy compression of vectors and the quantized vector(s) can be used
to produce an _estimate_ of the distance between two vectors -- either symmetrically (both
quantized) or asymmetrically (float vector + quantized vector).

We can observed how accurate a quantizer is by examining the magnitude of the quantization
residual -- the euclidean distance between the original vector v and quantized vector q.

accuracy = ||v - q||

This figure can also be used as an arithmetic bound on the accuracy of a distance comparison.
For aggressive quantization (< 4 bits/dim) this bound is so large it is useless :(. However,
we can compute a _statistical_ bound. For unit vectors where the components are in a Gaussian
distribution the standard deviation is around 1/sqrt(dimensions) (blessing of dimensionality!)

bound = ||v - q|| / sqrt(D)

We can get vector components in a Gaussian distribution through rotation either by QR
decomposition of a random matrix (matmul; slow, accurate) or Fast Walsh-Hadamard Rotation
(FWHT; fast, less accurate). The output distribution is normal so we can adjust the bound
by a Z-score depending on our tolerance for failure -- Z=1.96 for 95% accurate, Z=3 for 99.5%.

This is all phrase in terms of euclidean distance, but if we unit normalize vectors before
computing cosine similarity it is trivial to scale the bounds into the right scale.

The output of distance computation is now (distance, bound) -- additional information we
may be able to apply in a variety of ways to improve the product.

# Auto tuned reranking

When using a very aggressive quantization function (1-2 bits/dimension) we need to read a
higher fidelity vector for some subset of the results and compute more accurate distances
to reorder the results. This requires over retrieving results and a simple multipler to
rerank more vectors than $limit. This works fine but sometimes it omits results that should
be reranked or more frequently pays to rerank non-competitive results.

We can use the bounds to automatically select the set of vectors to re-rank using a higher
fidelity vector representation. If the user wants $limit docs I can maintain a dual queue:

results: $limit ordered by _upper bound distance_.
overflow: additional results whose _lower bound distance_ is competitive.

This reacts automatically to the distribution of results within a query and the efficacy of
the quantizer -- more bits result in less reranking. This effect can be observed in exhaustive
recall tests.

# Auto selecting quantization

Sample a small set of "docs" (10000) and "queries" (1000) from a larger data set. Compute
ground truth rank for each query, then computed bound rerank depth exhaustively. Feed
this information into a _cost function_ to determine which quantizer is cheaper overall.

This observes some of the clustering behavior of the data set but the parameters change
as the size of the "docs" sample increases, but it's still reliable enough to extrapolate.
