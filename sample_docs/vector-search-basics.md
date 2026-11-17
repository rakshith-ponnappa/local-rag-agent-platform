# Vector Search Basics

An embedding model turns text into a fixed-length vector. Texts with similar meaning produce vectors that point in similar directions.

Cosine similarity measures the angle between two vectors. If vectors are normalized to unit length, cosine similarity equals the dot product.

A brute-force search compares the query vector with every stored vector. This is exact and simple, and fast enough for a few thousand chunks.

Approximate nearest neighbor indexes such as HNSW trade a little recall for much faster search over millions of vectors.

Chunking splits long documents into smaller passages before embedding. Overlap between neighbouring chunks keeps sentences that cross a boundary retrievable.
