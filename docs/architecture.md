# Architecture

## Components

1. URL preprocessing pipeline
2. Char-trigram logistic regression model
3. Backup Bloom filter storing model false negatives
4. Learned Bloom query router (model first, backup filter second)
5. Benchmark harness for memory, FPR, and latency

## Query Flow

1. Normalize incoming URL
2. Compute model score
3. If score >= threshold, return member
4. Else check backup Bloom filter and return result
