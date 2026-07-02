# Experiment Plan

## Goals

- Match target FPR bands
- Reduce memory versus standard Bloom filter
- Reduce query latency for end-to-end inference

## Steps

1. Train logistic regression with char trigram features
2. Sweep thresholds to control false negatives
3. Build backup Bloom filter from false negatives
4. Compare against standard Bloom filter baseline
5. Report memory, FPR, and throughput
