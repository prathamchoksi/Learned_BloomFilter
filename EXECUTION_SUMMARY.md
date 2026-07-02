# Step 01–03 Execution Complete

## Summary

You have successfully implemented and benchmarked the first three major phases of the Learned Bloom Filter research pipeline:

### Step 01: Word-Token Hashing Pipeline (Python)

- **Feature Extractor**: HashingVectorizer with `analyzer='word'` (unigrams)
- **Model**: SGDClassifier with logistic regression loss
- **Mean Latency**: 154,726 ns/query
- **Throughput**: 6,463 QPS
- **Model Quality**: ROC-AUC 0.9920
- **Avg NNZ per URL**: 6.4 (sparse, low feature density)

### Step 02: Char-Trigram Hashing Pipeline (Python)

- **Feature Extractor**: HashingVectorizer with `analyzer='char'`, `ngram_range=(3, 3)`
- **Model**: SGDClassifier with same hyperparameters
- **Mean Latency**: 183,084 ns/query
- **Throughput**: 5,462 QPS
- **Model Quality**: ROC-AUC 0.9929 (better)
- **Avg NNZ per URL**: 43.4 (denser features)
- **Note**: Slightly slower latency due to more features, but better AUC

### Step 03: Fused Char-Trigram Inference (Pure Python with C++ Template)

- **Feature Extraction**: Char-trigram rolling hash (single-pass, no CSR matrix construction)
- **Scorer**: Fused inference without vectorization overhead
- **Mean Latency**: 50,390 ns/query
- **Throughput**: 19,845 QPS
- **Model Quality**: ROC-AUC 0.9926 (preserved)
- **Speedup vs Step 01**: **3.1x faster**
- **Speedup vs Step 02**: **3.6x faster** (183 µs → 50 µs per query)
- **Note**: Pure Python version; native C++ compiled path available when C++ toolchain is installed

## Project Artifacts

### Benchmarks

- `results/hash_benchmark/` — Step 01/02 results
  - `summary.json` — Overall report
  - `step_01_word_tokens.json` — Word tokenizer details
  - `step_02_char_trigrams.json` — Char-trigram details
- `results/fused_cpp_benchmark/summary.json` — Step 03 pure Python scorer
- `results/fused_model/` — Exported model weights
  - `weights.bin` — Binary weight matrix and intercept
  - `metadata.json` — Model hyperparameters

### Visualizations

- `results/hash_benchmark_plot.png` — Step 01 vs 02 side-by-side
- `results/step_01_02_03_comparison.png` — All three steps: latency, QPS, speedup, AUC

### Code Modules

- `src/python/learned_bloom/fused_trigram.py` — Fused model export and matrix building
- `src/python/learned_bloom/fused_scorer_py.py` — Pure Python rolling-hash scorer
- `scripts/train_model.py` — Training and export pipeline
- `scripts/benchmark.py` — Step 01/02 benchmark
- `scripts/benchmark_fused_cpp.py` — Step 03 benchmark (with C++ fallback)
- `scripts/plot_hash_benchmark.py` — Step 01/02 plot
- `scripts/plot_step_comparison.py` — All three steps comparison plot
- `cpp/src/fused_trigram_scorer.cpp` — C++ native scorer (ready to compile)
- `cpp/include/fused_trigram_scorer.hpp` — C++ header

## Key Insights

1. **Feature Density Trade-off**: Word tokens (6.4 NNZ) are much sparser than char trigrams (43.4 NNZ), leading to faster vectorization in Step 01, but Step 02's richer features yield better model quality.

2. **Inference Optimization**: Eliminating CSR matrix construction in Step 03 provides a **3.6x speedup** over the sklearn pipeline, even using pure Python. The native C++ path (when built) is expected to provide another 10–50x reduction.

3. **Consistent Model Quality**: All three methods preserve ROC-AUC ≈ 0.99, confirming that the optimization is pure latency/throughput without sacrificing model quality.

## Next Steps (Optional)

1. **Compile Native C++ Extension**: Build the `learned_bloom_cpp` module on a machine with a C++ toolchain (MSVC, g++, clang) to replace the pure Python scorer with the native implementation.

2. **Backup Bloom Filter Integration**: Implement the learned filter composition:
   - Run the model to identify false negatives
   - Store only false negatives in a standard Bloom filter
   - Query: `if model_positive(url) return True; else return backup_bloom.contains(url)`

3. **Visualization Layer**: Create live dashboards or notebooks comparing:
   - Memory footprint (model + backup Bloom vs standard Bloom)
   - FPR at target operating points
   - Query throughput and p99 latency

4. **Final Benchmark**: Measure the full learned filter against a standard Bloom filter baseline on the same dataset.

---

**Date**: July 1, 2026  
**Dataset**: 420K URLs (337K train, 84K test)  
**Hardware**: Windows, Python 3.14.2, scikit-learn 1.5+
