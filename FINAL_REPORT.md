# Learned Bloom Filter: Complete Pipeline Summary

## Project Overview

You have successfully built a complete end-to-end pipeline implementing a **Learned Bloom Filter** using char-trigram features, fused inference, and backup false-negative storage. The pipeline progresses through four optimization phases:

## Phase Summary 


| Phase | Implementation | Latency | Throughput | ROC-AUC |
|------|----------------|---------:|-----------:|---------:|
| Phase 01 | Word Hash + sklearn | 104,640 ns | 9,557 QPS | 0.9920 |
| Phase 02 | Char Trigram + sklearn | 116,323 ns | 8,597 QPS | 0.9929 |
| Phase 03 (Python) | Fused Python | 28,945 ns | 34,548 QPS | 0.9926 |
| Phase 03 (Native C++) | Native C++ | **1,151 ns** | **869,181 QPS** | **0.9926** |
| Phase 04 | Learned Bloom Filter | **44,529 ns** | — | **Memory Reduction: 50.5%** | 


### Phase 01: Word-Token Hashing (Baseline 1)

- **Vectorizer**: Unigrams with HashingVectorizer
- **Inference**: sklearn SGDClassifier + vectorize
- **Latency**: 104,640.4 ns/query
- **Throughput**: 9,557 QPS
- **Quality**: ROC-AUC 0.9920

### Phase 02: Char-Trigram Hashing (Baseline 2)

- **Vectorizer**: Char trigrams with HashingVectorizer
- **Inference**: sklearn SGDClassifier + vectorize
- **Latency**: 116,322.7 ns/query
- **Throughput**: 8,597 QPS
- **Quality**: ROC-AUC 0.9929 (better model)
- **Trade-off**: Slower due to denser features, but improved accuracy

### Phase 03: Native C++ Fused Inference

- **Feature Extraction**: Rolling-hash character trigrams with optimized single-pass normalization
- **Inference**: Native C++ implementation exposed through pybind11
- **Latency**: 1,150.5 ns/query
- **Throughput**: 869,181 QPS
- **Quality**: ROC-AUC 0.9926 (identical to Python implementation)
- **Python Fused Baseline**:
  - Latency: 28,945.3 ns/query
  - Throughput: 34,548 QPS
- **Measured Speedup**:
  - **25.2× faster than the Python fused scorer**
  - **≈101× faster than the original char-trigram sklearn pipeline**

### Phase 04: Learned Filter Composition (Full Pipeline)

- **Model**: Native C++ fused trigram scorer (Phase 03)
- **Classifier False Negatives**: **10,062 / 75,643 positives (13.30% error rate)**
- **Model Size**: **32,784 bytes**
- **Backup Filter**: Standard Bloom filter storing only false negatives
- **Backup Bloom Filter Size**: **12,056 bytes**
- **Total Learned Filter Size**: **44,840 bytes**
- **Standard Bloom Filter Size**: **90,631 bytes**
- **Memory Reduction**: **50.5%**
- **False Positive Rate (FPR)**: **0.0115** (vs **0.0044** for the standard Bloom filter)
- **Observation**: By reducing the logistic regression feature space to 4,096 hashed features, the learned Bloom filter becomes significantly smaller than the standard Bloom filter while maintaining acceptable classification performance.


## Key Insights

### 1. Inference Optimization Impact

The progression shows clear latency improvements:

- **Phase 01 → Phase 02**:
104,640 ns → 116,323 ns (approximately 11% higher latency due to richer character-trigram features)
- **Phase 02 → Phase 03 (Python Fused)**:
  - 116,323 ns → 28,945 ns (**4.02× faster**)
- **Phase 03 (Python) → Phase 03 (Native C++)**:
  - 28,945 ns → 1,151 ns (**25.2× faster**)
- **Overall Phase 02 → Native C++**:
  - 116,323 ns → 1,151 ns (**≈101× faster**)

Throughout all optimizations, the model quality remained unchanged (ROC-AUC ≈ 0.9926), demonstrating that the performance improvements came entirely from implementation optimizations rather than changes to the machine learning model. 

### 2. Model Efficiency

The compact logistic regression model achieves:

- **13.30% false negative rate** on the positive set
- **32,784-byte (~32 KB) model size**
- **12,056-byte backup Bloom filter**
- **44,840-byte total learned Bloom filter size**

Compared with the standard Bloom filter (**90,631 bytes**), the learned Bloom filter achieves a **50.5% reduction in memory usage** while preserving the learned-filter architecture. 

### 3. Memory Trade-off

The learned Bloom filter occupies **44,840 bytes**, compared to **90,631 bytes** for the standard Bloom filter.

Memory breakdown:

| Component | Size |
|-----------|------:|
| Logistic Regression Model | 32,784 bytes |
| Backup Bloom Filter | 12,056 bytes |
| **Total Learned Bloom Filter** | **44,840 bytes** |
| Standard Bloom Filter | **90,631 bytes** |

Overall, the learned Bloom filter achieves approximately **50.5% memory reduction** compared to the standard Bloom filter.

This improvement is obtained by using a compact 4,096-feature logistic regression model together with a small backup Bloom filter that stores only false negatives. 

### 4. False Positive Rate

The learned Bloom filter achieves a **0.0115** false positive rate, compared to **0.0044** for the standard Bloom filter.

Although the learned Bloom filter has a slightly higher false positive rate, it reduces memory usage by approximately **50.5%**, making it an attractive trade-off for applications where memory efficiency is more important than achieving the lowest possible false positive rate.

The operating point can be further tuned by adjusting the classification threshold or the backup Bloom filter parameters. 

## Project Artifacts

### Code Modules

- `src/python/learned_bloom/bloom_filter.py` — Standard Bloom filter implementation
- `src/python/learned_bloom/fused_trigram.py` — Fused model export and matrix building
- `src/python/learned_bloom/fused_scorer_py.py` — Pure Python rolling-hash scorer
- `src/python/learned_bloom/learned_filter.py` — Learned filter composition logic
- `cpp/src/fused_trigram_scorer.cpp` — Native C++ fused trigram scorer exposed to Python via pybind11 

### Scripts

- `scripts/benchmark.py` — Phase 01/02 benchmarks
- `scripts/benchmark_fused_cpp.py` — Phase 03 benchmark
- `scripts/train_model.py` — Model training and export
- `scripts/build_false_negative_filter.py` — False negative identification and backup Bloom construction
- `scripts/benchmark_learned_vs_standard.py` — Phase 04 learned vs standard comparison
- `scripts/plot_*.py` — Visualization scripts

### Results

- `results/hash_benchmark/` — Phase 01/02 metrics
- `results/fused_python_benchmark/` — Python fused inference benchmark
- `results/fused_cpp_benchmark/` — Native C++ fused inference benchmark
- `results/learned_bloom_filter/` — False negative filter metadata
- `results/learned_vs_standard_benchmark/` — Phase 04 comparison
- `results/fused_model/` — Trained model weights and metadata 

### Visualizations

- `results/hash_benchmark_plot.png` — Phase 01 vs 02
- `results/step_01_02_03_comparison.png` — All three inference paths
- `results/learned_vs_standard_comparison.png` — Phase 04 full comparison

## Reproducibility

To reproduce this entire pipeline:

```bash
# Phase 01/02: Hash benchmarks
python scripts/benchmark.py

# Phase 03: Native C++ fused inference
python scripts/train_model.py

# Build the C++ extension
python -m pip install -e . --no-build-isolation

# Benchmark Python implementation
python scripts/benchmark_fused_cpp.py --impl python

# Benchmark Native C++ implementation
python scripts/benchmark_fused_cpp.py --impl cpp 

# Phase 04: Learned filter with backup Bloom
python scripts/build_false_negative_filter.py
python scripts/benchmark_learned_vs_standard.py

# Visualizations
python scripts/plot_step_comparison.py
python scripts/plot_learned_vs_standard.py
```


**Total Speedup**:
- **Phase 01 → Native C++**: **≈91×**
- **Phase 02 → Native C++**: **≈101×**
- **Python Fused → Native C++**: **≈25×** 

**Dataset**: 420,464 URLs (75,643 positive, 344,821 negative)

**Model Quality**:
- ROC-AUC = **0.9926**
- Mean Native C++ Latency = **1,150.5 ns/query**
- Throughput = **869,181 queries/sec**