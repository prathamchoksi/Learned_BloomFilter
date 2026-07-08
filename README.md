# Learned Bloom Filter

This project implements a complete **Learned Bloom Filter** pipeline for URL classification using character-trigram hashing, logistic regression, and a backup Bloom Filter for classifier false negatives.

The project includes model training, optimized fused inference in both Python and Native C++, benchmarking, memory analysis, threshold trade-off evaluation, and visualization of experimental results.

---

## Features

- URL preprocessing and dataset preparation
- Word-token and character-trigram hashing baselines
- Logistic Regression classifier
- Fused inference implementation in Python
- Native C++ fused scorer exposed through pybind11
- Backup Bloom Filter storing classifier false negatives
- Learned Bloom Filter vs Standard Bloom Filter benchmarking
- Memory usage analysis across multiple configurations
- Threshold trade-off analysis
- Automatic generation of benchmark plots

---

## Project Structure

```
configs/        Experiment configurations
cpp/            Native C++ implementation and pybind11 bindings
data/           Dataset files
docs/           Project documentation
results/        Benchmark outputs, plots, and exported models
scripts/        Training, benchmarking, and plotting scripts
src/python/     Python package implementation
tests/          Unit tests
```

---

## Installation

Create a virtual environment and install the project dependencies:

```bash
pip install -r requirements.txt
```

Build the Native C++ extension:

```bash
pip install -e . --no-build-isolation
```

---

## Running the Pipeline

Run the scripts in the following order:

```bash
python scripts/preprocess_urls.py

python scripts/train_model.py

python scripts/benchmark.py

python scripts/benchmark_fused_cpp.py --impl python

python scripts/benchmark_fused_cpp.py --impl cpp

python scripts/build_false_negative_filter.py

python scripts/benchmark_learned_vs_standard.py

python scripts/benchmark_memory_variants.py

python scripts/benchmark_threshold_tradeoff.py
```

Generate the visualizations:

```bash
python scripts/plot_memory_variants.py

python scripts/plot_threshold_tradeoff.py

python scripts/plot_memory_reduction_heatmap.py
```

---

## Results

The pipeline generates:

- Trained model weights and metadata
- Benchmark summaries
- Learned Bloom Filter metadata
- Memory usage analysis
- Threshold trade-off analysis
- Publication-quality plots

All generated outputs are stored in the `results/` directory.

---

## Notebook

A complete end-to-end demonstration of the project, including benchmarking, visualizations, and experimental observations, is provided in:

```
Learned_Bloom_Filter_Pipeline.ipynb
```

---

## Technologies Used

- Python
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- C++
- pybind11