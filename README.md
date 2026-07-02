# Learned Bloom Filter

Project scaffold for building and benchmarking a Learned Bloom Filter pipeline:

- URL preprocessing
- char trigram logistic regression model
- backup Bloom filter storing model false negatives
- C++ Bloom filter + pybind11 extension path

## Project Layout

- `configs/`: experiment configs
- `data/`: raw, processed, and split data files
- `scripts/`: runnable pipeline scripts
- `src/python/learned_bloom/`: Python package code
- `cpp/`: C++ Bloom filter and pybind bindings
- `tests/`: unit tests
- `docs/`: architecture and experiment docs
- `results/`: benchmark and evaluation outputs

## Quick Start

1. Create and activate a Python environment.
2. Install requirements:
   - `pip install -r requirements.txt`
3. Run pipeline scripts in order:
   - `python scripts/preprocess_urls.py`
   - `python scripts/train_model.py`
   - `python scripts/build_false_negative_filter.py`
   - `python scripts/benchmark.py`

## Notes

This scaffold is intentionally lightweight. Replace placeholders with your dataset paths and model/filter logic.
