#!/usr/bin/env python3
"""Test C++ module loading and functionality."""

import sys
from pathlib import Path

root = Path(__file__).parent
sys.path.insert(0, str(root / "src" / "python"))

# Test C++ module
try:
    from learned_bloom import learned_bloom_cpp
    print("✓ C++ module loaded successfully!")
    
    # Create scorer
    scorer = learned_bloom_cpp.FusedTrigramScorer(str(root / "results" / "fused_model"))
    print("✓ C++ FusedTrigramScorer created")
    
    # Test score
    score = scorer.score("http://example.com")
    print(f"✓ Test score computed: {score:.4f}")
    
    # Test prediction
    pred = scorer.predict("http://example.com", 0.5)
    print(f"✓ Test prediction: {pred}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
