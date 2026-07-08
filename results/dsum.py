from pathlib import Path

weights = Path("results/fused_model/weights.bin")

print(weights.stat().st_size)