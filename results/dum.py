from pathlib import Path

print(Path("results/fused_model/weights.bin").stat().st_size)