from pathlib import Path


def main() -> None:
    input_path = Path("data/raw/urls.csv")
    output_path = Path("data/processed/urls_processed.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # TODO: Add URL normalization and cleaning.
    print(f"[preprocess] Input: {input_path}")
    print(f"[preprocess] Output: {output_path}")


if __name__ == "__main__":
    main()
