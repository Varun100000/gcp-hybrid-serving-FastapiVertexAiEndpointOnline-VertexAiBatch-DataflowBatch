"""
Create JSON Lines input for Vertex AI Batch Prediction.

Vertex AI Batch Prediction expects each prediction instance as a separate row.
JSONL is a convenient format because each line is one JSON object.

Input:
- data/head_products.csv

Output:
- batch-input/head_products.jsonl
"""

from pathlib import Path
import json

import pandas as pd


HEAD_PRODUCTS_FILE = Path("data/head_products.csv")
OUTPUT_DIR = Path("batch-input")
OUTPUT_FILE = OUTPUT_DIR / "head_products.jsonl"

TOP_K = 5


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    head_products_df = pd.read_csv(HEAD_PRODUCTS_FILE)

    with OUTPUT_FILE.open("w", encoding="utf-8") as output_file:
        for product_id in head_products_df["product_id"]:
            row = {
                "product_id": product_id,
                "top_k": TOP_K,
            }

            output_file.write(json.dumps(row) + "\n")

    print(f"Wrote {len(head_products_df)} prediction instances")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()