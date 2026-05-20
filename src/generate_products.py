"""
Generate a small synthetic product catalog for the hybrid serving demo.

Why synthetic data?
- Keeps the project simple.
- Avoids downloading large datasets.
- Keeps GCP cost low.
- Still lets us demonstrate realistic batch vs online serving behavior.

The catalog contains:
- product metadata
- popularity score
- head/tail product flag

Head products are the top 20% by popularity.
These will be precomputed by batch serving.

Tail products are the remaining 80%.
These will fall back to online serving.
"""

from pathlib import Path
import random

import pandas as pd


# Keep the demo small to avoid unnecessary cloud cost.
# In the README, we can explain that production could have millions of rows.
NUM_PRODUCTS = 1000

# Top 20% products are considered "head" products.
HEAD_PERCENTAGE = 0.20

OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "products.csv"
HEAD_PRODUCTS_FILE = OUTPUT_DIR / "head_products.csv"
TAIL_PRODUCTS_FILE = OUTPUT_DIR / "tail_products.csv"


CATEGORIES = [
    "electronics",
    "home",
    "fashion",
    "sports",
    "beauty",
    "books",
    "toys",
    "grocery",
]

BRANDS = [
    "Nova",
    "UrbanCo",
    "Apex",
    "DailyPro",
    "Zenith",
    "BrightLife",
    "CoreLine",
    "FreshNest",
]


def generate_product(product_number: int) -> dict:
    """
    Create one fake product row.

    We intentionally use simple fields because the recommendation model
    will use these fields to calculate product similarity later.
    """

    category = random.choice(CATEGORIES)
    brand = random.choice(BRANDS)

    # Prices are loosely varied by category to make products feel realistic.
    base_price_by_category = {
        "electronics": 120,
        "home": 60,
        "fashion": 45,
        "sports": 75,
        "beauty": 35,
        "books": 20,
        "toys": 30,
        "grocery": 15,
    }

    base_price = base_price_by_category[category]
    price = round(random.uniform(base_price * 0.5, base_price * 1.8), 2)

    # Rating between 3.0 and 5.0.
    rating = round(random.uniform(3.0, 5.0), 1)

    # Popularity is intentionally skewed.
    # A few products should look much more popular than the rest.
    popularity_score = int(random.expovariate(1 / 1000))

    return {
        "product_id": f"P{product_number:05d}",
        "title": f"{brand} {category.title()} Item {product_number}",
        "category": category,
        "brand": brand,
        "price": price,
        "rating": rating,
        "popularity_score": popularity_score,
    }


def main() -> None:
    """
    Generate product data and write three CSV files:
    1. all products
    2. head products only
    3. tail products only
    """

    random.seed(42)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    products = [generate_product(i) for i in range(1, NUM_PRODUCTS + 1)]
    products_df = pd.DataFrame(products)

    # Rank products by popularity.
    products_df = products_df.sort_values(
        by="popularity_score",
        ascending=False,
    ).reset_index(drop=True)

    head_count = int(NUM_PRODUCTS * HEAD_PERCENTAGE)

    products_df["is_head_product"] = False
    products_df.loc[: head_count - 1, "is_head_product"] = True

    head_products_df = products_df[products_df["is_head_product"]].copy()
    tail_products_df = products_df[~products_df["is_head_product"]].copy()

    products_df.to_csv(OUTPUT_FILE, index=False)
    head_products_df.to_csv(HEAD_PRODUCTS_FILE, index=False)
    tail_products_df.to_csv(TAIL_PRODUCTS_FILE, index=False)

    print(f"Generated {len(products_df)} total products")
    print(f"Generated {len(head_products_df)} head products")
    print(f"Generated {len(tail_products_df)} tail products")
    print(f"Wrote {OUTPUT_FILE}")
    print(f"Wrote {HEAD_PRODUCTS_FILE}")
    print(f"Wrote {TAIL_PRODUCTS_FILE}")


if __name__ == "__main__":
    main()