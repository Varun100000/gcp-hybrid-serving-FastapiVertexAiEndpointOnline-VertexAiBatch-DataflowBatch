"""
Train a tiny product similarity model.

This is not a deep learning model.
It is intentionally simple because the project is about MLOps serving patterns,
not complex model accuracy.

The model:
- reads product metadata from data/products.csv
- converts category and brand into one-hot encoded features
- scales numeric fields like price and rating
- finds nearest products using cosine distance
- saves everything needed for prediction into model/model.joblib
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_FILE = Path("data/products.csv")
MODEL_DIR = Path("model")
MODEL_FILE = MODEL_DIR / "model.joblib"

CATEGORICAL_FEATURES = ["category", "brand"]
NUMERIC_FEATURES = ["price", "rating"]


def main() -> None:
    """
    Train a nearest-neighbor similarity model and save it to disk.

    We save:
    - the original product table
    - the feature transformer
    - the nearest-neighbor model
    - the list of feature columns used
    """

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    products_df = pd.read_csv(DATA_FILE)

    # The transformer converts product metadata into numeric vectors.
    # OneHotEncoder handles text categories like "electronics" or "Nova".
    # StandardScaler normalizes numeric values so price does not dominate rating.
    feature_transformer = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
        ]
    )

    # Fit the transformer and produce numeric vectors for every product.
    product_features = feature_transformer.fit_transform(products_df)

    # NearestNeighbors lets us find similar products quickly.
    # cosine distance works well enough for simple similarity search.
    nearest_neighbors = NearestNeighbors(
        n_neighbors=6,
        metric="cosine",
    )

    nearest_neighbors.fit(product_features)

    model_artifact = {
        "products": products_df,
        "feature_transformer": feature_transformer,
        "nearest_neighbors": nearest_neighbors,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
    }

    joblib.dump(model_artifact, MODEL_FILE)

    print(f"Loaded {len(products_df)} products")
    print(f"Saved model artifact to {MODEL_FILE}")


if __name__ == "__main__":
    main()