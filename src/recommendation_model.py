"""
Reusable recommendation logic.

This module is intentionally shared across multiple serving paths:

1. Local testing
2. Vertex AI online serving
3. Vertex AI Batch Prediction
4. Dataflow batch scoring

Keeping model logic in one place helps prove that batch and online serving
are using the same model behavior.
"""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


DEFAULT_MODEL_PATH = Path("model/model.joblib")


class ProductRecommendationModel:
    """
    Small wrapper around the saved nearest-neighbor model artifact.

    The class loads the model once and then serves repeated predictions.
    This matters for online serving and Dataflow workers because loading
    the model for every single request would be wasteful.
    """

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self.model_artifact = joblib.load(self.model_path)

        self.products_df: pd.DataFrame = self.model_artifact["products"]
        self.feature_transformer = self.model_artifact["feature_transformer"]
        self.nearest_neighbors = self.model_artifact["nearest_neighbors"]

    def recommend(self, product_id: str, top_k: int = 5) -> dict[str, Any]:
        """
        Return top_k similar products for one product_id.

        Output is a dictionary because this shape works well for:
        - REST APIs
        - Vertex AI prediction responses
        - BigQuery JSON-like records
        """

        matching_rows = self.products_df[
            self.products_df["product_id"] == product_id
        ]

        if matching_rows.empty:
            return {
                "product_id": product_id,
                "recommendations": [],
                "error": "unknown_product_id",
            }

        product_row = matching_rows.iloc[[0]]
        product_features = self.feature_transformer.transform(product_row)

        # Request top_k + 1 because the closest item is usually the product itself.
        distances, indices = self.nearest_neighbors.kneighbors(
            product_features,
            n_neighbors=top_k + 1,
        )

        recommendations = []

        for distance, index in zip(distances[0], indices[0]):
            candidate = self.products_df.iloc[index]

            # Do not recommend the same product back to the user.
            if candidate["product_id"] == product_id:
                continue

            recommendations.append(
                {
                    "recommended_product_id": candidate["product_id"],
                    "title": candidate["title"],
                    "category": candidate["category"],
                    "brand": candidate["brand"],
                    "similarity_score": round(1 - float(distance), 4),
                }
            )

            if len(recommendations) == top_k:
                break

        return {
            "product_id": product_id,
            "recommendations": recommendations,
            "error": None,
        }


# This module-level variable caches the loaded model.
# It starts as None and is loaded only when recommend() is called.
_model: ProductRecommendationModel | None = None


def get_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> ProductRecommendationModel:
    """
    Load the model once and reuse it.

    This is useful because:
    - online serving containers should not reload the model on every request
    - Dataflow workers should load the model once per worker process
    """

    global _model

    if _model is None:
        _model = ProductRecommendationModel(model_path)

    return _model


def recommend(
    product_id: str,
    top_k: int = 5,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """
    Convenience function for callers that only need one recommendation response.
    """

    model = get_model(model_path)
    return model.recommend(product_id=product_id, top_k=top_k)