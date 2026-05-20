"""
Vertex AI custom prediction container using FastAPI.

Vertex AI will call:
- GET /health to check container health
- POST /predict to get predictions

Expected request:
{
    "instances": [
        {"product_id": "P00042", "top_k": 5}
    ]
}

Expected response:
{
    "predictions": [
        {
            "product_id": "P00042",
            "recommendations": [...],
            "error": null
        }
    ]
}
"""

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from recommendation_model import ProductRecommendationModel


MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/app/model/model.joblib"))

# Load the model once when the container starts.
# This avoids reloading model.joblib for every prediction request.
model = ProductRecommendationModel(model_path=MODEL_PATH)

app = FastAPI(title="Hybrid Recommendation Vertex Serving")


class PredictionInstance(BaseModel):
    """
    One prediction request item.

    product_id is required because recommendations are generated for one product.
    top_k is optional and defaults to 5 recommendations.
    """

    product_id: str
    top_k: int = 5


class PredictionRequest(BaseModel):
    """
    Vertex AI sends prediction requests as a list of instances.
    """

    instances: list[PredictionInstance]


@app.get("/health")
def health() -> dict[str, str]:
    """
    Health check used by Vertex AI.
    """

    return {"status": "healthy"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, list[dict[str, Any]]]:
    """
    Prediction route used by Vertex AI.

    Each instance is converted into one recommendation response.
    """

    predictions = []

    for instance in request.instances:
        prediction = model.recommend(
            product_id=instance.product_id,
            top_k=instance.top_k,
        )
        predictions.append(prediction)

    return {"predictions": predictions}