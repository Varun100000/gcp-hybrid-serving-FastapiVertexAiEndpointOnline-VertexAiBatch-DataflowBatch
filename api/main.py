"""
Cloud Run hybrid recommendation API.

This service hides the serving strategy from the client.

For every product_id:
1. Check BigQuery recommendation_lookup table.
2. If found, return precomputed batch recommendations.
3. If not found, call Vertex AI Endpoint for dynamic online prediction.

This demonstrates hybrid serving:
- static batch serving for head products
- dynamic online serving for tail products
"""

import json
import os
from typing import Any

from fastapi import FastAPI
from google.cloud import aiplatform
from google.cloud import bigquery
from pydantic import BaseModel


PROJECT_ID = os.environ.get("PROJECT_ID", "hybrid-serving-static-dynamic")
REGION = os.environ.get("REGION", "us-central1")
ENDPOINT_ID = os.environ.get("ENDPOINT_ID", "1320056617877635072")

BQ_DATASET = os.environ.get("BQ_DATASET", "hybrid_rec")
BQ_TABLE = os.environ.get("BQ_TABLE", "recommendation_lookup")

app = FastAPI(title="Hybrid Recommendation API")

bigquery_client = bigquery.Client(project=PROJECT_ID)


class RecommendationResponse(BaseModel):
    """
    Response returned to the client.

    source tells us which serving path was used:
    - batch_bigquery
    - online_vertex_ai
    """

    product_id: str
    source: str
    recommendations: list[dict[str, Any]]
    error: str | None = None


def get_batch_recommendations(product_id: str) -> list[dict[str, Any]] | None:
    """
    Look up precomputed recommendations from BigQuery.

    Returns:
    - list of recommendations if product exists in lookup table
    - None if product is not found
    """

    table_id = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"

    query = f"""
        SELECT recommendations_json
        FROM `{table_id}`
        WHERE product_id = @product_id
        LIMIT 1
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "product_id",
                "STRING",
                product_id,
            )
        ]
    )

    query_job = bigquery_client.query(query, job_config=job_config)
    rows = list(query_job.result())

    if not rows:
        return None

    return json.loads(rows[0]["recommendations_json"])


def get_online_recommendations(product_id: str, top_k: int) -> dict[str, Any]:
    """
    Call Vertex AI online endpoint for dynamic recommendation serving.
    """

    endpoint = aiplatform.Endpoint(
        endpoint_name=f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}"
    )

    response = endpoint.predict(
        instances=[
            {
                "product_id": product_id,
                "top_k": top_k,
            }
        ]
    )

    # Vertex returns a list of predictions because the endpoint supports
    # multiple instances in one request.
    return response.predictions[0]


@app.get("/health")
def health() -> dict[str, str]:
    """
    Health route for Cloud Run.
    """

    return {"status": "healthy"}


@app.get("/recommendations/{product_id}", response_model=RecommendationResponse)
def recommend(product_id: str, top_k: int = 5) -> RecommendationResponse:
    """
    Main hybrid serving route.

    Example:
    /recommendations/P00042?top_k=5
    """

    batch_recommendations = get_batch_recommendations(product_id)

    if batch_recommendations is not None:
        return RecommendationResponse(
            product_id=product_id,
            source="batch_bigquery",
            recommendations=batch_recommendations,
            error=None,
        )

    online_prediction = get_online_recommendations(
        product_id=product_id,
        top_k=top_k,
    )

    return RecommendationResponse(
        product_id=product_id,
        source="online_vertex_ai",
        recommendations=online_prediction.get("recommendations", []),
        error=online_prediction.get("error"),
    )