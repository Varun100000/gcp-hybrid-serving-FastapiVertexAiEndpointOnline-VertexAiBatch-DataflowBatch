"""
Dataflow batch scoring pipeline.

This is Option B for batch serving.

Instead of using managed Vertex AI Batch Prediction, this pipeline:
1. Reads head products from BigQuery.
2. Loads the trained recommendation model from Cloud Storage.
3. Scores products in parallel across Dataflow workers.
4. Writes recommendations to a BigQuery lookup table.

For the demo, keep input small and max workers low to control cost.
"""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import apache_beam as beam
from apache_beam.io import ReadFromBigQuery, WriteToBigQuery
from apache_beam.io.filesystems import FileSystems
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import SetupOptions

from src.recommendation_model import ProductRecommendationModel


class ScoreProductFn(beam.DoFn):
    """
    Scores one product_id.

    setup() runs once per worker process.
    That is where we load the model locally.

    process() runs for each row assigned to that worker.
    """

    def __init__(self, model_gcs_path: str, top_k: int):
        self.model_gcs_path = model_gcs_path
        self.top_k = top_k
        self.model = None

    def setup(self):
        """
        Download model.joblib from GCS to the worker's local disk
        and load it once.

        This demonstrates the pattern:
        each Dataflow worker keeps a local copy of the model.
        """

        local_model_path = Path(tempfile.gettempdir()) / "model.joblib"

        with FileSystems.open(self.model_gcs_path) as source_file:
            with local_model_path.open("wb") as local_file:
                local_file.write(source_file.read())

        self.model = ProductRecommendationModel(model_path=local_model_path)

    def process(self, row: dict[str, Any]):
        """
        Score one product row and emit one BigQuery row.
        """

        product_id = row["product_id"]

        prediction = self.model.recommend(
            product_id=product_id,
            top_k=self.top_k,
        )

        yield {
            "product_id": prediction["product_id"],
            "recommendations_json": json.dumps(prediction["recommendations"]),
            "error": prediction["error"],
            "batch_source": "dataflow_batch_scoring",
        }


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project_id", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--model_gcs_path", required=True)
    parser.add_argument("--input_table", required=True)
    parser.add_argument("--output_table", required=True)
    parser.add_argument("--top_k", type=int, default=5)

    known_args, pipeline_args = parser.parse_known_args()

    query = f"""
        SELECT product_id
        FROM `{known_args.input_table}`
    """

    table_schema = {
        "fields": [
            {"name": "product_id", "type": "STRING", "mode": "REQUIRED"},
            {"name": "recommendations_json", "type": "STRING", "mode": "NULLABLE"},
            {"name": "error", "type": "STRING", "mode": "NULLABLE"},
            {"name": "batch_source", "type": "STRING", "mode": "NULLABLE"},
        ]
    }

    pipeline_options = PipelineOptions(pipeline_args)

    google_cloud_options = pipeline_options.view_as(GoogleCloudOptions)
    google_cloud_options.project = known_args.project_id
    google_cloud_options.region = known_args.region

    setup_options = pipeline_options.view_as(SetupOptions)
    setup_options.save_main_session = True

    with beam.Pipeline(options=pipeline_options) as pipeline:
        (
            pipeline
            | "Read head products from BigQuery"
            >> ReadFromBigQuery(
                query=query,
                use_standard_sql=True,
            )
            | "Score products with local model per worker"
            >> beam.ParDo(
                ScoreProductFn(
                    model_gcs_path=known_args.model_gcs_path,
                    top_k=known_args.top_k,
                )
            )
            | "Write recommendations to BigQuery"
            >> WriteToBigQuery(
                known_args.output_table,
                schema=table_schema,
                write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
            )
        )


if __name__ == "__main__":
    run()