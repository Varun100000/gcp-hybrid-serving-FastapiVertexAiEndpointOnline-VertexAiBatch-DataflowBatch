"""
Quick local test for the saved recommendation model.

This confirms the model can load and return similar products
before we move anything to GCP.
"""
"""
Quick local test for the shared recommendation module.
"""

from recommendation_model import recommend


TEST_PRODUCT_ID = "P00042"


def main() -> None:
    response = recommend(TEST_PRODUCT_ID, top_k=5)

    print(f"Recommendations for {TEST_PRODUCT_ID}:")
    print(response)


if __name__ == "__main__":
    main()