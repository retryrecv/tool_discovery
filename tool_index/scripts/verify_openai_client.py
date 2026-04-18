"""Load `.env`, make one responses request, and make one embeddings request."""
from __future__ import annotations

from dotenv import load_dotenv
import os
import sys
from openai import AzureOpenAI


load_dotenv()
RESPONSE_MODEL = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
EMBEDDING_MODEL = os.environ.get("AZURE_EMBEDDINGS_DEPLOYMENT_NAME")
print("RESPONSE_MODEL:", RESPONSE_MODEL)
print("EMBEDDING_MODEL:", EMBEDDING_MODEL)

openAIClient = AzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
)

openAIEmbeddingClient = AzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
)


# def build_client():
#     azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
#     azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
#     azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")

#     return AzureOpenAI(
#         api_key=azure_api_key,
#         azure_endpoint=azure_endpoint,
#         api_version=azure_api_version,
#     )


def main() -> int:

    response = openAIClient.responses.create(
        model=RESPONSE_MODEL,
        input="Reply with exactly: ok",
    )
    print("Responses request: PASS")
    print(f"Response text: {response.output_text}")

    embedding = openAIEmbeddingClient.embeddings.create(
        model=EMBEDDING_MODEL,
        input="verify embedding request",
    )
    print("Embeddings request: PASS")
    print(f"Embedding length: {len(embedding.data[0].embedding)}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
