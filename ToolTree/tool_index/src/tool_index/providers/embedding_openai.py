"""OpenAI embedding adapter — hosted `EmbeddingProvider`.

Supports two modes:

1. **Regular OpenAI** — ``OPENAI_API_KEY`` env or ``api_key`` ctor kwarg.
2. **Azure OpenAI** — set ``azure_endpoint`` (or ``AZURE_OPENAI_ENDPOINT``
   env). Azure uses *deployment names*, not model names, for the ``model``
   parameter on requests — the ``azure_deployment`` kwarg (or
   ``AZURE_EMBEDDINGS_DEPLOYMENT_NAME`` env) controls that.

Client is loaded lazily (on first call) so the SDK isn't required unless
this provider is actually used. ``dim`` must match the hosted model's
true output dimensionality — downstream code trusts it when allocating
numpy arrays.
"""
from __future__ import annotations
import os


class OpenAIEmbeddingProvider:
    """Hosted embedding adapter via the OpenAI / Azure OpenAI SDK."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        azure_api_version: str | None = None,
        azure_deployment: str | None = None,
    ):
        """
        Args:
            model: Regular-OpenAI model name. Ignored when running against
                Azure (Azure uses ``azure_deployment`` instead).
            dim: Expected output dimension. Must match the real model's
                output size; a wrong value won't crash here but will break
                clustering silently.
            api_key: Explicit key. Falls back to ``OPENAI_API_KEY`` or
                ``AZURE_OPENAI_API_KEY`` (Azure mode).
            azure_endpoint: Azure resource base URL like
                ``https://<resource>.openai.azure.com/``. When set (or
                inferred from ``AZURE_OPENAI_ENDPOINT``), the provider
                switches to Azure mode.
            azure_api_version: Azure API version (e.g. ``"2024-02-01"``).
                Falls back to ``AZURE_OPENAI_API_VERSION``.
            azure_deployment: Name of the Azure deployment to target.
                This — not ``model`` — is what the SDK sends as the
                ``model`` field on embedding requests in Azure mode.
                Falls back to ``AZURE_EMBEDDINGS_DEPLOYMENT_NAME``.
        """
        # Azure-mode detection happens at construction so the provider's
        # `id` reflects the actual model used (useful in build traces).
        self.azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.azure_api_version = azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION")
        self.azure_deployment = azure_deployment or os.environ.get("AZURE_EMBEDDINGS_DEPLOYMENT_NAME")
        self._is_azure = bool(self.azure_endpoint)

        if self._is_azure:
            # `api_key` precedence: explicit → AZURE_OPENAI_API_KEY → OPENAI_API_KEY.
            self.api_key = (
                api_key
                or os.environ.get("AZURE_OPENAI_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )
            # Build trace wants a stable identifier — prefer the deployment
            # name, fall back to the model name for clarity.
            self.model = self.azure_deployment or model
            self.id = f"azure:{self.model}"
        else:
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            self.model = model
            self.id = model

        self.dim = dim
        self._client = None  # lazy; populated by `_ensure`

    def _ensure(self):
        """Lazy client construction so the SDK import is deferred.

        Missing credentials only error out at first call, not at
        config-load time — matches the behavior of the Anthropic adapter.
        """
        if self._client is not None:
            return
        if self._is_azure:
            from openai import AzureOpenAI  # type: ignore
            self._client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.azure_api_version,
            )
        else:
            from openai import OpenAI  # type: ignore
            self._client = OpenAI(api_key=self.api_key)

    def embed(self, text: str) -> list[float]:
        """Embed one string. Routes through `embed_batch` to keep the
        per-call path single-sourced."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed many strings in a single API call.

        Order of the returned list matches the input. The SDK handles
        rate limits and retries internally — we don't add another layer.
        """
        self._ensure()
        # Azure's SDK still takes a `model` kwarg, but its value must be
        # the *deployment name*, not the model name. We set
        # ``self.model`` accordingly in ``__init__``.
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]
