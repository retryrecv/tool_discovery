from .base import LLMProvider, EmbeddingProvider
from .cache import DiskCache
from .llm_fake import FakeLLMProvider
from .embedding_fake import FakeEmbeddingProvider


def make_llm(kind: str, **kwargs) -> LLMProvider:
    if kind == "fake":
        return FakeLLMProvider(**kwargs)
    if kind == "anthropic":
        from .llm_anthropic import AnthropicLLMProvider
        return AnthropicLLMProvider(**kwargs)
    raise ValueError(f"unknown llm kind: {kind}")


def make_embedding(kind: str, **kwargs) -> EmbeddingProvider:
    if kind == "fake":
        return FakeEmbeddingProvider(**kwargs)
    if kind == "openai":
        from .embedding_openai import OpenAIEmbeddingProvider
        return OpenAIEmbeddingProvider(**kwargs)
    if kind == "azure_openai":
        from .embedding_openai import OpenAIEmbeddingProvider
        # Same class, Azure mode auto-detected from env or from kwargs.
        # Caller can still pass azure_endpoint/azure_deployment/api_key explicitly.
        return OpenAIEmbeddingProvider(**kwargs)
    raise ValueError(f"unknown embedding kind: {kind}")


__all__ = [
    "LLMProvider", "EmbeddingProvider", "DiskCache",
    "FakeLLMProvider", "FakeEmbeddingProvider",
    "make_llm", "make_embedding",
]
