"""Anthropic LLM adapter — hosted `LLMProvider` implementation.

Loaded lazily (SDK imported on first call) so the ``anthropic`` package
is only required when this provider is actually used.

Supports two ways to point at a non-default endpoint:
    1. Constructor args ``base_url`` / ``api_key``.
    2. Environment variables ``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_API_KEY``.

The env fallback exists so CI or ops can swap endpoints without editing
code. The constructor path is used by `scripts/verify_dynamic_tools.py`
to point at the local Agent Maestro proxy.
"""
from __future__ import annotations
import os


class AnthropicLLMProvider:
    """Thin adapter around the Anthropic SDK.

    Not used by tests — test paths always go through `FakeLLMProvider`.
    Requires the ``anthropic`` package and a credential (API key).
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        max_tokens: int = 1024,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        """
        Args:
            model: Claude model ID. Becomes `self.id` (used in cache keys
                and build traces), so switching models invalidates caches.
            max_tokens: Per-call response cap. 1024 is conservative —
                enrichment/labeling responses are short.
            base_url: Optional override for the API base URL. Handy for
                local proxies. Falls back to ``ANTHROPIC_BASE_URL`` env.
            api_key: Explicit key. Falls back to ``ANTHROPIC_API_KEY`` env.
        """
        self.id = model
        self.model = model
        self.max_tokens = max_tokens
        self.base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None  # lazy; populated by `_ensure`

    def _ensure(self):
        """Build the SDK client on first call.

        Deferred so the ``anthropic`` import doesn't happen at config-load
        time — important because `make_llm` may be called just to register
        providers the user doesn't end up exercising.
        """
        if self._client is None:
            import anthropic  # type: ignore
            kwargs = {"api_key": self.api_key}
            # Only pass `base_url` when set — the SDK's default handles
            # the normal case, and passing `None` explicitly errors.
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = anthropic.Anthropic(**kwargs)

    def call(self, prompt: str, *, schema: str = "") -> str:
        """Send a single user-message and return concatenated text blocks.

        ``schema`` is accepted for protocol compatibility but ignored here
        — we don't yet use Anthropic's structured-output features. The
        LLM's text response is expected to already be valid for whatever
        format the prompt requested (JSON, single line, etc.).
        """
        self._ensure()
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        # Responses may contain non-text blocks (e.g. tool use); skip
        # those and join text blocks in arrival order.
        return "".join(block.text for block in msg.content if getattr(block, "type", None) == "text")
