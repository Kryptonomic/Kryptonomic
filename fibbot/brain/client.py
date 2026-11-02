"""Thin wrapper around the Anthropic SDK: the agent loop (tool runner) and structured JSON calls.

`anthropic` is imported lazily so the rest of fibbot works without it. Request shape (see
DESIGN.md): adaptive thinking, effort via output_config, a cached system prompt plus top-level
automatic caching, and server-side refusal fallbacks when cfg.brain.fallbacks is set. Never sent:
budget_tokens, temperature/top_p/top_k, forced tool_choice, assistant prefill.
"""
from __future__ import annotations

import json
import logging
import time

log = logging.getLogger(__name__)

RETRY_SLEEP = 5.0                 # seconds before the single retry of a transient API failure
# Each fallbacks form has its own beta header: the "default" scalar vs an explicit model list.
FALLBACK_BETAS = {"scalar": "server-side-fallback-2026-07-01", "list": "server-side-fallback-2026-06-01"}
USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


class BrainUnavailable(RuntimeError):
    """Claude cannot be used: SDK missing, no or rejected credentials, or a persistent API failure."""


def _sdk():
    try:
        import anthropic
    except ImportError as e:
        raise BrainUnavailable("the 'anthropic' package is not installed "
                               "(pip install 'anthropic>=2.4.0')") from e
    return anthropic


def _classify(exc: BaseException, sdk) -> str | None:
    """'auth' for credential problems, 'transient' for retryable failures, else None."""
    if isinstance(exc, (sdk.AuthenticationError, sdk.PermissionDeniedError, sdk.CredentialsError)):
        return "auth"
    if isinstance(exc, (sdk.RateLimitError, sdk.APIConnectionError)):
        return "transient"
    if isinstance(exc, sdk.APIStatusError) and exc.status_code >= 500:
        return "transient"
    return None


def _unavailable(kind: str, exc: BaseException) -> BrainUnavailable:
    if kind == "auth":
        return BrainUnavailable(f"Anthropic API rejected the credentials ({type(exc).__name__}: {exc}); "
                                "set ANTHROPIC_API_KEY or run `ant auth login`")
    return BrainUnavailable(f"Anthropic API unavailable after a retry ({type(exc).__name__}: {exc})")


class Brain:
    """Claude access for the research/review cycles and signal reviews (cfg["brain"] settings).

    `client` may be injected (tests, custom configuration); otherwise `anthropic.Anthropic()` is
    built on first use and resolves credentials itself (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or
    an `ant auth login` profile).
    """

    def __init__(self, cfg: dict, client=None):
        self.cfg = cfg
        self.bcfg = cfg["brain"]
        self._client = client

    @property
    def client(self):
        """The SDK client; BrainUnavailable if the SDK is missing or no credentials are configured."""
        if self._client is None:
            c = _sdk().Anthropic()
            if not any(getattr(c, a, None) for a in ("api_key", "auth_token", "credentials")):
                raise BrainUnavailable("no Anthropic credentials found: set ANTHROPIC_API_KEY "
                                       "(or ANTHROPIC_AUTH_TOKEN, or run `ant auth login`)")
            self._client = c
        return self._client

    def available(self) -> tuple[bool, str]:
        """(True, model) when the brain is enabled and a client can be built, else (False, why)."""
        if not self.bcfg.get("enabled", True):
            return False, "brain disabled in config (brain.enabled = false)"
        try:
            self.client
        except BrainUnavailable as e:
            return False, str(e)
        except Exception as e:  # client construction can fail on malformed profiles/config
            return False, f"cannot build an Anthropic client: {type(e).__name__}: {e}"
        return True, f"model {self.bcfg['model']}"

    def _common(self, system: str, prompt: str, max_tokens: int, output_config: dict) -> dict:
        kw = {
            "model": self.bcfg["model"],
            "max_tokens": int(max_tokens),
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "adaptive"},
            "output_config": output_config,
        }
        fb = self.bcfg.get("fallbacks")
        if fb:
            kw["betas"] = [FALLBACK_BETAS["scalar" if isinstance(fb, str) else "list"]]
            kw["fallbacks"] = fb
        return kw

    def _attempts(self, fn, retry_ok=lambda: True):
        """Run fn(); map SDK errors to BrainUnavailable, retrying a transient failure once (only
        while retry_ok() is true, e.g. before any tool ran). Other errors propagate unchanged."""
        sdk = _sdk()
        for attempt in (1, 2):
            try:
                return fn()
            except Exception as e:
                kind = _classify(e, sdk)
                if kind is None:
                    raise
                if kind == "auth" or attempt == 2 or not retry_ok():
                    raise _unavailable(kind, e) from e
                log.warning("transient Anthropic API error, retrying in %.0fs: %s", RETRY_SLEEP, e)
                time.sleep(RETRY_SLEEP)

    def run_agent(self, system: str, prompt: str, tools: list, max_iterations: int | None = None,
                  effort: str | None = None) -> dict:
        """Run the tool-runner loop until Claude stops (or max_iterations replies).

        Returns {"text", "stop_reason", "turns", "usage": {input/output/cache_read/cache_creation
        token sums}, "refused"}. text is the final message's text ("" when refused). A transient
        failure is retried from scratch once, but only if no reply had been received yet (tools may
        already have spent budget); otherwise BrainUnavailable is raised.
        """
        kw = self._common(system, prompt, self.bcfg["max_tokens"],
                          {"effort": effort or self.bcfg["effort"]})
        kw.update(tools=tools, cache_control={"type": "ephemeral"},
                  max_iterations=int(max_iterations or self.bcfg["max_iterations"]))
        state: dict = {}

        def drive() -> None:
            state.update(turns=0, last=None, usage=dict.fromkeys(USAGE_KEYS, 0))
            for message in self.client.beta.messages.tool_runner(**kw):
                state["turns"] += 1
                state["last"] = message
                usage = getattr(message, "usage", None)
                for k in USAGE_KEYS:
                    state["usage"][k] += int(getattr(usage, k, 0) or 0)

        self._attempts(drive, retry_ok=lambda: state.get("turns", 0) == 0)
        last = state["last"]
        stop = getattr(last, "stop_reason", None)
        refused = stop == "refusal"
        text = "" if refused or last is None else "\n".join(
            b.text for b in last.content if getattr(b, "type", None) == "text").strip()
        return {"text": text, "stop_reason": stop, "turns": state["turns"],
                "usage": state["usage"], "refused": refused}

    def structured(self, system: str, prompt: str, schema: dict, effort: str | None = None,
                   max_tokens: int = 4000) -> dict | None:
        """One JSON-schema-constrained reply parsed into a dict; None on refusal or bad output."""
        fmt = {"type": "json_schema", "schema": schema}
        kw = self._common(system, prompt, max_tokens,
                          {"effort": effort or self.bcfg["signal_effort"], "format": fmt})
        msg = self._attempts(lambda: self.client.beta.messages.create(**kw))
        if msg.stop_reason == "refusal":
            return None
        text = next((b.text for b in msg.content if getattr(b, "type", None) == "text"), None)
        try:
            out = json.loads(text) if text is not None else None
        except json.JSONDecodeError:
            log.warning("structured reply was not valid JSON (stop_reason=%s)", msg.stop_reason)
            return None
        return out if isinstance(out, dict) else None
