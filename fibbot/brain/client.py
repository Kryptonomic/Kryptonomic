"""Thin wrapper around the Anthropic SDK: the agent loop (tool runner) and structured JSON calls.

`anthropic` is imported lazily so the rest of fibbot works without it. Request shape (see
DESIGN.md): adaptive thinking, effort via output_config, a cached system prompt plus top-level
automatic caching, and server-side refusal fallbacks when cfg.brain.fallbacks is set. Never sent:
budget_tokens, temperature/top_p/top_k, forced tool_choice, assistant prefill.

The agent loop streams (tool_runner(stream=True), get_final_message() per turn), so
cfg.brain.max_tokens can be raised past the SDK's ~21k non-streaming ceiling; thinking counts
toward max_tokens. It mirrors the conversation while it runs, so a transient API failure after
some turns resumes the same conversation (tools that already ran are not run again) instead of
discarding it, and a run cut off by its turn cap or by max_tokens can get one bounded wrap-up
turn (e.g. to write the notebook).
"""
from __future__ import annotations

import json
import logging
import time

log = logging.getLogger(__name__)

RETRY_SLEEP = 5.0                 # seconds before the single retry of a transient structured() failure
RESUME_SLEEPS = (RETRY_SLEEP, 60.0, 180.0)   # backoff before each resume of an agent run
DEFAULT_MAX_RETRIES = 8           # SDK-level retries per request (honours retry-after; brain.max_retries)
SIGNAL_MAX_TOKENS = 16000         # structured() default (brain.signal_max_tokens); thinking counts toward it
NONSTREAMING_MAX_TOKENS = 21000   # the SDK refuses larger non-streaming requests (structured() is one)
# Each fallbacks form has its own beta header: the "default" scalar vs an explicit model list.
FALLBACK_BETAS = {"scalar": "server-side-fallback-2026-07-01", "list": "server-side-fallback-2026-06-01"}
USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
# Error types an SSE `error` event can carry mid-stream (the SDK raises them with HTTP status 200).
TRANSIENT_ERROR_TYPES = frozenset({"overloaded_error", "api_error", "rate_limit_error", "timeout_error"})
AUTH_ERROR_TYPES = frozenset({"authentication_error", "permission_error"})


class BrainUnavailable(RuntimeError):
    """Claude cannot be used: SDK missing, no or rejected credentials, an account that cannot serve
    requests (credits exhausted, model retired), or a persistent API failure."""


def _sdk():
    try:
        import anthropic
    except ImportError as e:
        raise BrainUnavailable("the 'anthropic' package is not installed "
                               "(pip install -r requirements.txt)") from e
    return anthropic


# What the brain uses of the SDK: an older or newer anthropic without these cannot run cycles.
SDK_FEATURES = ("beta_tool", "lib.tools.ToolError", "beta.messages.tool_runner")
INCOMPATIBLE = (ImportError, AttributeError, TypeError)   # what an SDK mismatch raises before any turn


def _incompatible(exc: BaseException) -> BrainUnavailable:
    return BrainUnavailable(f"incompatible anthropic SDK: {type(exc).__name__}: {exc}"
                            " (pip install -r requirements.txt)")


def sdk_missing(sdk, client=None) -> list[str]:
    """The SDK_FEATURES the installed anthropic lacks (client: checked for beta.messages.tool_runner)."""
    missing = []
    if not callable(getattr(sdk, "beta_tool", None)):
        missing.append("beta_tool")
    try:
        from anthropic.lib.tools import ToolError  # noqa: F401
    except ImportError:
        missing.append("lib.tools.ToolError")
    if client is not None:
        beta = getattr(getattr(client, "beta", None), "messages", None)
        if not callable(getattr(beta, "tool_runner", None)):
            missing.append("beta.messages.tool_runner")
    return missing


def _transport_errors() -> tuple:
    """Network errors that can surface raw while a streamed reply is read (not wrapped by the SDK)."""
    try:
        import httpx2
        return (httpx2.TransportError,)
    except ImportError:  # pragma: no cover - httpx2 ships with the SDK
        return ()


def _error_info(exc: BaseException) -> tuple[str, str]:
    """(error type, message) from an API error's JSON body; ("", str(exc)) when it has none."""
    body = getattr(exc, "body", None)
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        return str(err.get("type") or ""), str(err.get("message") or "")
    return "", str(exc)


def _classify(exc: BaseException, sdk) -> str | None:
    """'auth' for credential problems, 'account' for an account or model that cannot serve
    requests (404 model not found, 402 billing, 400 credit balance too low), 'transient' for
    retryable failures, else None (other errors, e.g. a malformed request, propagate)."""
    etype, emsg = _error_info(exc)
    # getattr: an SDK without one of these symbols must not turn every error into AttributeError
    auth = tuple(t for t in (getattr(sdk, "AuthenticationError", None),
                             getattr(sdk, "PermissionDeniedError", None),
                             getattr(sdk, "CredentialsError", None)) if isinstance(t, type))
    if (auth and isinstance(exc, auth)) or etype in AUTH_ERROR_TYPES:
        return "auth"
    if isinstance(exc, sdk.NotFoundError) or etype in ("not_found_error", "billing_error"):
        return "account"
    if isinstance(exc, sdk.APIStatusError) and exc.status_code == 402:
        return "account"
    if isinstance(exc, sdk.BadRequestError) and "credit balance" in f"{emsg} {exc}".lower():
        return "account"
    if isinstance(exc, (sdk.RateLimitError, sdk.APIConnectionError)):
        return "transient"
    if isinstance(exc, sdk.APIStatusError) and (exc.status_code >= 500 or etype in TRANSIENT_ERROR_TYPES):
        return "transient"
    if isinstance(exc, _transport_errors()):
        return "transient"
    return None


def _unavailable(kind: str, exc: BaseException, model: str | None = None) -> BrainUnavailable:
    if kind == "auth":
        return BrainUnavailable(f"Anthropic API rejected the credentials ({type(exc).__name__}: {exc}); "
                                "set ANTHROPIC_API_KEY or run `ant auth login`")
    if kind == "account":
        return BrainUnavailable(f"Anthropic account cannot serve requests: credits exhausted or model "
                                f"{model!r} not available ({type(exc).__name__}: {exc}); top up the "
                                "credits or change brain.model")
    return BrainUnavailable(f"Anthropic API unavailable after retrying ({type(exc).__name__}: {exc})")


def _final_message(item):
    """The turn's message: a streamed turn (stream=True runner) is read to its final message."""
    get = getattr(item, "get_final_message", None)
    return get() if callable(get) else item


def _as_param(message) -> dict:
    """An assistant reply as a message param for the mirrored history (thinking blocks included)."""
    to_param = getattr(message, "to_param", None)
    if callable(to_param):
        return to_param()
    return {"role": "assistant", "content": list(message.content)}


def _with_note(history: list, note: str) -> list | None:
    """history up to its last user message, with `note` as a text block at the end of that
    message (tool_result blocks stay first, as the API requires); None if there is none."""
    msgs = list(history)
    while msgs and msgs[-1].get("role") != "user":
        msgs.pop()
    if not msgs:
        return None
    content = msgs[-1]["content"]
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else list(content)
    msgs[-1] = {**msgs[-1], "content": [*blocks, {"type": "text", "text": note}]}
    return msgs


def _text(message) -> str:
    return "\n".join(b.text for b in message.content if getattr(b, "type", None) == "text").strip()


class Brain:
    """Claude access for the research/review cycles and signal reviews (cfg["brain"] settings).

    `client` may be injected (tests, custom configuration); otherwise `anthropic.Anthropic()` is
    built on first use (max_retries = brain.max_retries, default DEFAULT_MAX_RETRIES) and resolves
    credentials itself (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or an `ant auth login` profile).

    After each structured() call, `last_failure` says why it returned None ("refusal",
    "max_tokens", "no text", "invalid JSON", "not an object"; None after a usable reply) and
    `last_usage` holds the reply's token usage.
    """

    def __init__(self, cfg: dict, client=None):
        self.cfg = cfg
        self.bcfg = cfg["brain"]
        self._client = client
        self._client_injected = client is not None      # a test double: no SDK feature check
        self.last_failure: str | None = None
        self.last_usage: dict = dict.fromkeys(USAGE_KEYS, 0)

    @property
    def client(self):
        """The SDK client; BrainUnavailable if the SDK is missing or no credentials are configured."""
        if self._client is None:
            retries = self.bcfg.get("max_retries")
            c = _sdk().Anthropic(max_retries=int(DEFAULT_MAX_RETRIES if retries is None else retries))
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
            client = self.client
        except BrainUnavailable as e:
            return False, str(e)
        except Exception as e:  # client construction can fail on malformed profiles/config
            return False, f"cannot build an Anthropic client: {type(e).__name__}: {e}"
        if self._client_injected:
            return True, f"model {self.bcfg['model']}"
        sdk = _sdk()
        missing = sdk_missing(sdk, client)
        if missing:
            return False, (f"installed anthropic {getattr(sdk, '__version__', '?')} lacks "
                           f"{' / '.join(missing)}; pip install -r requirements.txt")
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

    def _attempts(self, fn):
        """Run fn(); map SDK errors to BrainUnavailable, retrying a transient failure once.
        Credential and account errors are not retried; other errors propagate unchanged."""
        sdk = _sdk()
        for attempt in (1, 2):
            try:
                return fn()
            except Exception as e:
                kind = _classify(e, sdk)
                if kind is None:
                    raise
                if kind != "transient" or attempt == 2:
                    raise _unavailable(kind, e, self.bcfg.get("model")) from e
                log.warning("transient Anthropic API error, retrying in %.0fs: %s", RETRY_SLEEP, e)
                time.sleep(RETRY_SLEEP)

    def run_agent(self, system: str, prompt: str, tools: list, max_iterations: int | None = None,
                  effort: str | None = None, wrap_up: str | None = None,
                  wrap_up_tools: tuple[str, ...] = (), stop=None) -> dict:
        """Run the streaming tool-runner loop until Claude stops (or max_iterations replies).

        Returns {"text", "stop_reason", "turns", "usage": {input/output/cache_read/cache_creation
        token sums}, "refused", "resumes", "wrapped_up"} (+ "stopped_by" after a stop). text is the
        final message's text ("" when refused); stop_reason is that of the main loop's last reply
        ("tool_use" = the turn cap), or "stopped".

        stop: optional callable checked after every turn (once its tool calls ran); a non-empty
        string it returns (e.g. "the bot is shutting down") ends the run there: the runner is
        closed, nothing is resumed, and the run is treated like the turn cap (wrap-up below),
        with stop_reason "stopped" and "stopped_by" = that reason.

        A transient API failure is retried by resuming the mirrored conversation (up to
        len(RESUME_SLEEPS) times, with that backoff): completed tool calls keep their results and
        are not run again, and turns and usage stay cumulative. Afterwards, and for credential or
        account errors at once, BrainUnavailable is raised.

        wrap_up: when the loop ends on the turn cap, a stop or max_tokens, one more reply is requested
        with this text (its "{reason}" filled in) appended to the conversation (a max_tokens-cut
        reply is dropped first). Only the tools named in wrap_up_tools are run from that reply, and
        its text becomes "text". The wrap-up is best effort: a failing one is logged and ignored.
        """
        kw = self._common(system, prompt, self.bcfg["max_tokens"],
                          {"effort": effort or self.bcfg["effort"]})
        kw.update(tools=tools, cache_control={"type": "ephemeral"}, stream=True)
        history = list(kw.pop("messages"))
        cap = int(max_iterations or self.bcfg["max_iterations"])
        st: dict = {"turns": 0, "last": None, "usage": dict.fromkeys(USAGE_KEYS, 0), "resumes": 0,
                    "stopped": ""}

        def should_stop() -> str:
            try:
                return str(stop() or "") if stop is not None else ""
            except Exception as e:  # a broken stop callable must not kill a paid cycle
                log.warning("stop callable failed: %s", e)
                return ""

        def reply(message) -> None:
            st["turns"] += 1
            usage = getattr(message, "usage", None)
            for k in USAGE_KEYS:
                st["usage"][k] += int(getattr(usage, k, 0) or 0)

        def drive() -> None:
            try:
                runner = self.client.beta.messages.tool_runner(
                    **kw, messages=list(history), max_iterations=cap - st["turns"])
                items = iter(runner)
                item = next(items, None)
            except INCOMPATIBLE as e:
                if st["turns"] == 0:      # the SDK cannot run this request at all
                    raise _incompatible(e) from e
                raise
            while item is not None:
                message = _final_message(item)
                reply(message)
                st["last"] = message
                history.append(_as_param(message))
                if getattr(message, "stop_reason", None) == "tool_use":
                    # Runs the turn's tools now (the runner reuses this cached response), so the
                    # mirrored history holds the results a resumed conversation must continue from.
                    gen = getattr(runner, "generate_tool_call_response", None)
                    response = gen() if callable(gen) else None
                    if response is not None:
                        history.append(response)
                    why = should_stop()
                    if why:
                        st["stopped"] = why
                        close = getattr(runner, "close", None)
                        if callable(close):
                            close()
                        return
                item = next(items, None)

        sdk = _sdk()
        while True:
            try:
                drive()
                break
            except Exception as e:
                kind = _classify(e, sdk)
                if kind is None:
                    raise
                if kind != "transient" or st["resumes"] >= len(RESUME_SLEEPS):
                    raise _unavailable(kind, e, self.bcfg.get("model")) from e
                if st["turns"] >= cap:   # nothing left to resume: the cap was reached
                    break
                why = should_stop() if st["turns"] else ""
                if why:                  # never resume once the run must end
                    st["stopped"] = why
                    break
                delay = RESUME_SLEEPS[st["resumes"]]
                st["resumes"] += 1
                log.warning("transient Anthropic API error after %d turns; resuming the conversation "
                            "in %.0fs: %s", st["turns"], delay, e)
                time.sleep(delay)

        last = st["last"]
        stop = getattr(last, "stop_reason", None)
        refused = stop == "refusal"
        text = "" if refused or last is None else _text(last)
        wrapped = False
        reason = None
        if st["stopped"]:
            reason, stop = st["stopped"], "stopped"
        elif stop == "tool_use" and st["turns"] >= cap:
            reason = f"you have used all {cap} model turns of this cycle"
        elif stop == "max_tokens":
            reason = ("your last reply was cut off at the output token limit (max_tokens) and was "
                      "discarded; its tool calls did not run")
        if wrap_up and reason:
            out = self._wrap_up(kw, history, wrap_up.replace("{reason}", reason), tools,
                                wrap_up_tools, reply)
            if out is not None:
                wrapped, text = True, out
        out = {"text": text, "stop_reason": stop, "turns": st["turns"], "usage": st["usage"],
               "refused": refused, "resumes": st["resumes"], "wrapped_up": wrapped}
        if st["stopped"]:
            out["stopped_by"] = st["stopped"]
        return out

    def _wrap_up(self, kw: dict, history: list, note: str, tools: list, names: tuple,
                 reply) -> str | None:
        """One extra reply after a cut-off run (see run_agent); its text, or None if it failed."""
        msgs = _with_note(history, note)
        if msgs is None:
            return None

        def once():
            runner = self.client.beta.messages.tool_runner(**kw, messages=msgs, max_iterations=1)
            try:
                for item in runner:      # stop at the first reply: the runner must not run its tools
                    return _final_message(item)
            finally:
                close = getattr(runner, "close", None)
                if callable(close):
                    close()
            return None

        try:
            message = self._attempts(once)
        except Exception as e:  # best effort: the cycle's own results are already recorded
            log.warning("wrap-up turn failed: %s: %s", type(e).__name__, e)
            return None
        if message is None:
            return None
        reply(message)
        if getattr(message, "stop_reason", None) == "refusal":
            return None
        by_name = {getattr(t, "name", None): t for t in tools}
        for b in message.content:
            if getattr(b, "type", None) == "tool_use" and b.name in names and b.name in by_name:
                try:
                    by_name[b.name].call(b.input)
                except Exception as e:  # e.g. ToolError: the tool recorded the failure itself
                    log.warning("wrap-up tool %s failed: %s", b.name, e)
        return _text(message)

    def structured(self, system: str, prompt: str, schema: dict, effort: str | None = None,
                   max_tokens: int | None = None) -> dict | None:
        """One JSON-schema-constrained reply parsed into a dict; None on refusal or unusable output
        (self.last_failure says which). max_tokens defaults to brain.signal_max_tokens
        (SIGNAL_MAX_TOKENS) and is capped at NONSTREAMING_MAX_TOKENS; thinking counts toward it,
        so a reply cut off at max_tokens is reported as such, not parsed."""
        if max_tokens is None:
            max_tokens = self.bcfg.get("signal_max_tokens") or SIGNAL_MAX_TOKENS
        max_tokens = min(int(max_tokens), NONSTREAMING_MAX_TOKENS)
        fmt = {"type": "json_schema", "schema": schema}
        kw = self._common(system, prompt, max_tokens,
                          {"effort": effort or self.bcfg["signal_effort"], "format": fmt})
        self.last_failure, self.last_usage = None, dict.fromkeys(USAGE_KEYS, 0)
        try:
            msg = self._attempts(lambda: self.client.beta.messages.create(**kw))
        except (AttributeError, TypeError) as e:     # e.g. an SDK without beta / a kwarg
            raise _incompatible(e) from e
        usage = getattr(msg, "usage", None)
        self.last_usage = {k: int(getattr(usage, k, 0) or 0) for k in USAGE_KEYS}
        if msg.stop_reason == "refusal":
            self.last_failure = "refusal"
            return None
        if msg.stop_reason == "max_tokens":
            self.last_failure = "max_tokens"
            log.warning("structured reply was cut off at max_tokens=%d (thinking counts toward it); "
                        "treating it as a failed attempt", max_tokens)
            return None
        text = next((b.text for b in msg.content if getattr(b, "type", None) == "text"), None)
        if text is None:
            self.last_failure = "no text"
            log.warning("structured reply had no text block (stop_reason=%s)", msg.stop_reason)
            return None
        try:
            out = json.loads(text)
        except json.JSONDecodeError:
            self.last_failure = "invalid JSON"
            log.warning("structured reply was not valid JSON (stop_reason=%s)", msg.stop_reason)
            return None
        if not isinstance(out, dict):
            self.last_failure = "not an object"
            return None
        return out
