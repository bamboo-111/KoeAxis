"""Spread Gemini ``generateContent`` requests over several relays.

The pool looks like an OpenAI client (``pool.chat.completions.create(...)``), so the proofread code that
builds OpenAI-style messages runs unchanged on the native Gemini protocol:

* every relay has its own in-flight cap; a request takes a slot on the healthy relay with the most room;
* a request still running after the hedge delay is sent once more to another relay with a free slot, and
  the first success wins (the loser finishes in the background and only then frees its slot);
* a failed request is retried on another relay right away; a relay that keeps failing is rested for a
  cool-down, and one that rejects the key is dropped for the rest of the run.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

from optimizer import gemini_native
from optimizer.exceptions import LLMAuthError

LOGGER = logging.getLogger(__name__)

# Before enough latencies are known, hedge only clearly stuck requests.
DEFAULT_HEDGE_INITIAL_S = 120.0
DEFAULT_HEDGE_MIN_S = 20.0
HEDGE_P90_FACTOR = 1.5
HEDGE_SAMPLE_MIN = 10
DEFAULT_FAILURE_THRESHOLD = 3
DEFAULT_COOLDOWN_S = 60.0


@dataclass(frozen=True)
class GeminiEndpoint:
    name: str
    base_url: str
    api_key: str
    model: str
    max_inflight: int = 4


@dataclass
class _EndpointState:
    endpoint: GeminiEndpoint
    inflight: int = 0
    consecutive_failures: int = 0
    down_until: float = 0.0
    disabled: bool = False
    latencies: deque = field(default_factory=lambda: deque(maxlen=50))
    stats: dict[str, int] = field(default_factory=lambda: {
        "sent": 0, "succeeded": 0, "failed": 0, "hedges_sent": 0, "wins": 0})

    def available(self, now: float) -> bool:
        return not self.disabled and now >= self.down_until and self.inflight < self.endpoint.max_inflight


Sender = Callable[[GeminiEndpoint, dict[str, Any], float], Any]


def _send_native(endpoint: GeminiEndpoint, payload: dict[str, Any], timeout: float) -> Any:
    return gemini_native.generate_content(
        base_url=endpoint.base_url, api_key=endpoint.api_key, model=endpoint.model, payload=payload, timeout=timeout)


class GeminiRelayPool:
    def __init__(
        self,
        endpoints: list[GeminiEndpoint],
        *,
        timeout: float = 240.0,
        require_json: bool = True,
        max_output_tokens: int | None = None,
        hedge_initial_s: float = DEFAULT_HEDGE_INITIAL_S,
        hedge_min_s: float = DEFAULT_HEDGE_MIN_S,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        cooldown_s: float = DEFAULT_COOLDOWN_S,
        sender: Sender | None = None,
    ) -> None:
        if not endpoints:
            raise ValueError("GeminiRelayPool needs at least one endpoint")
        self._states = [_EndpointState(endpoint) for endpoint in endpoints]
        self.timeout = timeout
        self.require_json = require_json
        self.max_output_tokens = max_output_tokens
        self.hedge_initial_s = hedge_initial_s
        self.hedge_min_s = hedge_min_s
        self.failure_threshold = max(1, failure_threshold)
        self.cooldown_s = cooldown_s
        self._send = sender or _send_native
        self._cond = threading.Condition()
        capacity = sum(max(1, e.max_inflight) for e in endpoints)
        self._executor = ThreadPoolExecutor(max_workers=capacity, thread_name_prefix="gemini-pool")
        self._local = threading.local()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    # ------------------------------------------------------------------ public
    @property
    def capacity(self) -> int:
        return sum(state.endpoint.max_inflight for state in self._states)

    def last_served(self) -> dict[str, Any]:
        """Which relay answered the calling thread's latest request, and whether it was hedged."""
        return dict(getattr(self._local, "served", {}) or {})

    def stats(self) -> dict[str, Any]:
        with self._cond:
            return {
                state.endpoint.name: {
                    **state.stats,
                    "model": state.endpoint.model,
                    "max_inflight": state.endpoint.max_inflight,
                    "disabled": state.disabled,
                    "median_latency_s": _median(state.latencies),
                }
                for state in self._states
            }

    def hedge_delay(self) -> float:
        with self._cond:
            samples = sorted(value for state in self._states for value in state.latencies)
        if len(samples) < HEDGE_SAMPLE_MIN:
            return self.hedge_initial_s
        p90 = samples[min(len(samples) - 1, int(len(samples) * 0.9))]
        return max(self.hedge_min_s, p90 * HEDGE_P90_FACTOR)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def create(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int | None = None,  # noqa: ARG002 - thinking counts against it, so the pool decides
        extra_body: dict[str, Any] | None = None,
        model: str | None = None,  # noqa: ARG002 - every relay names its model itself
        **_: Any,
    ) -> Any:
        payload = gemini_native.build_payload(
            messages,
            temperature=temperature,
            require_json=self.require_json,
            disable_thinking=False,
            extra_body=extra_body,
            max_tokens=self.max_output_tokens,
        )
        started = time.monotonic()
        tried: set[str] = set()
        running: dict[Future, tuple[_EndpointState, bool]] = {}
        last_error: Exception | None = None
        hedged = False
        attempts = 0  # failover launches after every running request failed

        state = self._acquire(exclude=set(), block=True)
        running[self._launch(state, payload)] = (state, False)
        tried.add(state.endpoint.name)
        while running:
            timeout = None
            if not hedged:
                timeout = max(0.0, self.hedge_delay() - (time.monotonic() - started))
            done, _ = wait(list(running), timeout=timeout, return_when=FIRST_COMPLETED)
            if not done:
                hedged = True
                backup = self._acquire(exclude=tried, block=False)
                if backup is not None:
                    with self._cond:
                        backup.stats["hedges_sent"] += 1
                    LOGGER.info("Gemini pool: hedging a %.0fs request on %s", time.monotonic() - started,
                                backup.endpoint.name)
                    running[self._launch(backup, payload)] = (backup, True)
                    tried.add(backup.endpoint.name)
                continue
            for future in done:
                served, is_hedge = running.pop(future)
                error = future.exception()
                if error is None:
                    with self._cond:
                        served.stats["wins"] += 1
                    self._local.served = {
                        "endpoint": served.endpoint.name,
                        "model": served.endpoint.model,
                        "hedged": hedged,
                        "won_by_hedge": is_hedge,
                        "elapsed_s": round(time.monotonic() - started, 2),
                    }
                    return future.result()
                last_error = error
            if not running:
                attempts += 1
                if attempts > len(self._states):
                    break
                # Same-relay retries are the caller's job (with back-off); here fail over to a fresh relay,
                # waiting for one of its slots: a busy healthy relay beats giving up on the request.
                retry = self._acquire(exclude=tried, block=True)
                if retry is None:
                    break
                LOGGER.info("Gemini pool: retrying on %s after %s", retry.endpoint.name, last_error)
                running[self._launch(retry, payload)] = (retry, False)
                tried.add(retry.endpoint.name)
                hedged = True  # one retry is enough; do not hedge it again
        if isinstance(last_error, LLMAuthError):
            raise RuntimeError(f"all Gemini relays failed: {last_error}") from last_error
        # "upstream" marks the failure as transient for the callers' back-off retry loops.
        raise RuntimeError(f"all Gemini relays failed (upstream unavailable): {last_error}") from last_error

    # ----------------------------------------------------------------- private
    def _acquire(self, *, exclude: set[str], block: bool) -> _EndpointState | None:
        with self._cond:
            while True:
                now = time.monotonic()
                candidates = [s for s in self._states if s.endpoint.name not in exclude and s.available(now)]
                if candidates:
                    state = min(candidates, key=lambda s: s.inflight / s.endpoint.max_inflight)
                    state.inflight += 1
                    state.stats["sent"] += 1
                    return state
                if not block:
                    return None
                if not any(s.endpoint.name not in exclude and not s.disabled for s in self._states):
                    if exclude:
                        return None
                    raise RuntimeError("every Gemini relay is disabled (authentication failed)")
                resting = [s.down_until - now for s in self._states if not s.disabled and s.down_until > now]
                self._cond.wait(timeout=min(resting) if resting else 1.0)

    def _launch(self, state: _EndpointState, payload: dict[str, Any]) -> Future:
        return self._executor.submit(self._run, state, payload)

    def _run(self, state: _EndpointState, payload: dict[str, Any]) -> Any:
        started = time.monotonic()
        try:
            response = self._send(state.endpoint, payload, self.timeout)
        except Exception as exc:
            with self._cond:
                state.stats["failed"] += 1
                state.consecutive_failures += 1
                if isinstance(exc, LLMAuthError):
                    state.disabled = True
                    LOGGER.warning("Gemini pool: %s rejected the key, dropping it: %s", state.endpoint.name, exc)
                elif state.consecutive_failures >= self.failure_threshold:
                    state.down_until = time.monotonic() + self.cooldown_s
                    LOGGER.warning("Gemini pool: %s failed %d times in a row, resting %.0fs: %s",
                                   state.endpoint.name, state.consecutive_failures, self.cooldown_s, exc)
            raise
        else:
            with self._cond:
                state.stats["succeeded"] += 1
                state.consecutive_failures = 0
                state.latencies.append(time.monotonic() - started)
            return response
        finally:
            with self._cond:
                state.inflight -= 1
                self._cond.notify_all()


def _median(values: deque) -> float | None:
    ordered = sorted(values)
    return round(ordered[len(ordered) // 2], 2) if ordered else None
