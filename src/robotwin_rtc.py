"""Asynchronous action-chunk adapter for RoboTwin's XPolicyLab evaluator.

The simulator owns control time: one get_action call corresponds to one
take_action call. Inference runs on a separate thread while the simulator steps.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from utils.real_time_chunking import RealTimeChunkingBuffer


class RTCModelClient:
    def __init__(self, backend: Any, *, request_interval: int = 8,
                 first_action_timeout: float = 60.0, exp_weight_factor: float = 0.3):
        if request_interval < 1 or first_action_timeout <= 0:
            raise ValueError("request_interval and first_action_timeout must be positive")
        self.backend = backend
        self.request_interval = request_interval
        self.first_action_timeout = first_action_timeout
        self.exp_weight_factor = exp_weight_factor
        self._robotwin_protocol = getattr(backend, "_robotwin_protocol", None)
        self._condition = threading.Condition()
        self._buffer = None
        self._step = 0
        self._last_request = -request_interval
        self._observation = None
        self._worker = None
        self._error = None
        self._generation = 0
        self._closed = False
        self._action_layout = None

    def call(self, *, func_name: str, **kwargs):
        if func_name == "update_obs":
            with self._condition:
                self._observation = kwargs["obs"]
                self._start_request_locked()
            return None
        if func_name == "get_action":
            return self._next_action()
        if func_name == "reset":
            self._reset()
            return None
        return self.backend.call(func_name=func_name, **kwargs)

    def _start_request_locked(self):
        if (self._closed or self._observation is None or self._worker is not None
                or self._step - self._last_request < self.request_interval):
            return
        cursor = self._step
        observation = self._observation
        generation = self._generation
        self._last_request = cursor
        self._worker = threading.Thread(
            target=self._infer, args=(cursor, observation, generation), daemon=True
        )
        self._worker.start()

    def _infer(self, cursor, observation, generation):
        try:
            self.backend.call(func_name="update_obs", obs=observation)
            result = self.backend.call(func_name="get_action")
            if isinstance(result, dict):
                result = result.get("actions", result)
            chunk, layout = self._pack_actions(result)
            if chunk.ndim != 2 or not chunk.shape[0] or not chunk.shape[1] or not np.isfinite(chunk).all():
                raise ValueError("policy must return a finite numeric [horizon, action_dim] chunk")
            with self._condition:
                if generation == self._generation and not self._closed:
                    if self._action_layout is None:
                        self._action_layout = layout
                    if layout != self._action_layout:
                        raise ValueError("policy action layout changed within an episode")
                    if self._buffer is None:
                        self._buffer = RealTimeChunkingBuffer(chunk.shape[0], self.exp_weight_factor)
                    if chunk.shape[0] != self._buffer.chunk_size:
                        raise ValueError("policy action horizon changed within an episode")
                    self._buffer.enqueue(chunk, cursor)
        except Exception as exc:
            with self._condition:
                if generation == self._generation:
                    self._error = exc
        finally:
            with self._condition:
                self._worker = None
                self._condition.notify_all()

    def _next_action(self):
        deadline = time.monotonic() + self.first_action_timeout
        with self._condition:
            while True:
                if self._closed:
                    raise RuntimeError("RTC client is closed")
                if self._error is not None:
                    raise RuntimeError("policy inference failed") from self._error
                action = None if self._buffer is None else self._buffer.get_action(self._step)
                if action is not None:
                    self._step += 1
                    # RoboTwin's evaluator accepts a one-step action chunk.
                    if self._action_layout is None:
                        return np.asarray(action, dtype=np.float32)[None, :]
                    result = {}
                    offset = 0
                    for key, shape, size in self._action_layout:
                        result[key] = np.asarray(action[offset:offset + size], dtype=np.float32).reshape(shape)
                        offset += size
                    return [result]
                self._start_request_locked()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("no policy action covers the current simulator step")
                self._condition.wait(remaining)

    def _reset(self):
        with self._condition:
            worker = self._worker
        if worker is not None:
            worker.join()
        with self._condition:
            self.backend.call(func_name="reset")
            self._generation += 1
            self._buffer = None
            self._step = 0
            self._last_request = -self.request_interval
            self._observation = None
            self._error = None
            self._action_layout = None

    def close(self):
        with self._condition:
            self._closed = True
            worker = self._worker
            self._condition.notify_all()
        if worker is not None:
            worker.join()
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _pack_actions(result):
        if isinstance(result, np.ndarray):
            return np.asarray(result, dtype=np.float32), None
        if not isinstance(result, (list, tuple)) or not result:
            raise ValueError("policy must return a nonempty action chunk")
        if not isinstance(result[0], dict):
            return np.asarray(result, dtype=np.float32), None
        layout = []
        for key in sorted(result[0]):
            value = np.asarray(result[0][key], dtype=np.float32)
            layout.append((key, value.shape, value.size))
        rows = []
        for action in result:
            if sorted(action) != [key for key, _, _ in layout]:
                raise ValueError("action keys changed within a chunk")
            rows.append(np.concatenate([
                np.asarray(action[key], dtype=np.float32).reshape(-1)
                for key, _, _ in layout
            ]))
        return np.stack(rows), layout
