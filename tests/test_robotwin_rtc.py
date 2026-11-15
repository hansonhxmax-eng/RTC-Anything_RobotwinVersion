import sys
from pathlib import Path
import time
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from robotwin_rtc import RTCModelClient


class FakeBackend:
    def __init__(self):
        self.calls = []

    def call(self, *, func_name, **kwargs):
        self.calls.append(func_name)
        if func_name == "get_action":
            time.sleep(0.02)
            return np.arange(12, dtype=np.float32).reshape(6, 2)
        return None


class RTCTests(unittest.TestCase):
    def test_chunk_executes_while_next_inference_runs(self):
        backend = FakeBackend()
        client = RTCModelClient(backend, request_interval=2, first_action_timeout=1)
        try:
            for step in range(5):
                client.call(func_name="update_obs", obs={"step": step})
                action = client.call(func_name="get_action")
                np.testing.assert_array_equal(action, [[2 * step, 2 * step + 1]])
            self.assertGreaterEqual(backend.calls.count("get_action"), 1)
            client.call(func_name="reset")
            client.call(func_name="update_obs", obs={"step": 0})
            np.testing.assert_array_equal(client.call(func_name="get_action"), [[0, 1]])
        finally:
            client.close()

    def test_invalid_action_propagates(self):
        class BadBackend(FakeBackend):
            def call(self, *, func_name, **kwargs):
                if func_name == "get_action":
                    return np.array([float("nan")])
                return None

        client = RTCModelClient(BadBackend(), first_action_timeout=1)
        try:
            client.call(func_name="update_obs", obs={})
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                client.call(func_name="get_action")
        finally:
            client.close()

    def test_xpolicylab_dictionary_actions(self):
        class DictBackend(FakeBackend):
            def call(self, *, func_name, **kwargs):
                if func_name == "get_action":
                    return [{"left_arm_joint_state": np.array([float(i), 1.0]),
                             "left_ee_joint_state": np.array([0.5])} for i in range(4)]
                return None

        client = RTCModelClient(DictBackend(), first_action_timeout=1)
        try:
            client.call(func_name="update_obs", obs={})
            action = client.call(func_name="get_action")
            self.assertIsInstance(action, list)
            np.testing.assert_array_equal(action[0]["left_arm_joint_state"], [0, 1])
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()
